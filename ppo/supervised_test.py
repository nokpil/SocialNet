import time
import os
import pickle

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim import AdamW
from copy import deepcopy

import envs
import ppo.core as core
from utils.logger import EpochLogger
from utils.sampler import BatchSampler, BatchSampler_split


def supervised_test(
    env_name,
    env_kwargs=dict(),
    SocialActor=core.SocialActor,
    arch="st",
    ac_kwargs=dict(),
    seed=42,
    ensemble_num=16,
    agent_num=100,
    trj_len=100,
    epochs=100,
    batch_size=4000,
    test_batch_size=20000,
    gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-5,
    vf_lr=1e-4,
    weight_decay=0.00025,
    train_pi_iters=100,
    train_v_iters=100,
    lam=0.96,
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
    distributed=True,
    baselines=[],
    local_rank=0
):
    mpl.use('Agg')
    rank_0 = False
    device = local_rank if distributed else 'cpu'
    n_GPUs = 1  # Can be modified according to local gpu settings
    # Set up logger and save configuration
    if local_rank == 0:
        rank_0 = True
        print('Logger initiated')
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = envs.__dict__[env_name](**env_kwargs)
    action_type = env_kwargs['action_type']
    extra_type = env_kwargs['extra_type']

    # Instantiate environment
    extra_num = len(extra_type)
    if action_type == 'total':
        obs_dim = (env.neighbor_num + 1, env.N + extra_num)
        act_dim = env.action_space.n
        dim_len = env.N
    elif action_type == 'split':
        obs_dim = (env.neighbor_num + 1, 1 + extra_num)
        act_dim = (2,)
        dim_len = env.N
    else:
        raise NotImplementedError

    # Create actor-critic module
    pi = SocialActor(obs_dim, act_dim, arch, **ac_kwargs)
    pi.to(device)

    # Set up function for computing value loss

    def compute_loss_supervised(action, answer):
        return ((action - answer) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = AdamW(pi.parameters(), lr=pi_lr, weight_decay=weight_decay)

    if action_type == 'total':
        train_sampler = BatchSampler(ensemble_num, agent_num, trj_len, batch_size, device=device)
        test_sampler = BatchSampler(ensemble_num, agent_num, trj_len, test_batch_size, device=device, train=False)
    elif action_type == 'split':
        train_sampler = BatchSampler_split(ensemble_num, agent_num, env.N, trj_len, batch_size, device=device)
        test_sampler = BatchSampler_split(ensemble_num, agent_num, env.N, trj_len, test_batch_size, device=device, train=False)

    # Parellelize
    
    if distributed:
        Parallel = DistributedDataParallel
        parallel_args = {
            "device_ids": [local_rank],
            "output_device": local_rank,
        }
    else:
        Parallel = DataParallel
        parallel_args = {
            'device_ids': list(range(n_GPUs)),
            'output_device': 0
        }
    
    pi = Parallel(pi, **parallel_args)
    
    # Set up model saving
    if rank_0:
        logger.setup_pytorch_saver(pi)
        env.env_scheduler.initialize_landscape(fixed=True)

    torch.distributed.barrier()

    train_sampler = torch.utils.data.DistributedSampler(train_data)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    test_sampler = torch.utils.data.DistributedSampler(test_data)
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        sampler=test_sampler,
    )
        
    def update():
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            batch = next(train_sampler)
            loss_pi = compute_loss_supervised(
                obs[batch], act[batch], adv[batch], logp_old[batch]
            )

            loss_pi.backward()
            pi_optimizer.step()
    
        #print(f'here : {local_rank}')
        #torch.distributed.barrier()
        #print(f'passed : {local_rank}')
        
        if rank_0:
            logger.store(StopIter=i)
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            batch = next(train_sampler)
            loss_v = compute_loss_v(obs[batch], ret[batch])
            loss_v.backward()
            vf_optimizer.step()

        loss_pi, pi_info = validate_pi(obs, act, adv, logp_old)
        loss_pi = loss_pi.item()
        loss_v = validate_v(obs, ret).item()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        if rank_0:
            logger.store(
                LossPi=pi_l_old,
                LossV=v_l_old,
                KL=kl,
                Entropy=ent,
                ClipFrac=cf,
                DeltaLossPi=(loss_pi - pi_l_old),
                DeltaLossV=(loss_v - v_l_old),
            )

    # Prepare for interaction with environment
    start_time = time.time()
    best_ep_ret = -np.inf

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        o, ep_ret, ep_len = env.reset(), 0, 0
        for t in range(trj_len):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))

            next_o, r, s = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            buf.store(o, a, r, v, s, logp)
            if rank_0:
                logger.store(VVals=v)

            # Update obs (critical!)
            o = next_o
            epoch_ended = t == trj_len - 1

            if epoch_ended:
                a, v, logp = ac.step(
                    torch.as_tensor(o, dtype=torch.float32, device=device)
                )
                _, _, s = env.step(a)
                buf.finish_path(v)

        # Save model
        if rank_0 and (epoch % save_freq) == 0:
            logger.store(Ret=ep_ret / ep_len, EpLen=ep_len, FinalScore=np.mean(s))
            best_model = (best_ep_ret < np.mean(ep_ret))
            if best_model:
                best_ep_ret = np.mean(ep_ret)
                print(f'best model : saving at Epoch {epoch}')
                logger.save_state({"env": env}, None)

        # Perform PPO update!
        if rank_0:
            env.env_scheduler.step(epoch + 1)
        update()
        ep_ret, ep_len = 0, 0

        # Log info about epoch
        if rank_0:
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("Ret", with_min_and_max=True)
            logger.log_tabular("FinalScore", average_only=True)
            logger.log_tabular("VVals", with_min_and_max=True)
            logger.log_tabular("TotalEnvInteracts", (epoch + 1) * trj_len)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossV", average_only=True)
            logger.log_tabular("DeltaLossPi", average_only=True)
            logger.log_tabular("DeltaLossV", average_only=True)
            logger.log_tabular("Entropy", average_only=True)
            logger.log_tabular("KL", average_only=True)
            logger.log_tabular("ClipFrac", average_only=True)
            logger.log_tabular("StopIter", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()

            # Figure drawing
            fig = plt.figure(figsize=(4,4), dpi=150)
            ax = fig.add_subplot(111)

            if baselines:
                for baseline_name in baselines:
                    x = baseline_data_dict[baseline_name]['scr_buf']
                    avg_pf = np.mean(x, axis=tuple(range(0, len(x.shape) - 1)))
                    ax.plot(np.arange(x.shape[-1]), avg_pf, label=baseline_name)

            x = buf.scr_buf
            avg_pf = np.mean(x, axis=tuple(range(0, len(x.shape) - 1)))
            ax.plot(np.arange(x.shape[-1]), avg_pf, label='RL')
            ax.set_xlabel('Time')
            ax.set_ylabel('Average Performance')
            ax.legend()
            logger.writer.add_figure('Average Performance', fig, global_step=epoch)
            print('finish')
    if rank_0:
        logger.close()
