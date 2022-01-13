import time

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim import AdamW
from copy import deepcopy

import envs
import ppo.core as core
from utils.logger import EpochLogger, setup_logger_kwargs
from utils.sampler import BatchSampler, BatchSampler_split

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, ensemble_num, agent_num, dim_len, trj_len, gamma=0.99, lam=0.95, split=False):
        self.split = split
        self.N = dim_len
        if split:
            self.obs_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len, *obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)  
            self.adv_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)
            self.rew_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)
            self.ret_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)
            self.scr_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.val_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)
            self.logp_buf = np.zeros((ensemble_num, agent_num, self.N, trj_len), dtype=np.float32)
        else:
            self.obs_buf = np.zeros((ensemble_num, agent_num, trj_len, *obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((ensemble_num, agent_num, trj_len, self.N), dtype=np.float32)  
            self.adv_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.rew_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.ret_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.scr_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.val_buf = np.zeros((ensemble_num, agent_num, trj_len), dtype=np.float32)
            self.logp_buf = np.zeros((ensemble_num, agent_num, trj_len, self.N), dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.trj_len = 0, 0, trj_len

    def store(self, obs, act, rew, val, score, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.trj_len + 1  # buffer has to have room so you can store
        if self.split:
            self.obs_buf[:, :, :, self.ptr] = obs
            self.act_buf[:, :, :, self.ptr] = act
            self.rew_buf[:, :, :, self.ptr] = rew
            self.scr_buf[:, :, self.ptr] = score
            self.val_buf[:, :, :, self.ptr] = val
            self.logp_buf[:, :, :, self.ptr] = logp
        else:
            self.obs_buf[:, :, self.ptr] = obs
            self.act_buf[:, :, self.ptr] = act
            self.rew_buf[:, :, self.ptr] = rew
            self.scr_buf[:, :, self.ptr] = score
            self.val_buf[:, :, self.ptr] = val
            self.logp_buf[:, :, self.ptr] = logp
        self.ptr += 1

    def get(self, device="cpu"):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.trj_len  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
            ret=self.ret_buf,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in data.items()
        }

    def finish_path(self, v):
        v = np.expand_dims(v, axis=-1)
        rews = np.append(self.rew_buf, v, axis=-1)
        vals = np.append(self.val_buf, v, axis=-1)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[..., :-1] + self.gamma * vals[..., 1:] - vals[..., :-1]
        self.adv_buf = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf = core.discount_cumsum(rews, self.gamma)[..., :-1]

        self.path_start_idx = self.ptr

def ppo(
    env_name,
    env_kwargs=dict(),
    actor_critic=core.ActorCritic,
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

    # Instantiate environment
    if action_type == 'total':
        obs_dim = (env.neighbor_num + 1, env.N + 2)  # (3+1, 15+2)
        act_dim = env.action_space.n
        dim_len = env.N
    elif action_type == 'split':
        obs_dim = (env.neighbor_num + 1, 1 + 2)
        act_dim = (2,)
        dim_len = env.N

    # Simulated baselines
    baseline_data_dict = {}
    baseline_data_dict['keys'] = ['Ret', 'FinalScore']
    if rank_0 and baselines:
        env_base = envs.__dict__[env_name](**env_kwargs)
        print("Baseline initiated")
        for baseline_name in baselines:
            print(f"Baseline : {baseline_name}")
            baseline_data = {}
            test_ensemble_num = 128
            ac_base = core.__dict__[baseline_name](env_base, action_type)
            scr_buf = np.zeros((test_ensemble_num, agent_num, trj_len), dtype=np.float32)

            o, ep_ret, ep_len = env_base.reset(E=test_ensemble_num), 0, 0
            for t in range(trj_len):
                a = ac_base.step(o)

                next_o, r, s = env_base.step(a)
                ep_ret += r
                ep_len += 1
                scr_buf[..., t] = s
                o = next_o

            baseline_data['Ret'] = np.mean(ep_ret / ep_len)
            baseline_data['FinalScore'] = np.mean(s)
            baseline_data['scr_buf'] = scr_buf
            baseline_data_dict[baseline_name] = baseline_data
        logger.set_baseline(baseline_data_dict)
        print("Baseline finished")

    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, arch, **ac_kwargs)
    ac.to(device)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    if rank_0:
        logger.log("NN architecture: %s" % arch)
        logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    buf = PPOBuffer(obs_dim, act_dim, ensemble_num, agent_num, dim_len, trj_len, gamma, lam, split=True if action_type == 'split' else False)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        adv = adv.unsqueeze(-1)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = (
            torch.as_tensor(clipped, dtype=torch.float32, device=device).mean().item()
        )
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(obs, ret):
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = AdamW(ac.pi.parameters(), lr=pi_lr, weight_decay=weight_decay)
    vf_optimizer = AdamW(ac.v.parameters(), lr=vf_lr, weight_decay=weight_decay)

    
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
    
    ac.pi = Parallel(ac.pi, **parallel_args)
    ac.v = Parallel(ac.v, **parallel_args)

    def validate_pi(obs, act, adv, logp_old):
        loss_pi, approx_kl, ent, clipfrac = 0, 0, 0, 0
        adv = adv.unsqueeze(-1)
        with torch.no_grad():
            for batch in test_sampler:
                pi, logp = ac.pi(obs[batch], act[batch])
                ratio = torch.exp(logp - logp_old[batch])
                clip_adv = (
                    torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[batch]
                )
                loss_pi += -(torch.min(ratio * adv[batch], clip_adv)).sum()

                # Useful extra info
                approx_kl += (logp_old[batch] - logp).sum().item()
                ent += pi.entropy().sum().item()
                clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                clipfrac += (
                    torch.as_tensor(clipped, dtype=torch.float32, device=device)
                    .sum()
                    .item()
                )

        loss_pi = loss_pi / test_sampler.size
        pi_info = dict(
            kl=approx_kl / test_sampler.size,
            ent=ent / test_sampler.size,
            cf=clipfrac / test_sampler.size,
        )
        return loss_pi, pi_info

    def validate_v(obs, ret):
        loss_v = 0
        with torch.no_grad():
            for batch in test_sampler:
                loss_v += ((ac.v(obs[batch]) - ret[batch]) ** 2).sum()
        loss_v = loss_v / test_sampler.size
        return loss_v
    
    #def validate_strategy():


    # Set up model saving
    if rank_0:
        logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get(device)
        obs, act, adv, logp_old, ret = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logp"],
            data["ret"],
        )
        pi_l_old, pi_info_old = validate_pi(obs, act, adv, logp_old)
        pi_l_old = pi_l_old.item()
        v_l_old = validate_v(obs, ret).item()

        # Train policy with multiple steps of sgd
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            batch = next(train_sampler)
            loss_pi, pi_info = compute_loss_pi(
                obs[batch], act[batch], adv[batch], logp_old[batch]
            )
            kl = pi_info["kl"]
            if kl > 1.5 * target_kl:
                if rank_0:
                    logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            pi_optimizer.step()
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
    o, ep_ret, ep_len = env.reset(), 0, 0
    best_ep_ret = -np.inf

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
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
        if rank_0:
            logger.store(Ret=ep_ret / ep_len, EpLen=ep_len, FinalScore=np.mean(s))
            best_model = (best_ep_ret < np.mean(ep_ret) / ep_len)

            if best_model:
                best_ep_ret = np.mean(ep_ret)
                print(f'best model : saving at Epoch {epoch}')
                logger.save_state({"env": env}, None)

        # Perform PPO update!
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
    if rank_0:
        logger.close()
