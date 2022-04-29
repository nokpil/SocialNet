import time
import os
import pickle

import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy

from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim import Adam, AdamW
from torch.distributed.algorithms.join import Join

import envs
import ppo.core as core
from utils.utils import explained_variance, init_orthogonal, RewardNormalizer
from utils.logger import EpochLogger
from utils.sampler import BatchSampler, BatchSampler_split

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, ensemble_num, agent_num, dim_len, total_trj_len, gamma=0.99, lam=0.95, split=False):
        self.split = split
        self.N = dim_len
        if split:
            self.obs_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len, *obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)  
            self.adv_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)
            self.rew_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)
            self.ret_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)
            self.scr_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.val_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)
            self.logp_buf = np.zeros((ensemble_num, agent_num, self.N, total_trj_len), dtype=np.float32)
        else:
            self.obs_buf = np.zeros((ensemble_num, agent_num, total_trj_len, *obs_dim), dtype=np.float32)
            self.act_buf = np.zeros((ensemble_num, agent_num, total_trj_len, self.N), dtype=np.float32)
            self.adv_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.rew_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.ret_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.scr_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.val_buf = np.zeros((ensemble_num, agent_num, total_trj_len), dtype=np.float32)
            self.logp_buf = np.zeros((ensemble_num, agent_num, total_trj_len, self.N), dtype=np.float32)
        self.reward_normalizer = RewardNormalizer(ensemble_num, agent_num)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.total_trj_len = 0, 0, total_trj_len

    def store(self, obs, act, rew, val, score, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.total_trj_len + 1  # buffer has to have room so you can store
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
        assert self.ptr == self.total_trj_len  # buffer has to be full before you can get
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
    #'''
    def finish_path(self, v):
        v = np.expand_dims(v, axis=-1)
        rews = np.append(self.rew_buf, v, axis=-1)
        vals = np.append(self.val_buf, v, axis=-1)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[..., :-1] + self.gamma * vals[..., 1:] - vals[..., :-1]
        self.adv_buf = core.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        #self.ret_buf = core.discount_cumsum(rews, self.gamma)[..., :-1]
        self.ret_buf = self.adv_buf + self.val_buf
        self.path_start_idx = self.ptr

    '''
    def finish_path(self, v):
        v = np.expand_dims(v, axis=-1)
        first = np.zeros_like(self.rew_buf)
        first[..., 0] = 1.
        rews = self.reward_normalizer(self.rew_buf, first)
        vals = np.append(self.val_buf, v, axis=-1)

        # the next two lines implement GAE-Lambda advantage calculation
        #deltas = rews[..., :-1] + self.gamma * vals[..., 1:] - vals[..., :-1]
        deltas = rews + self.gamma * vals[..., 1:] - vals[..., :-1]
        self.adv_buf = core.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        #self.ret_buf = core.discount_cumsum(rews, self.gamma)[..., :-1]
        self.ret_buf = self.adv_buf + self.val_buf
        self.path_start_idx = self.ptr
    
    #'''

def ppo(
    env_name,
    env_kwargs=dict(),
    actor_critic=core.ActorCritic,
    arch="st",
    ac_kwargs=dict(),
    seed=42,
    ensemble_num=16,
    agent_num=100,
    trj_len=50,
    repeat_len=4,
    epochs=100,
    batch_size=4000,
    test_batch_size=20000,
    gamma=0.99,
    clip_ratio=0.2,
    ent_coef=0.05,
    pi_lr=3e-5,
    vf_lr=1e-4,
    weight_decay=0.00025,
    train_pi_iters=100,
    train_v_iters=100,
    lam=0.96,
    target_kl=0.01,
    norm_type='disc',
    terminate=False,
    logger_kwargs=dict(),
    save_freq=100,
    distributed=True,
    baselines=[],
    env_scheduler_type=None,
    checkpoint=('', ''),
    local_rank=0
):

    if clip_ratio < 0:
        clip_ratio = 100  # effectively no clip
    os.environ['MPLCONFIGDIR'] = "./"
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
    action_type = env.action_type
    extra_type = env.extra_type
    state_correction = env.state_correction
    reward_correction = env.reward_correction
    reward_supply_type = env.reward_supply_type

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

    # Simulated baselines
    baseline_data_dict = {}
    baseline_data_dict['keys'] = ['Ret', 'FinalScore']
    baseline_data_name = f'baseline_{env.graph_type}_N{env.N}K{env.K}NN{env.neighbor_num}'
    try:
        if rank_0:
            print('Use pre-constructed baseline')
        with open(f'./data/baseline_data/{baseline_data_name}.pkl', 'rb') as f:
            baseline_data_dict = pickle.load(f)
    except FileNotFoundError as e:
        print(e)
        
    # Create actor-critic module
    ac = actor_critic(obs_dim, act_dim, arch, **ac_kwargs)
    ac.to(device)
    #ac.apply(init_orthogonal)

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr, weight_decay=weight_decay)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr, weight_decay=weight_decay)

    # Load checkpoint if needed
    if checkpoint[0]:  # not an emtpy string, which menas there is a checkpoint
        if rank_0:
            print(f'checkpoint:{checkpoint[1]}')
        rel_path = f'data/runs/{checkpoint[0]}/{checkpoint[0]}_s{seed}/'
        checkpoint = torch.load(rel_path + f'pyt_save/model{checkpoint[1]}.pth', map_location=lambda storage, loc: storage)
        ac.pi.load_state_dict(checkpoint['pi'])
        ac.v.load_state_dict(checkpoint['v'])
        pi_optimizer.load_state_dict(checkpoint['pi_optimizer'])
        vf_optimizer.load_state_dict(checkpoint['vf_optimizer'])
    del checkpoint
    torch.cuda.empty_cache()
    
    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    if rank_0:
        logger.log("NN architecture: %s" % arch)
        logger.log("\nNumber of parameters: \t pi: %d, \t v: %d\n" % var_counts)

    # Set up experience buffer
    total_trj_len = trj_len * repeat_len
    buf = PPOBuffer(obs_dim, act_dim, ensemble_num, agent_num, dim_len, total_trj_len, gamma, lam, split=True if action_type == 'split' else False)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old, norm_type='disc', epoch=0):
        # Policy loss
        if norm_type == 'disc':
            adv = adv.unsqueeze(-1)
            pi, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            sgn = torch.ones_like(ratio) * torch.sign(adv)
            ratio_clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
            r = torch.prod(sgn * torch.min(ratio * sgn, ratio_clip * sgn), dim=-1).unsqueeze(-1)
            r_mean = r.mean().detach()
            loss_pi = -(r * adv / r_mean).mean()  - ent_coef * (pi.entropy()).mean()
        elif norm_type == 'gene_ent':
            adv = adv.unsqueeze(-1)
            pi, logp = ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean() - ent_coef * (pi.entropy()).mean()
        else:
            raise NotImplementedError

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

    if action_type == 'total':
        train_sampler = BatchSampler(ensemble_num, agent_num, total_trj_len, batch_size, device=device)
        test_sampler = BatchSampler(ensemble_num, agent_num, total_trj_len, test_batch_size, device=device, train=False)
    elif action_type == 'split':
        train_sampler = BatchSampler_split(ensemble_num, agent_num, env.N, total_trj_len, batch_size, device=device)
        test_sampler = BatchSampler_split(ensemble_num, agent_num, env.N, total_trj_len, test_batch_size, device=device, train=False)

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

    def validate_pi(obs, act, adv, logp_old, norm_type='disc', epoch=0):
        loss_pi, loss_ent, approx_kl, ent, clipfrac = 0, 0, 0, 0, 0
        e_const = 1
        adv = adv.unsqueeze(-1)
        with torch.no_grad():
            for batch in test_sampler:
                if norm_type == 'disc':
                    pi, logp = ac.pi(obs[batch], act[batch])
                    ratio = torch.exp(logp - logp_old[batch])
                    sgn = torch.ones_like(ratio) * torch.sign(adv[batch])
                    ratio_clip = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
                    r = torch.prod(sgn * torch.min(ratio * sgn, ratio_clip * sgn), dim=-1).unsqueeze(-1) 
                    r_mean = r.mean().detach()
                    entropy = pi.entropy()
                    loss_pi += -(r * adv[batch] / r_mean).sum() - ent_coef * e_const * (entropy).sum()
                    loss_ent += - (entropy).sum()
                elif norm_type == 'gene_ent':
                    pi, logp = ac.pi(obs[batch], act[batch])
                    ratio = torch.exp(logp - logp_old[batch])
                    entropy = pi.entropy()
                    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv[batch]
                    loss_pi += -(torch.min(ratio * adv[batch], clip_adv)).sum() - ent_coef * e_const * (entropy).sum()
                    loss_ent += - (entropy).sum()
                else:
                    raise NotImplementedError

                # Useful extra info
                approx_kl += (logp_old[batch] - logp).sum().item()
                ent += pi.entropy().sum().item()
                clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
                clipfrac += (
                    torch.as_tensor(clipped, dtype=torch.float32, device=device)
                    .sum()
                    .item()
                )
        
        total_num = test_sampler.size if norm_type in ['agent', 'agent_entP', 'agent_entN'] else test_sampler.size * obs.shape[-1]
        loss_pi = loss_pi / total_num
        loss_ent = loss_ent / total_num
        pi_info = dict(
            kl=approx_kl / total_num,
            ent=ent / total_num,
            cf=clipfrac / total_num,
        )
        return loss_pi, loss_ent, pi_info

    def validate_v(obs, ret):
        loss_v = 0
        with torch.no_grad():
            for batch in test_sampler:
                value_tmp = ac.v(obs[batch])
                loss_v += ((value_tmp - ret[batch]) ** 2).sum()
        total_num = test_sampler.size if norm_type in ['agent', 'agent_entP', 'agent_entN'] else test_sampler.size * obs.shape[-1]
        loss_v = loss_v / total_num
        return loss_v

    # Set up model saving
    if rank_0:
        logger.setup_pytorch_saver({'pi': ac.pi.module, 'v': ac.v.module, 'pi_optimizer': pi_optimizer, 'vf_optimizer': vf_optimizer})
        if env_scheduler_type == 'random':
            env.env_scheduler.initialize_landscape(value=False, fixed=False)
        else:
            env.env_scheduler.initialize_landscape(value=True, fixed=False)

    torch.distributed.barrier()
        
    def update(world_size=10, epoch=0):
        data = buf.get(device)
        obs, act, adv, logp_old, ret = (
            data["obs"],
            data["act"],
            data["adv"],
            data["logp"],
            data["ret"],
        )
        pi_l_old, ent_l_old, pi_info_old = validate_pi(obs, act, adv, logp_old, norm_type, epoch)
        pi_l_old = pi_l_old.item()
        v_l_old = validate_v(obs, ret).item()

        # Train policy with multiple steps of sgd
        with Join([ac.pi]):
            for i in range(train_pi_iters):
                pi_optimizer.zero_grad()
                batch = next(train_sampler)
                loss_pi, pi_info = compute_loss_pi(
                    obs[batch], act[batch], adv[batch], logp_old[batch], norm_type, epoch
                )
                kl = pi_info["kl"]
                loss_pi.backward()
                pi_optimizer.step()

                if kl > 1.5 * target_kl:
                    print(f"Early stopping at step {i} due to reaching max kl at rank {local_rank}.")
                    #logger.log("Early stopping at step %d due to reaching max kl." % i)
                    break  # currently not suitable for DDP, ac.pi.join does not work.

        if rank_0:
            logger.store(StopIter=i)
        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            batch = next(train_sampler)
            loss_v = compute_loss_v(obs[batch], ret[batch])
            loss_v.backward()
            vf_optimizer.step()

        loss_pi, _, pi_info = validate_pi(obs, act, adv, logp_old, norm_type, epoch)
        loss_pi = loss_pi.item()
        loss_v = validate_v(obs, ret)
        loss_v = loss_v.item()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        if rank_0:
            logger.store(
                LossPi=pi_l_old,
                LossEnt=ent_l_old,
                LossV=v_l_old,
                KL=kl,
                Entropy=ent,
                ClipFrac=cf,
                DeltaLossPi=(loss_pi - pi_l_old),
                DeltaLossV=(loss_v - v_l_old)
            )


    # Prepare for interaction with environment
    start_time = time.time()
    best_ep_ret = -np.inf

    # Main loop: collect experience in env and update/log each epoch
    
    scr_buf_figure = np.zeros((ensemble_num, agent_num, trj_len * repeat_len), dtype=np.float32)  # scr_buf for continuous figure
    for epoch in range(epochs):
        ep_ret, ep_len = 0, 0
        o, _ = env.reset()
        for rep in range(repeat_len):
            for t in range(trj_len):
                epoch_ended = (t == trj_len - 1) and (rep == repeat_len - 1)
                a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32, device=device))
                next_o, r, s = env.step(a)
                ep_ret += r
                ep_len += 1
                
                # save and log
                scr_buf_figure[..., int(rep * trj_len + t)] = s
                if reward_supply_type == 'full':
                    buf.store(o, a, r, v, s, logp)
                else:
                    if epoch_ended:
                        if reward_supply_type == 'final':
                            buf.store(o, a, r * total_trj_len, v, s, logp)
                        elif reward_supply_type == 'finalmean':
                            buf.store(o, a, ep_ret, v, s, logp)
                        else:
                            raise NotImplementedError
                    else:
                        buf.store(o, a, 0, v, s, logp)

                if rank_0:
                    logger.store(VVals=v)

                # Update obs (critical!)
                o = next_o

                if epoch_ended:
                    a, v, logp = ac.step(
                        torch.as_tensor(o, dtype=torch.float32, device=device)
                    )
                    _, _, s = env.step(a)
                    if terminate:
                        buf.finish_path(np.zeros_like(v))
                    else:
                        buf.finish_path(v)
            if not epoch_ended:
                o, _ = env.reset(states=env.states)
            
        update(world_size=env.env_scheduler.NGPU, epoch=epoch)
            
        # Save model
        if rank_0:
            logger.store(Ret=ep_ret / ep_len, EpLen=ep_len, FinalScore=np.mean(s))
            best_model = (best_ep_ret < np.mean(ep_ret))
            if best_model:
                best_ep_ret = np.mean(ep_ret)
                print(f'best model saving : Epoch {epoch}')
                logger.save_state(itr=epoch)
            elif epoch % save_freq == 0:
                print(f'regular model saving : Epoch {epoch}')
                logger.save_state(itr=epoch)

        # Perform PPO update!
        if rank_0:
            env.env_scheduler.step(epoch + 1)
        
        ep_ret, ep_len = 0, 0
        '''
        if rank_0:
            logger.store(
                VarianceV=explained_variance(buf.val_buf, buf.ret_buf)
            )
        '''
        
        # Log info about epoch
        if rank_0:
            logger.log_tabular("Epoch", epoch)
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("Ret", with_min_and_max=True)
            logger.log_tabular("FinalScore", average_only=True)
            logger.log_tabular("VVals", with_min_and_max=True)
            logger.log_tabular("TotalEnvInteracts", (epoch + 1) * total_trj_len)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossEnt", average_only=True)
            logger.log_tabular("LossV", average_only=True)
            #logger.log_tabular("VarianceV", average_only=True)
            logger.log_tabular("DeltaLossPi", average_only=True)
            logger.log_tabular("DeltaLossV", average_only=True)
            logger.log_tabular("Entropy", average_only=True)
            logger.log_tabular("KL", average_only=True)
            logger.log_tabular("ClipFrac", average_only=True)
            logger.log_tabular("StopIter", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()

            # Figure drawing
            draw_figure = True
            if draw_figure:
                fig = plt.figure(figsize=(4, 4), dpi=150)
                ax = fig.add_subplot(111)
                if env.rescale:
                    rescale_const = 1.
                else:
                    rescale_const = env.reward_constant

                if baselines:
                    for baseline_name in baselines:
                        x = baseline_data_dict[baseline_name]['scr_buf'] / rescale_const
                        avg_pf = np.mean(x, axis=tuple(range(0, len(x.shape) - 1)))
                        ax.plot(np.arange(x.shape[-1]), avg_pf, label=baseline_name)

                x = scr_buf_figure
                avg_pf = np.mean(x, axis=tuple(range(0, len(x.shape) - 1)))
                ax.plot(np.arange(x.shape[-1]), avg_pf, label='RL')
                ax.set_xlabel('Time')
                ax.set_ylabel('Average Performance')
                ax.legend()
                logger.writer.add_figure('Average Performance', fig, global_step=epoch)
                print('finish')
    if rank_0:
        logger.close()
