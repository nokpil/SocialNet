import numpy as np
import networkx as nx
import argparse

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from utils.utils import max_mean_clustering_network
import pickle

import ppo.core as core
from ppo.ppo import PPOBuffer
import envs
import json

parser = argparse.ArgumentParser(description="Running reinforcement learning of SocialLearning model.")
parser.add_argument("-name", type=str, default="", help="Experiment name")
parser.add_argument("-nametest", type=str, default="", help="Experiment name for test")
parser.add_argument("-env", type=int, default='50', help="Env num")
parser.add_argument("-ensemble", type=int, default='100', help="Ensemble num")
args = parser.parse_args()


def load_model(exp_name, epoch):

    # rel_path = f'data/runs/ds_complete_indv_raw_random_SIR_N10K3NN3_new_rand/{exp_name}/{exp_name}_s42/'
    rel_path = f'./data/runs/{exp_name}/{exp_name}_s42/'

    with open(rel_path + "config.json") as json_file:
        json_data = json.load(json_file)
    env_kwargs = json_data['env_kwargs']
    env_name = json_data['env_name']
    env_kwargs['graph'] = max_mean_clustering_network if json_data['env_kwargs']['graph_type'] == 'maxmc' else nx.complete_graph
    ac_kwargs = json_data['ac_kwargs']
    ac_kwargs['activation'] = nn.Tanh()
    arch = json_data['arch']
    trj_len = json_data['trj_len']
    gamma = json_data['gamma']
    lam = json_data['lam']
    epochs = json_data['epochs']
    seed = json_data['seed']
    ensemble_num = env_kwargs['E']
    agent_num = env_kwargs['M']
    env_scheduler_kwargs = {
        'local_rank': 0,
        'exp_name': exp_name,
        'E': env_kwargs['E'],
        'N': env_kwargs['N'],
        'K': env_kwargs['K'],
        'exp': env_kwargs['exp'],
        'NGPU': 1,
        'data_dir': './data/runs'
    }
    env_kwargs['env_scheduler'] = envs.__dict__['random_env_scheduler'](**env_scheduler_kwargs)
    json_data['corr_type'] = 'TT'
    env_kwargs['corr_type'] = 'TT'
    if len(env_kwargs['reward_type']) < 9:
        print('modify')
        env_kwargs['reward_type'] = env_kwargs['reward_type'] + '_full'
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = envs.__dict__[env_name](**env_kwargs)
    action_type = env_kwargs['action_type']
    extra_type = env_kwargs['extra_type']
    extra_num = len(extra_type)
    # Instantiate environment
    if action_type == 'total':
        obs_dim = (env.neighbor_num + 1, env.N + extra_num)  # (3+1, 15+2)
        act_dim = env.action_space.n
        dim_len = env.N
    elif action_type == 'split':
        obs_dim = (env.neighbor_num + 1, 1 + extra_num)
        act_dim = (2,)
        dim_len = env.N

    checkpoint = torch.load(rel_path + f'pyt_save/model{epoch}.pth')
    ac = core.ActorCritic(obs_dim, act_dim, arch, **ac_kwargs)
    ac.pi.load_state_dict(checkpoint['pi'])
    ac.v.load_state_dict(checkpoint['v'])

    Parallel = DataParallel
    parallel_args = {
        'device_ids': list(range(1)),
        'output_device': 0
    }

    ac.pi = Parallel(ac.pi, **parallel_args)
    ac.v = Parallel(ac.v, **parallel_args)
    ac.eval()
    return ac, obs_dim, act_dim, dim_len, gamma, lam, env_kwargs


if __name__ == "__main__":

    name_dict = {'basic': 'st_complete_indv_raw_full_total_random_SIRF_TT_gene_ent_EC0.003_N15K7NN3RSFTMT_Z_adam_cr-1_lr1e-5_g98_cp_E5400',
                'maxmc': 'st_maxmc_indv_raw_full_total_random_SIRF_TT_gene_ent_EC0.003_N15K7NN3RSFTMT_Z_adam_cr-1_lr1e-5_g98',
                'L50R4' : 'st_complete_indv_raw_full_total_random_SIRF_TT_gene_ent_EC0.003_N15K7NN3RSFTMT_Z_adam_cr-1_lr1e-5_g98_L50R4',
                'L400' : 'st_complete_indv_raw_full_total_random_SIRF_TT_gene_ent_EC0.003_N15K3NN3RSFTMT_Z_adam_cr-1_lr1e-5_g98_L400'}
    epoch_dict = {'basic': 550, 'maxmc': 6250, 'L50R4': 5200, 'L400': 4300}

    env_name = 'SL_NK_total'
    exp_name = name_dict[args.name]
    exp_name_test = name_dict[args.nametest]
    epoch = epoch_dict[args.name]
    env_num = args.env
    test_ensemble_num = args.ensemble
    trj_len = 50 if args.nametest == 'L50R4' else 400 if args.nametest == 'L400' else 200
    repeat_len = 4 if args.nametest == 'L50R4' else 1
    print(args.name, args.nametest, flush=True)
    print(trj_len, repeat_len, flush=True)
    
    ac, obs_dim, act_dim, dim_len, gamma, lam, env_kwargs = load_model(exp_name, epoch)
    print(env_kwargs, flush=True)
    _, _, _, _, _, _, env_kwargs = load_model(exp_name_test, 0)
    reward_supply_type = 'full'
    env_kwargs['rescale'] = False
    terminate = True
    print(env_kwargs, flush=True)

    # normal test, trj_len/repeat_len
    total_trj_len = trj_len * repeat_len

    scr_buf_list = []
    final_score_list = []
    Ret_list = []

    env_list = [envs.__dict__[env_name](**env_kwargs) for i in range(env_num)]

    for i in range(env_num):
        print(i, flush=True)
        buf = PPOBuffer(
            obs_dim,
            act_dim,
            test_ensemble_num,
            env_kwargs['M'],
            dim_len,
            total_trj_len,
            gamma,
            lam,
            split=True if env_kwargs['action_type'] == 'split' else False)

        env = env_list[i]
        o, _ = env.reset(test_ensemble_num, base=True)
        ep_ret, ep_len = 0, 0
        best_ep_ret = -np.inf

        for rep in range(repeat_len):
            for t in range(trj_len):
                epoch_ended = (t == trj_len - 1) and (rep == repeat_len - 1)
                a, v, logp, pi = ac.step(torch.as_tensor(o, dtype=torch.float32, device='cuda'), return_pi=True)
                next_o, r, s = env.step(a)
                ep_ret += r
                ep_len += 1
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

                # Update obs (critical!)
                o = next_o

                if epoch_ended:
                    a, v, logp, pi = ac.step(
                        torch.as_tensor(o, dtype=torch.float32, device='cuda'),
                        return_pi=True
                    )
                    _, _, s = env.step(a)
                    if terminate:
                        buf.finish_path(np.zeros_like(v))
                    else:
                        buf.finish_path(v)
                        
            if not epoch_ended:
                o, _ = env.reset(states=env.states)
        
        Ret = ep_ret / ep_len
        Ret_list.append(Ret)
        EpLen = ep_len
        FinalScore = np.mean(s)
        print(f'Env {i} / Ret : {Ret} / Final : {FinalScore}', flush=True)
        scr_buf_list.append(buf.scr_buf)
        final_score_list.append(FinalScore)
        ep_ret, ep_len = 0, 0

    print(np.mean(Ret_list), np.mean(final_score_list), flush=True)
    scr_buf_list = np.array(scr_buf_list)
    inspection_dict = {}
    inspection_dict['scr_buf_list'] = scr_buf_list
    with open(f'./result/inspection_dict_{args.name}_E{epoch}_{args.nametest}.pkl', 'wb') as f:
        pickle.dump(inspection_dict, f, pickle.HIGHEST_PROTOCOL)