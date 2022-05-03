import os
import argparse
from distutils.util import strtobool
import networkx as nx

import torch
import torch.nn as nn
import envs
import ppo.core as core
from ppo.ppo import ppo

from utils.logger import setup_logger_kwargs
from utils.utils import max_mean_clustering_network

def str2bool(v):
    return bool(strtobool(v))

def bool2str(v):
    return 'T' if v else 'F'

env_names = ["NK"]
nn_names = ["mlp", "ds", "st"]
graph_names = ['complete', 'ba', 'er', 'maxmc']
action_names = ['total', 'split']
scheduler_names = ['random', 'multifixed', 'periodic', 'gradual']
norm_names = ['disc', 'gene', 'agent', 'group']
baseline_names = ['FollowBest', 'FollowBest_indv', 'FollowMajor', 'FollowMajor_indv', 'IndvLearning', 'RandomCopy']
#baseline_names = []

parser = argparse.ArgumentParser(description="Running reinforcement learning of SocialLearning model.")
# Due to the sampler. Since our sampler does // (floor division) only with positive integers, we don't have to worry about its behavior change.

# Meta-hyperparameters : Responsible for overall training scheme and parallale computing.
parser.add_argument("-d", "--distributed", default=True, type=str2bool, help="Distributed training when enabled ")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="W", help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="SE", help="manual epoch number (useful on restarts)")
parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--NGPU", default=1, type=int)

# Hyperparameters : Responsible for a single experiment
parser.add_argument("-a", "--arch", metavar="ARCH", default="ds", choices=nn_names, help="NN architecture: " + " | ".join(nn_names) + " (default: mlp)")
parser.add_argument("--env", metavar="ENV", default="NK", choices=env_names, help="Environments: " + " | ".join(env_names) + " (default: NK)")
parser.add_argument("--gamma", metavar="GAM", type=float, default=0.98, help="Discounting factor (default: 0.999)")
parser.add_argument("--lamda", metavar="LAM", type=float, default=0.95, help="Parameter of Generalized Advantage Estimator (GAE) (default: 0.95)")
parser.add_argument("--pi-lr", metavar="PL", type=float, default=1e-5, help="Learning rate for actor (default: 1e-4)")
parser.add_argument("--vf-lr", metavar="VL", type=float, default=3e-5, help="Learning rate for critic (default: 3e-4)")
parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, metavar="W", help="weight decay (default: 0.0)", dest="weight_decay")
parser.add_argument("--clip-ratio", metavar="CR", type=float, default=-1, help="Clip ratio (default: 0.2)")
parser.add_argument("--seed", default=42, type=int, metavar="S", help="seed (default: 42)", dest="seed")

# Model specification : Responsible for neural model construction and environment settings.
parser.add_argument("-B", metavar="B", type=int, default=1000, help="Number of batches (for gradient update)")
parser.add_argument("-E", metavar="E", type=int, default=10, help="Number of ensembles")
parser.add_argument("-M", metavar="M", type=int, default=100, help="Number of particles")
parser.add_argument("-N", metavar="N", type=int, default=15, help="Number of gene dimension (N of NK model)")
parser.add_argument("-K", metavar="K", type=int, default=7, help="Enviromental complexity (K or NK model)")
parser.add_argument("-NN", metavar="NN", type=int, default=3, help="Number of neighbors for social reference")
parser.add_argument("-exp", metavar="EXP", type=int, default=8, help="Ruggedness of the landscape (default : 8)")
parser.add_argument("-L", metavar="L", type=int, default=200, help="Trajectory length of a single episode divided by repeat length R.")
parser.add_argument("-R", metavar="R", type=int, default=1, help="Numbers of trajectory repetition. L * R = total length (usually 200).")
parser.add_argument("-I", metavar="I", type=int, default=100, help="Number of iterations for each training")
parser.add_argument("--ent-coef", metavar="EC", type=float, default=0.05, help="Coeffcient for entropy loss")
parser.add_argument("--graph-type", "-gt", metavar='GT', choices=graph_names, help="Interaction graph generator: " + " | ".join(graph_names) + " (default: complete)")
parser.add_argument("--reward-type", "-rt", metavar='RT', type=str, help="Type of reward: [indv/pop] + [raw/diff] + [full/final/finalmean] (default: indv)")
parser.add_argument("--action-type", "-at", metavar='AT', choices=action_names, help="Type of action space: " + " | ".join(action_names) + " (default: total)")
parser.add_argument("--env-scheduler-type", "-st", metavar='ST', choices=scheduler_names, help="Type of environment scheduler: " + " | ".join(scheduler_names) + " (default: random)")
parser.add_argument("--extra-type", "-xt", metavar='XT', type=str, default='SI', help="Type of extra features, S : Score, I : Identifier, R : Ranking (default: SI)")
parser.add_argument("--corr-type", "-ct", metavar='CT', type=str, default='FF', help="Type of rectification from environment. 1st T/F : state correction, 2nd T/F : reward correction")
parser.add_argument("--norm-type", "-nt", metavar='NT', type=str, default='disc', help="Type of normalization for policy gradient: " + " | ".join(norm_names) + " (default: disc)")
parser.add_argument("--rescale", "-rs", metavar='RS', type=str2bool, default=False, help="Rescaling. If true, scores in obs will be multiplied by reward_const of environment (typically 100)")
parser.add_argument("--terminate", "-tm", metavar='TM', type=str2bool, default=False, help="Epoch termination. If true, value will be reaplced with 0 when finishing the path.")
parser.add_argument("--checkpoint", "-cp", metavar='CP', default='-1', type=str, help="(Epoch of saved checkpoint of the model)_(New run's name). if not, -1.")
parser.add_argument("-n", "--exp-name", type=str, default="ppo", help="Experiment name")

args = parser.parse_args()
local_rank = int(os.environ["LOCAL_RANK"])

if __name__ == "__main__":
    data_dir = None
    nx_dict = {'complete': nx.complete_graph, 'ba': nx.barabasi_albert_graph, 'er': nx.erdos_renyi_graph, 'maxmc': max_mean_clustering_network}
    nx_arg_dict = {'complete': {'n': args.M}, 'ba': {'n': args.M, 'm': 19}, 'er': {'n': args.M, 'p': 0.3}, 'maxmc': {'n': args.M}}

    exp_name = f'{args.arch}_{args.graph_type}_{args.reward_type}_{args.action_type}_{args.env_scheduler_type}_{args.extra_type}_{args.corr_type}_{args.norm_type}_EC{args.ent_coef}_N{args.N}K{args.K}NN{args.NN}RS{bool2str(args.rescale)}TM{bool2str(args.terminate)}_{args.exp_name}'
    exp_name_cp = ''
    checkpoint = (args.checkpoint).split('_')
    if int(checkpoint[0]) > 0:
        exp_name_cp = f'{args.arch}_{args.graph_type}_{args.reward_type}_{args.action_type}_{args.env_scheduler_type}_{args.extra_type}_TT_{args.norm_type}_EC{args.ent_coef}_N{args.N}K{args.K}NN{args.NN}RS{bool2str(args.rescale)}TM{bool2str(args.terminate)}_{args.exp_name}'
        exp_name += '_cp_' + checkpoint[1]

    #args.env_scheduler_type = 'multifixed'

    env_scheduler_kwargs = {
        'local_rank': local_rank,
        'exp_name': exp_name,
        'E': args.E,
        'N': args.N,
        'K': args.K,
        'exp': args.exp,
        'NGPU': args.NGPU
    }

    #env_scheduler_kwargs['K'] = 7

    env_kwargs = {
        'E': args.E,
        'M': args.M,
        'N': args.N,
        'K': args.K,
        'exp': args.exp,
        'neighbor_num': args.NN,
        'graph_type': args.graph_type,
        'graph': nx_dict[args.graph_type],
        'graph_dict': nx_arg_dict[args.graph_type],
        'reward_type': args.reward_type,
        'action_type': args.action_type,
        'extra_type': args.extra_type,
        'corr_type': args.corr_type,
        'rescale': args.rescale,
        'env_scheduler': envs.__dict__[args.env_scheduler_type + '_env_scheduler'](**env_scheduler_kwargs)
    }

    #env_kwargs['K'] = 7

    print(env_kwargs['env_scheduler'].local_rank)

    #ac_kwargs = {'activation': nn.ReLU()}
    ac_kwargs = {'activation': nn.Tanh()}
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    ppo(
        env_name=f'SL_{args.env}_{args.action_type}',
        env_kwargs=env_kwargs,
        actor_critic=core.ActorCritic,
        arch=args.arch,
        ac_kwargs=ac_kwargs,
        seed=args.seed + local_rank,
        ensemble_num=args.E,
        agent_num=args.M,
        trj_len=args.L,
        repeat_len=args.R,
        epochs=args.epochs,
        batch_size=args.B,
        test_batch_size=args.B * 5,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        ent_coef=args.ent_coef,
        pi_lr=args.pi_lr,
        vf_lr=args.vf_lr,
        weight_decay=args.weight_decay,
        train_pi_iters=args.I,
        train_v_iters=args.I,
        lam=args.lamda,
        target_kl=0.01,
        norm_type=args.norm_type,
        terminate=args.terminate,
        logger_kwargs=logger_kwargs,
        save_freq=50,
        distributed=args.distributed,
        baselines=baseline_names,
        env_scheduler_type=args.env_scheduler_type,
        checkpoint=(exp_name_cp, checkpoint[0]),
        local_rank=local_rank
    )
