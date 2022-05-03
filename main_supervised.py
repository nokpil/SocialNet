
import argparse
import os
import distutils
import pickle
import torch
import networkx as nx
import numpy as np

import envs
import ppo.net as net
import torch.nn as nn
from utils.utils import DataGen, AverageMeter
from utils.logger import EpochLogger, setup_logger_kwargs
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset


def str2bool(v):
    return bool(distutils.util.strtobool(v))


env_names = ["NK"]
nn_names = ["mlp", "ds", "st"]
graph_names = ['complete', 'ba', 'er']
action_names = ['total', 'split']
loss_names = ['CE', 'MSE']
baseline_names = ['FollowBest', 'FollowMajor']

parser = argparse.ArgumentParser(description="Running reinforcement learning of SocialLearning model.")

# Meta-hyperparameters : Responsible for overall training scheme and parallale computing.

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="W",
    help="number of data loading workers (default: 0)",
)

parser.add_argument(
    "--epochs", default=10000, type=int, metavar="N", help="number of total epochs to run"
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="SE",
    help="manual epoch number (useful on restarts)",
)

# Hyperparameters : Responsible for a single experiment

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="ds",
    choices=nn_names,
    help="NN architecture: " + " | ".join(nn_names) + " (default: mlp)",
)

parser.add_argument(
    "--env",
    metavar="ENV",
    default="NK",
    choices=env_names,
    help="Environments: " + " | ".join(env_names) + " (default: NK)",
)

parser.add_argument(
    "--lr",
    metavar="LR",
    type=float,
    default=3e-5,
    help="Learning rate for model (default: 1e-4)",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.00025,
    type=float,
    metavar="W",
    help="weight decay (default: 0.00025)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed",
    default=42,
    type=int,
    metavar="S",
    help="seed (default: 42)",
    dest="seed",
)

# Model specification : Responsible for neural model construction and environment settings.

parser.add_argument("-B", metavar="B", type=int, default=16, help="Number of batches (for gradient update)")
parser.add_argument("-M", metavar="M", type=int, default=100, help="Number of particles")
parser.add_argument("-N", metavar="N", type=int, default=15, help="Number of gene dimension (N of NK model)")
parser.add_argument("-K", metavar="K", type=int, default=7, help="Enviromental complexity (K or NK model)")
parser.add_argument("-NN", metavar="NN", type=int, default=3, help="Number of neighbors for social reference")
parser.add_argument("-exp", metavar="EXP", type=int, default=8, help="Ruggedness of the landscape (default : 8)")
parser.add_argument("-reward-constant", "-rc", metavar="RC", type=int, default=100, help="Reward factor (default : 100)")
parser.add_argument("--graph-type", "-gt", metavar='GT', choices=graph_names, help="Interaction graph generator: " + " | ".join(graph_names) + " (default: complete)")
parser.add_argument("--action-type", "-at", metavar='AT', choices=action_names, help="Type of action space: " + " | ".join(action_names) + " (default: total)")
parser.add_argument("--loss-type", "-lt", metavar='LT', choices=loss_names, help="Type of loss: " + " | ".join(loss_names) + " (default: CE)")
parser.add_argument("--extra-type", "-xt", metavar='XT', type=str, default='SI', help="Type of extra features, S : Score, I : Identifier, R : Ranking (default: SI)")
parser.add_argument("--baseline-type", "-bt", metavar='BT', choices=baseline_names, help="Type of reward: " + " | ".join(baseline_names) + " (default: indv)")
parser.add_argument("-n", "--exp-name", type=str, default="ppo", help="Experiment name")
args = parser.parse_args()
print('argparse finished')


def train(model, train_loader, optimizer, loss, criterion):
    train_losses = AverageMeter("TrainLoss", ":.4e")
    for image, label in train_loader:
        label = label.cuda()
        image = image.cuda()
        pred = model(image)
        train_loss = loss(pred, label, criterion)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses.update(train_loss.item(), image.shape[0])
    return train_losses.avg


def test(model, test_loader, loss, criterion):
    test_losses = AverageMeter("TestLoss", ":.4e")
    for image, label in test_loader:
        label = label.cuda()
        image = image.cuda()
        pred = model(image)
        test_loss = loss(pred, label, criterion)
        test_losses.update(test_loss.item(), image.shape[0])
    return test_losses.avg


def CE(pred, label, criterion):
    pred = pred.view(pred.shape[0], -1, 2)
    return criterion(pred.transpose(-2, -1), label.long())


def MSE(pred, label, criterion):
    return criterion(pred, label)


if __name__ == "__main__":
    print('Started!!')
    nx_dict = {'complete': nx.complete_graph, 'ba': nx.barabasi_albert_graph, 'er': nx.erdos_renyi_graph} 
    nx_arg_dict = {'complete': {'n': args.M}, 'ba': {'n': args.M, 'm': 19}, 'er': {'n': args.M, 'p': 0.3}}
    env_kwargs = {
        'M': args.M,
        'N': args.N,
        'K': args.K,
        'neighbor_num': args.NN,
        'exp': args.exp,
        'graph': nx_dict[args.graph_type],
        'graph_dict': nx_arg_dict[args.graph_type],
        'action_type': args.action_type,
        'extra_type': args.extra_type,
        'E': 4,  # meaningless
        'reward_type': 'indv_raw_full',  # meaningless
        'corr_type': 'FF'  # meaningless
    }
    ac_kwargs = {'activation': nn.Tanh()}
    exp_name = f'{args.graph_type}_{args.action_type}_{args.baseline_type}_{args.extra_type}_N{args.N}K{args.K}NN{args.NN}'
    logger_kwargs = setup_logger_kwargs(args.arch + '_' + exp_name + '_' + args.loss_type + '_' + args.exp_name, args.seed)

    env = envs.__dict__[f'SL_{args.env}_{args.action_type}'](**env_kwargs)
    batch_size = 8
    batch_num = 4
    total_size = batch_num * batch_size  # real data size = (NN+1) * M * batch_size * batch_num
    train_ratio = 0.8

    print('env setting finished')

    generator = DataGen(env, args.baseline_type, batch_size, batch_num)
    if not os.path.isfile('./data/supervised/' + exp_name + '_train.pkl'):
        print('data generation started')
        generator.run(exp_name, total_size, batch_size, train_ratio)
        print('data generation finished')

    # Instantiate environment
    extra_num = len(args.extra_type)
    if args.action_type == 'total':
        obs_dim = (env.neighbor_num + 1, env.N + extra_num)
        act_dim = env.action_space.n
        dim_len = env.N
    elif args.action_type == 'split':
        obs_dim = (env.neighbor_num + 1, 1 + extra_num)
        act_dim = (2,)
        dim_len = env.N
    else:
        raise NotImplementedError

    # Loader
    
    with open('./data/supervised/' + exp_name + '_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./data/supervised/' + exp_name + '_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    train_data_image = np.concatenate(train_data['Image'], axis=0)
    train_data_image = train_data_image.reshape(-1, *train_data_image.shape[-2:])
    train_data_label = np.concatenate(train_data['Label'], axis=0)
    train_data_label = train_data_label.reshape(-1, *train_data_label.shape[-1:])
    test_data_image = np.concatenate(test_data['Image'], axis=0)
    test_data_image = test_data_image.reshape(-1, *test_data_image.shape[-2:])
    test_data_label = np.concatenate(test_data['Label'], axis=0)
    test_data_label = test_data_label.reshape(-1, *test_data_label.shape[-1:])

    train_data = TensorDataset(torch.FloatTensor(train_data_image), torch.FloatTensor(train_data_label))
    test_data = TensorDataset(torch.FloatTensor(test_data_image), torch.FloatTensor(test_data_label))

    train_loader = DataLoader(
                train_data,
                batch_size=args.B,
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )
    test_loader = DataLoader(
                test_data,
                batch_size=args.B,
                shuffle=False,
                pin_memory=True,
                num_workers=args.workers
            )

    print('data loading finished')

    # Model
    output_dim = (2 * args.N if args.loss_type == 'CE' else args.N)
    model = net.__dict__[args.arch](obs_dim, output_dim, **ac_kwargs).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = (CE if args.loss_type == 'CE' else MSE)
    criterion = (nn.CrossEntropyLoss() if args.loss_type == 'CE' else nn.MSELoss())
    best_loss = np.inf

    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())
    logger.setup_pytorch_saver(model)

    print('training starts')

    for epoch in range(0, args.epochs):
        train_loss = train(model, train_loader, optimizer, loss, criterion)
        test_loss = test(model, test_loader, loss, criterion)
        logger.store(TrainLoss=train_loss)
        logger.store(TestLoss=test_loss)
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("TrainLoss", average_only=True)
        logger.log_tabular("TestLoss", average_only=True)
        logger.dump_tabular()

        is_best = test_loss < best_loss
        best_loss = min(test_loss, best_loss)
        if is_best:
            logger.save_state({"env": env}, None, False)
