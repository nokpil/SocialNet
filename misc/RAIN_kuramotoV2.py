# coding=utf-8
import argparse
import os
import os.path as path
import random
import shutil
import time
import warnings
import sys
import numpy as np
import scipy as sp
import math

from time import sleep

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed as dist
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import (
    DataLoader,
)  # (testset, batch_size=4,shuffle=False, num_workers=4)
from torch.optim.lr_scheduler import ReduceLROnPlateau as RLRP
from torch.nn.init import xavier_normal
from torch.nn.parameter import Parameter
from torch.utils.data.dataset import TensorDataset

import pickle
import importlib
import itertools
import random
from datetime import datetime
from collections import OrderedDict
from copy import deepcopy
import tracemalloc
import gc
import distutils
import distutils.util

import src.DataStructure as DS
from src.mypytorch import *

def str2bool(v):
    return bool(distutils.util.strtobool(v))

parser = argparse.ArgumentParser(description="Pytorch VAINS Training")

parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=2, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=16,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=0.0005,
    type=float,
    metavar="W",
    help="weight decay (default: 0.0005)",
    dest="weight_decay",
)

parser.add_argument(
    "--world-size", default=1, type=int, help="number of nodes for distributed training"
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)

parser.add_argument("--local_rank", type=int)

parser.add_argument("--agent-num", default=10, type=int, help="Number of LSTM layers for encoding")
parser.add_argument("--dt", default=10, type=int, help="Number of LSTM layers for encoding")

parser.add_argument(
    "--model-type", default="gat", type=str, help="model type : mlp, gat, gatS"
)

parser.add_argument('-it', '--interaction-type', type=str, default='N',
                    help='interaction type / N (normal), S (signed), D (directed), SD (signed+directed)')

parser.add_argument('-sm', '--sample-mode', type=str, default='uniform',
                    help='interaction weight type / uniform, normal, duplex')

parser.add_argument(
    "--block-type",
    default="mlp",
    type=str,
    help="mlp : simple multi-layer perceptron, res : skip-connection",
)
parser.add_argument(
    "--att-type",
    default="gat",
    type=str,
    help="kqv : normal kqv (linear transformation), gat",
)

parser.add_argument(
    "--heads-num", default=1, type=int, help='For "multi", works as number of heads.'
)

parser.add_argument(
    "--heads-dim", default=128, type=int, help="Dimension of a single head of attention vector."
)
parser.add_argument("--mode-num", default=1, type=int, help="Number of gaussian mixture mode.")
parser.add_argument("--lstm-num", default=1, type=int, help="Number of LSTM layers for encoding")

parser.add_argument("--dropout", default=0.0, type=float, help="Rate of dropout on attention.")
parser.add_argument("--checkpoint", default="no", type=str, help="no, cp")
parser.add_argument(
    "--indicator", default="", type=str, help="Additional specification for file name."
)
parser.add_argument("--seed", default=0, type=int, help="Random seed for torch and numpy")
parser.add_argument("--forcing-period", default=30, type=int, help="Teacher forcing period")

parser.add_argument("--input-length", default=100, type=int, help="Input length of the sequence (max : 49)")
parser.add_argument("--output-length", default=50, type=int, help="Input length of the sequence (max : 50)")
parser.add_argument("--noise-var", default=0., type=float, help="Noise strength.")

parser.add_argument("--act-type", default='sigmoid', type=str, help="Activation type.")
parser.add_argument("--sig", default=True, type=str2bool, help="Whether using generated variance or not")
parser.add_argument("--use-sample", default=True, type=str2bool, help="Whether use generated mu or not")
parser.add_argument("--pa", default=True, type=str2bool, help="Whether use pairwise attention or not")
parser.add_argument("--gt", default=False, type=str2bool, help="Whether using GroundTruth weight")
parser.add_argument("--ww", default=True, type=str2bool, help="Whether using inherent frequency as a feature")

class Kuramoto():
    def __init__(self):
        self.name = 'Kuramoto'
        self.rule_name = 'RAIN_' + str(self.name)
      
    def assign_pp(self, plugin_parameters):
        self.pp = plugin_parameters
        assert 'agent_num' in plugin_parameters
        self.agent_num = self.pp['agent_num']
        assert 'dt' in plugin_parameters
        self.dt = self.pp['dt']
        assert 'data_step' in plugin_parameters
        self.data_step = self.pp['data_step']
        assert 'label_step' in plugin_parameters
        self.label_step = self.pp['label_step']
        assert 'state_num' in plugin_parameters
        self.state_num = self.pp['state_num']
        assert 'answer_num' in plugin_parameters
        self.answer_num = self.pp['answer_num']
        assert 'const_num' in plugin_parameters
        self.const_num = self.pp['const_num']

def group_weight(module, name_list):
    group_decay = []
    group_no_decay = []
    for m in module.named_parameters():
        if m[0] in name_list:
            group_no_decay.append(m[1])
        else:
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    if len(group_no_decay) != 0:
        groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
        return groups
    else:
        return [dict(params=group_decay)]

class Module_RAIN_kuramotoV2(nn.Module):
    def __init__(self, cfg_state_enc, cfg_ge_att, cfg_init, cfg_lstm, cfg_enc, cfg_self_func, cfg_intr_func, cfg_mu, cfg_sig, D_att, D_heads_num, D_agent, block_type, att_type, act_type, dropout, sig=True, use_sample=True, pa=True, gt=False):
        super(Module_RAIN_kuramotoV2, self).__init__()

        self.D_att = D_att
        self.heads_num = D_heads_num
        self.D_agent = D_agent
        self.block_type = block_type
        self.att_type = att_type
        self.dropout = dropout
        self.nl = 'MS'
        self.act_type = act_type
        self.sig = sig
        if not self.sig:
            self.fixed_var = 5e-4
        self.use_sample = use_sample
        self.gt = gt
        self.pa = pa
        self.reg_norm = 1.
        
        # Raw data encoder
        self.state_enc = cfg_Block(block_type, cfg_state_enc, D_agent, self.nl, False, False)

        # Graph Extraction transformer
        if not self.gt and self.pa:
            self.key_ct = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)
            self.query_ct = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)
            self.value_ct = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)

        # self.self = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)     

        if self.att_type == 'gat':
            self.att = cfg_Block(block_type, cfg_ge_att, D_agent, self.nl, False, False)
        elif self.att_type == 'kqv':
            self.key = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)
            self.query = cfg_Block(block_type, cfg_enc, D_agent, self.nl, False, False)

        # Encoding / Decoding LSTM
        self.init_hidden = cfg_Block(block_type, cfg_init, D_agent, self.nl, False, False)
        self.init_cell = cfg_Block(block_type, cfg_init, D_agent, self.nl, False, False)
        self.lstm = nn.LSTM(*cfg_lstm)
        self.lstm_num = cfg_lstm[-1]

        self.self_func = cfg_Block(block_type, cfg_self_func, D_agent, self.nl, False, False)
        self.intr_func = cfg_Block(block_type, cfg_intr_func, D_agent, self.nl, False, False)
        self.mu_dec = cfg_Block(block_type, cfg_mu, D_agent, self.nl, False, False)
        if self.sig:
            self.sig_dec = cfg_Block(block_type, cfg_sig, D_agent, self.nl, False, False)

        if block_type == 'mlp':
            self.D_k = 1
            self.D_s = 1
        elif block_type == 'res':
            self.D_k = 1
            self.D_s = 1

        self.SR = None
        if self.act_type == 'srelu':
            self.SR = SReLU_limited()

        if self.act_type == 'nrelu':
            self.norm_param = Parameter(
                torch.tensor(1.0, dtype=torch.float, requires_grad=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def initialize(self, x):
        # x.shape = [batch_num, agent_num, lstm_dim]
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = x.reshape(batch_num * agent_num, -1)
        h = self.init_hidden(x)
        c = self.init_cell(x)  # shape = (batch_num * agent_num, lstm_dim * 2)
        return (h.view(self.lstm_num, batch_num * agent_num, -1), c.view(self.lstm_num, batch_num * agent_num, -1))

    def encode(self, x, hidden, cell):
        # x.shape = (len_enc - 1, batch_num, agent_num, lstm_dim) (after transposed)
        x = self.state_enc(x)
        x = x.transpose(1, 0)
        len_enc_m1 = x.shape[0]
        batch_num = x.shape[1]
        agent_num = x.shape[2]
        x = x.reshape(len_enc_m1, batch_num * agent_num, -1)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        return output, (hidden, cell), (len_enc_m1, batch_num, agent_num, x.shape[-1])

    def extract(self, output, shape, weight=None, final=False):
        if self.gt:
            return None, weight / 2.
        else:
            if final:
                return None, None
            else:
                len_enc_m1, batch_num, agent_num, lstm_dim = shape
                if self.pa:
                    # output.shape = [len_enc_m1 = len_enc - 1, batch_num * agent_num, lstm_dim]

                    k_ct = self.key_ct(output)
                    q_ct = self.query_ct(output)
                    v_ct = self.value_ct(output)

                    # 1. Sequence contraction : merging time_series into weighted sum of agent vector with attention module
                    head_dim = lstm_dim // self.heads_num
                    assert head_dim * self.heads_num == lstm_dim, "embed_dim must be divisible by num_heads"

                    k_ct = self.key_ct(output).view(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)
                    q_ct = self.query_ct(output).view(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)
                    v_ct = self.value_ct(output).view(len_enc_m1, batch_num, agent_num, self.heads_num, head_dim)

                    # change order into (batch_num, self.heads_num, agent_num, len_enc_m1, lstm_dim)
                    k_ct = k_ct.permute(1, 3, 2, 0, 4)
                    q_ct = q_ct.permute(1, 3, 2, 0, 4)
                    v_ct = v_ct.permute(1, 3, 2, 0, 4)

                    k_cts = torch.stack([k_ct for _ in range(agent_num)], dim=-3).unsqueeze(-1)
                    q_cts = torch.stack([q_ct for _ in range(agent_num)], dim=-4).unsqueeze(-1)
                    v_cts = torch.stack([v_ct for _ in range(agent_num)], dim=-3)

                    attention_score = torch.softmax((torch.matmul(q_cts.transpose(-2, -1), k_cts) / math.sqrt(lstm_dim)).squeeze(-1), dim=-2)
                    output = torch.sum(attention_score * v_cts, dim=-2)  # sequence contracted in time dimension
                    output = output.permute(0, 2, 3, 1, 4).reshape(batch_num, agent_num, agent_num, -1)  # (batch_num, agent_num1, agent_num2, self.heads_num * lstm_dim)
                
                else:
                    assert self.heads_num == 1
                    output = output[-1].view(batch_num, agent_num, -1)
                    output = torch.stack([output for _ in range(agent_num)], dim=-2)
                    attention_score = None
                # 2. Graph extraction : exploit graph structure from merged agent vector with transformer

                weight = None

                if self.att_type == 'gat':
                    w = torch.cat((output, output.transpose(-3, -2)), dim=-1)  # swap two agent dimensions
                    if self.act_type == 'sigmoid':
                        mask = torch.eye(agent_num, agent_num).to(output.device)
                        mask = mask.float().masked_fill(mask == 1, float(-10000.)).masked_fill(mask == 0, float(0.0))
                        weight = torch.sigmoid(self.att(w).squeeze(-1) + mask)
                    elif self.act_type == 'tanh':
                        mask = torch.eye(agent_num, agent_num).to(output.device)
                        mask = 1 - (mask.float())
                        weight = torch.tanh(self.att(w).squeeze(-1)) * mask
                   
                elif self.att_type == 'kqv':
                    # raise NotImplementedError
                    k_ge = self.key(output).unsqueeze(-1)
                    q_ge = self.query(output).unsqueeze(-1)
                    weight = torch.softmax((torch.matmul(q_ge.transpose(-2, -1), k_ge) / math.sqrt(lstm_dim)).squeeze(-1).squeeze(-1)+mask, dim=-2)
                    
                return attention_score, weight

    def decode(self, x, hidden, cell, weight):
        epsilon = 1e-8
        batch_num = x.shape[0]
        agent_num = x.shape[1]
        x = self.state_enc(x)
        x = x.reshape(1, batch_num * agent_num, -1)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))

        h = hidden[-1].view(batch_num, agent_num, -1)
        self_func_output = self.self_func(h)
        intr_func_data = torch.cat([torch.stack([h for _ in range(agent_num)], dim=-2), torch.stack([h for _ in range(agent_num)], dim=-3)], dim=-1)
        intr_func_output = self.intr_func(intr_func_data)

        p_list = [self_func_output]
        # print(self_func_output.shape, intr_func_output.shape, weight.shape)
        assert self.D_agent == agent_num
        if self.att_type == 'gat':
            p_list.append(torch.sum((weight.unsqueeze(-1) * intr_func_output), dim=-2) / (self.D_agent - 1))  # except self
        else:
            p_list.append(torch.sum((weight.unsqueeze(-1) * intr_func_output), dim=-2))

        c = torch.cat(p_list, dim=-1)
        mu = self.mu_dec(c).squeeze()
        if self.sig:
            sig = (torch.sigmoid(self.sig_dec(c)).squeeze() + epsilon) / self.reg_norm

        if self.sig:
            return (mu, sig), hidden, cell
        else:
            return (mu, torch.ones_like(sig) * self.fixed_var), hidden, cell


def main():
    tracemalloc.start()
    best_test_loss = np.inf
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(torch.cuda.device_count(), args.local_rank)
    if args.local_rank == 0:
        print(f'sig : {args.sig}')
        print(f'use_sample : {args.use_sample}')
        print(f'pa : {args.pa}')
        print(f'gt : {args.gt}')
        print(f'ww : {args.ww}')
        print('data : ' + 'kuramotoS' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)) + '_' + args.sample_mode + '_' + args.interaction_type + 'L2')
    torch.distributed.init_process_group(backend="gloo", init_method="env://")
    torch.cuda.set_device(args.local_rank)

    # Plugin parameters
    pp = {}
    system = Kuramoto()

    # Size parameters
    pp["agent_num"] = args.agent_num
    pp["dt"] = args.dt
    pp["data_step"] = args.input_length
    pp["label_step"] = args.output_length
    if args.ww:
        pp["state_num"] = 3
        pp["answer_num"] = 2
        pp["const_num"] = 1
    else:
        pp["state_num"] = 2
        pp["answer_num"] = 2
        pp["const_num"] = 0
    system.assign_pp(pp)

    # Data loading code
    file_name = (
        system.rule_name
        + "_A"
        + str(system.agent_num)
        + "_dt"
        + str(args.dt)
    )
    indicator = (
        ('_GT_' if args.gt else '')
        + "AT"
        + str(args.att_type)
        + "_BT"
        + str(args.block_type)
        + "_HN"
        + str(args.heads_num)
        + "_Dk"
        + str(args.mode_num)
        + "_DL"
        + str(args.lstm_num)
        + "_NV"
        + str(args.noise_var)
        + "_AT"
        + str(args.act_type)
        + "_IT"
        + str(args.interaction_type).lower()
        + "_SM"
        + str(args.sample_mode)
        + "_"
        + args.indicator
    )

    def load_data_RAIN(batch_size=1, sim_folder="", data_folder="data", len_enc=50, len_dec=50, noise_var=0., ww=True):
        # the edges numpy arrays below are [ num_sims, N, N ]
        dphi_train = np.load(path.join(data_folder, sim_folder, "dphi_train.npy"))
        sinphi_train = np.load(path.join(data_folder, sim_folder, "sinphi_train.npy"))
        freq_train = np.load(path.join(data_folder, sim_folder, "freq_train.npy"))
        edges_train = np.load(path.join(data_folder, sim_folder, "edges_train.npy"))
        order_train = np.load(path.join(data_folder, sim_folder, "order_train.npy"))

        dphi_test = np.load(path.join(data_folder, sim_folder, "dphi_test.npy"))
        sinphi_test = np.load(path.join(data_folder, sim_folder, "sinphi_test.npy"))
        freq_test = np.load(path.join(data_folder, sim_folder, "freq_test.npy"))
        edges_test = np.load(path.join(data_folder, sim_folder, "edges_test.npy"))
        order_test = np.load(path.join(data_folder, sim_folder, "order_test.npy"))

        #phi_train = np.load(path.join(data_folder, sim_folder, "phi_train.npy"))
        #phi_test = np.load(path.join(data_folder, sim_folder, "phi_test.npy"))

        dphi_train += np.random.randn(*dphi_train.shape) * noise_var
        sinphi_train += np.random.randn(*sinphi_train.shape) * noise_var

        dphi_test += np.random.randn(*dphi_test.shape) * noise_var
        sinphi_test += np.random.randn(*sinphi_test.shape) * noise_var

        dphi_max = dphi_train.max()
        sinphi_max = sinphi_train.max()
        dphi_min = dphi_train.min()
        sinphi_min = sinphi_train.min()
        #phi_max = phi_train.max()
        #phi_min = phi_train.min()

        # Normalize to [-1, 1]
        dphi_train = (dphi_train - dphi_min) * 2 / (dphi_max - dphi_min) - 1
        sinphi_train = (sinphi_train - sinphi_min) * 2 / (sinphi_max - sinphi_min) - 1

        dphi_test = (dphi_test - dphi_min) * 2 / (dphi_max - dphi_min) - 1
        sinphi_test = (sinphi_test - sinphi_min) * 2 / (sinphi_max - sinphi_min) - 1

        #phi_train = (phi_train - phi_min) * 2 / (phi_max - phi_min) - 1
        #phi_test = (phi_test - phi_min) * 2 / (phi_max - phi_min) - 1

        # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
        
        # shape : train_len, agent_num, (len_enc+len_dec), 1
        dphi_train = dphi_train.reshape(*dphi_train.shape, 1)
        sinphi_train = sinphi_train.reshape(*sinphi_train.shape, 1)
        freq_train = np.expand_dims(np.expand_dims(freq_train, -1).repeat(dphi_train.shape[-2], -1), -1) 
        
        # shape : test_len, agent_num, (len_enc+len_dec), 1
        dphi_test = dphi_test.reshape(*dphi_test.shape, 1)
        sinphi_test = sinphi_test.reshape(*sinphi_test.shape, 1)
        freq_test = np.expand_dims(np.expand_dims(freq_test, -1).repeat(dphi_test.shape[-2], -1), -1)
        
        #phi_train = phi_train.reshape(*phi_train.shape, 1)  
        #phi_test = phi_test.reshape(*phi_test.shape, 1) 

        if ww:
            feat_train = np.transpose(np.concatenate([dphi_train, sinphi_train, freq_train], axis=-1), (0, 2, 1, 3))
            feat_test = np.transpose(np.concatenate([dphi_test, sinphi_test, freq_test], axis=-1), (0, 2, 1, 3))
        else:
            feat_train = np.transpose(np.concatenate([dphi_train, sinphi_train], axis=-1), (0, 2, 1, 3))
            feat_test = np.transpose(np.concatenate([dphi_test, sinphi_test], axis=-1), (0, 2, 1, 3))
        
        data_len = -1
        feat_train_enc = torch.FloatTensor(feat_train[:, :len_enc])[:data_len]
        feat_train_dec = torch.FloatTensor(feat_train[:, len_enc:len_enc + len_dec])[:data_len]
        freq_train = torch.FloatTensor(freq_train)[:data_len]
        edges_train = torch.FloatTensor(edges_train)[:data_len]
        order_train = torch.FloatTensor(order_train)[:data_len]

        feat_test_enc = torch.FloatTensor(feat_test[:, :len_enc])[:data_len]
        feat_test_dec = torch.FloatTensor(feat_test[:, len_enc:len_enc + len_dec])[:data_len]
        freq_test = torch.FloatTensor(freq_test)[:data_len]
        edges_test = torch.FloatTensor(edges_test)[:data_len]
        order_test = torch.FloatTensor(order_test)[:data_len]

        train_data = TensorDataset(feat_train_enc, feat_train_dec, freq_train[:, :, 0].squeeze(), edges_train.squeeze(), order_train)
        test_data = TensorDataset(feat_test_enc, feat_test_dec, freq_test[:, :, 0].squeeze(), edges_test.squeeze(), order_test)

        print('dataset finished')

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

        return (
            train_loader,
            test_loader,
            train_sampler,
            test_sampler,
            dphi_max,
            dphi_min,
            sinphi_max,
            sinphi_min,
        )

    # 'kuramotoS' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)) + '_' + args.sample_mode + '_' + args.interaction_type + 'L',
    #'''
    train_loader, test_loader, train_sampler, test_sampler, loc_max, loc_min, vel_max, vel_min = load_data_RAIN(
                                                        args.batch_size,
                                                        'kuramotoS' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)) + '_' + args.sample_mode + '_' + args.interaction_type + 'L_weak',
                                                        data_folder='data', len_enc=args.input_length, len_dec=args.output_length, 
                                                        noise_var=args.noise_var, ww=args.ww)
    '''
    train_loader, test_loader, train_sampler, test_sampler, loc_max, loc_min, vel_max, vel_min = load_data_RAIN(
                                                        args.batch_size, 
                                                        'kuramoto' + '_' + str(args.agent_num) + '_t' + str(int(args.dt)) + '_' + args.sample_mode,
                                                        data_folder='data', len_enc=args.input_length, len_dec=args.output_length, 
                                                        noise_var=args.noise_var, ww=args.ww)
    
    '''

    if args.local_rank == 0:
        print(file_name + '_' + indicator)

    if args.model_type == 'lstm':
        D_hidden_lstm = 256
        D_in_lstm = system.state_num
        D_lstm_num = args.lstm_num

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 256
        D_out_enc = 256

        D_in_dec = 256
        D_hidden_dec = 256
        D_out_dec = 256
        D_hidden_stat = 256

        D_agent = system.agent_num

        # cfg definition

        cfg_state_enc = [D_in_lstm, 256, 256]
        cfg_init = [D_in_lstm, D_hidden_lstm * D_lstm_num]
        cfg_lstm = [256, D_hidden_lstm, D_lstm_num]

        cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]
        cfg_dec = [D_in_dec, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec, D_hidden_stat, system.state_num]
        cfg_sig = [D_out_dec, D_hidden_stat, system.state_num]

        # model definition
        model = Module_RAIN_kuramoto_lstm(
            cfg_state_enc,
            cfg_init,
            cfg_lstm,
            cfg_enc,
            cfg_dec,
            cfg_mu,
            cfg_sig,
            D_agent,
            args.block_type,
            args.dropout,
            args.sig,
            args.use_sample,
        ).cuda()
        print("hello")

    elif args.model_type == "gat":

        # Dimension definition
        D_agent = system.agent_num
        D_head = args.heads_dim
        D_heads_num = args.heads_num
        D_att = D_head * D_heads_num

        D_hidden_lstm = 256
        D_in_lstm = system.state_num
        D_lstm_num = args.lstm_num

        D_in_enc = D_hidden_lstm
        D_hidden_enc = 256
        D_out_enc = D_att

        D_in_dec = D_hidden_lstm
        D_hidden_dec = 256
        D_out_dec = 256
        D_hidden_stat = 256

        # cfg definition

        cfg_state_enc = [D_in_lstm, 64, D_att]
        if args.pa:
            cfg_ge_att = [D_att * 2, 32, 16, 1]
        else:
            cfg_ge_att = [D_hidden_lstm * 2, 32, 16, 1]
        
        cfg_init = [D_in_lstm, D_hidden_lstm * D_lstm_num]
        cfg_lstm = [D_att, D_hidden_lstm, D_lstm_num]
        cfg_enc = [D_in_enc, D_out_enc]
        # cfg_enc = [D_in_enc, D_hidden_enc, D_out_enc]

        cfg_self_func = [D_in_dec, D_hidden_dec, D_out_dec]
        cfg_intr_func = [D_in_dec * 2, D_hidden_dec, D_out_dec]

        cfg_mu = [D_out_dec * 2, D_hidden_stat, system.state_num - system.const_num]
        cfg_sig = [D_out_dec * 2, D_hidden_stat, system.state_num - system.const_num]

        # model definition

        model = Module_RAIN_kuramotoV2(
            cfg_state_enc,
            cfg_ge_att,
            cfg_init,
            cfg_lstm,
            cfg_enc,
            cfg_self_func,
            cfg_intr_func,
            cfg_mu,
            cfg_sig,
            D_att,
            D_heads_num,
            D_agent,
            args.block_type,
            args.att_type,
            args.act_type,
            args.dropout,
            args.sig,
            args.use_sample,
            args.pa,
            args.gt
        ).cuda()
       
    else:
        print("hello")
    # define loss function (criterion) and optimizer

    criterion = gmm_criterion(1)  # 2 answers, but no correlation presumed
    # criterion = gmm_criterion(2)
    sampler = gmm_sample(1, r=True if args.sig else False)  # 2 answers, but no correlation presumed
    # sampler = gmm_sample(system.answer_num, r=False)
    name_list = []
    if model.act_type == 'srelu':
        name_list = ['SR.tl', 'SR.tr', 'SR.ar']
    elif model.act_type == 'nrelu':
        name_list = ['norm_param']
    else:
        pass

    params = group_weight(model, name_list)

    optimizer = torch.optim.AdamW(
        params, args.lr, weight_decay=args.weight_decay
        )
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1, eta_min=0)
    scheduler = RLRP(optimizer, "min", factor=0.5, patience=10, min_lr=0, verbose=1)

    if args.checkpoint[:2] == "cp":
        print("cp entered")
        checkpoint = torch.load(file_name + "_" + indicator + "_checkpoint.pth", map_location="cuda:" + str(args.local_rank))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_test_loss = checkpoint["best_test_loss"]
        epochs = args.epochs + start_epoch

        if len(args.checkpoint) > 2:
            change_rate = float(args.checkpoint[2:])
            for g in optimizer.param_groups:
                g['lr'] = change_rate
                if args.local_rank == 0:
                    print(f'change rate : {change_rate}, changed lr : {g["lr"]}')

    elif args.checkpoint[:2] == 'pp':
        print("pp entered")
        change_agent = int(args.checkpoint[2:])
        prev_file_name = (system.rule_name + "_A" + str(change_agent) + "_dt"+ str(args.dt))
        checkpoint = torch.load(prev_file_name + "_" + indicator + "_checkpoint.pth", map_location="cuda:" + str(args.local_rank))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        #scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_test_loss = checkpoint["best_test_loss"]
        epochs = args.epochs + start_epoch
        indicator += 'pp'

        if len(args.checkpoint) > 2:
            change_rate = float(args.checkpoint[2:])
            for g in optimizer.param_groups:
                g['lr'] *= change_rate
                if args.local_rank == 0:
                    print(f'change rate : {change_rate}, changed lr : {g["lr"]}')
                
    else:
        start_epoch = args.start_epoch
        epochs = args.epochs

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    train_loss, train_count, test_loss, test_count = None, None, None, None
    with torch.autograd.detect_anomaly():
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)
            # train for one epoch
            if args.local_rank == 0:
                print("============== Epoch {} =============".format(epoch))
            train_loss, train_count = train(train_loader, model, criterion, optimizer, epoch, sampler, args)
            if args.local_rank == 0:
                train_string = "Epoch {} / Train Loss : [Total : {}] {}, ".format(
                    str(epoch), str(train_count[-1]), str(train_loss[-1])
                )
                for i in range(system.label_step):
                    train_string += " [{} : {}] {}, ".format(
                        str(i + 1), str(train_count[i]), str(train_loss[i])
                    )
                train_string += f' / Learning Rate : {optimizer.param_groups[0]["lr"]}'
                print(train_string)

            # evaluate on test set
            if epoch > args.forcing_period:
                scheduler.step(train_loss[-1], epoch)
                test_loss, test_count = test(test_loader, model, criterion, sampler, args)

                if args.local_rank == 0:
                    test_string = "Epoch {} / Test Loss : [Total : {}] {}, ".format(
                        str(epoch), str(test_count[-1]), str(test_loss[-1])
                    )
                    for i in range(system.label_step):
                        test_string += " [{} : {}] {}, ".format(
                            str(i + 1), str(test_count[i]), str(test_loss[i])
                        )
                    print(test_string)

                    # remember best acc@1 and save checkpoint
                    is_best = test_loss[-1] < best_test_loss
                    best_test_loss = min(test_loss[-1], best_test_loss)
                    print(is_best, test_loss[-1], best_test_loss)
                    if is_best:
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "loss": test_loss,
                                "best_test_loss": best_test_loss,
                                "train_loss_list": (train_loss, train_count),
                                "test_loss_list": (test_loss, test_count)
                            },
                            file_name + "_" + indicator + "_checkpoint.pth",
                        )

def train(train_loader, model, criterion, optimizer, epoch, sampler, args):

    train_losses_list = []
    data_num = train_loader.dataset.tensors[0].shape[1]   # length of train_data_enc, 50
    step_num = train_loader.dataset.tensors[1].shape[1]   # length of train_data_dec, 50
    agent_num = train_loader.dataset.tensors[1].shape[2]  # number of agents, 5 / 30
    state_num = train_loader.dataset.tensors[1].shape[3]  # number of state, 3 (dphi, sinphi, omega)
    
    for i in range(step_num):
        train_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
    train_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
    forcing_period = args.forcing_period
    model.train()
    
    for i, (data_enc, data_dec, freq, edges, order) in enumerate(train_loader):
        data = data_enc.cuda(args.local_rank)
        labels = data_dec.cuda(args.local_rank)
        edges = edges.cuda(args.local_rank)
        hidden, cell = model.module.initialize(data[:, 0])
        output, (hidden, cell), shape = model.module.encode(data[:, :-1], hidden, cell)
        attention_score, weight = model.module.extract(output, shape, weight=(edges if args.gt else None))  # use final layer's hidden state
        data = data[:, -1]
        pred_mu = []
        pred_sig = []
        data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
        label_diff_list = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda(args.local_rank)

        for n in range(step_num):
            label = labels[:, n]
            # label_diff = label - data # (Not for V)
            label_diff = label_diff_list[:, n]  # (for V)
            (mu, sig), hidden, cell = model.module.decode(data, hidden, cell, weight)
            if args.use_sample:
                sample = sampler(mu, sig).cuda(args.local_rank)
            else:
                sample = mu.cuda(args.local_rank)

            if type(sample) == type(None):
                print("broke_train")
                sample = label_diff[:, n]
                break

            # Teacher forcing (depends on epoch)
            
            if args.ww:
                next_data = sample + data[:, :, :-1]
            else:
                next_data = sample + data

            if args.indicator[-2:] == "tf" and epoch < args.forcing_period:
                next_data_mask = (
                    torch.bernoulli(
                        torch.ones((sample.shape[0], sample.shape[1], 1))
                        * F.relu(torch.tensor(1 - (epoch + 1) / forcing_period))
                    ).cuda(args.local_rank)
                )
                if args.ww:
                    next_data = (next_data_mask * label[:, :, :-1] + (1 - next_data_mask) * next_data)  # W
                else:
                    next_data = (next_data_mask * label + (1 - next_data_mask) * next_data)   # noW

            elif args.indicator[-3:] == "tf2" and epoch < args.forcing_period:
                next_data_mask = (
                    torch.ones((sample.shape[0], sample.shape[1], 1)) * int(n / step_num > epoch / forcing_period)).cuda(args.local_rank)  # if the curretn step (ratio) exceeds current epoch (ratio), fill the maks to 1 for teacher forcing

                if args.ww:
                    next_data = (next_data_mask * label[:, :, :-1] + (1 - next_data_mask) * next_data)  # W
                else:
                    next_data = (next_data_mask * label + (1 - next_data_mask) * next_data)   # noW

            if args.ww:
                train_loss = torch.mean(torch.sum(criterion(label_diff[:, :, :-1], mu, sig), dim=-1))  # W
                #train_loss = 0.5*(torch.mean(torch.sum(criterion(label_diff[:, :, :-1], mu, sig), dim=-1)) + torch.mean(torch.sum(criterion(label[:, :, :-1], mu + data[:, :, :-1], sig), dim=-1)))  # W
            else:
                train_loss = torch.mean(torch.sum(criterion(label_diff, mu, sig), dim=-1))  # noW
            train_losses_list[n].update(train_loss.item(), data.size(0) * agent_num)
            train_losses_list[-1].update(train_loss.item(), data.size(0) * agent_num)
            
            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

            pred_mu.append(mu)
            pred_sig.append(sig)
            if args.ww:
                data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1) # W
            else:
                data = next_data  # noW

        pred_mu = torch.stack(pred_mu, dim=1)
        pred_sig = torch.stack(pred_sig, dim=1)

        if (args.local_rank == 0) and (i == 0):
            show_num = 10
            print(label_diff_list[0][:show_num, 0, 0], pred_mu[0][:show_num, 0, 0], pred_sig[0][:show_num, 0, 0])
            print(label_diff_list[0][:show_num, 0, 1], pred_mu[0][:show_num, 0, 1], pred_sig[0][:show_num, 0, 1])

        del mu, sig, hidden, cell
        del train_loss, sample, next_data, data
        del data_tmp, label_diff, label_diff_list
        torch.cuda.empty_cache()

    return (
        [train_losses_list[i].avg for i in range(len(train_losses_list))],
        [train_losses_list[i].count for i in range(len(train_losses_list))],
    )

def test(test_loader, model, criterion, sampler, args):

    test_losses_list = []
    data_num = test_loader.dataset.tensors[0].shape[1]  # length of train_data_enc, 50
    step_num = test_loader.dataset.tensors[1].shape[1]  # length of train_data_dec, 50
    agent_num = test_loader.dataset.tensors[1].shape[2] # number of agents, 5 / 30
    state_num = test_loader.dataset.tensors[1].shape[3] # number of state, 3 (dphi, sinphi, omega)
    
    for i in range(step_num):
        test_losses_list.append(AverageMeter("Loss_" + str(i), ":.4e"))
    test_losses_list.append(AverageMeter("Total_Loss" + str(i), ":.4e"))
    forcing_period = args.forcing_period
    model.eval()
    
    for i, (data_enc, data_dec, freq, edges, order) in enumerate(test_loader):
        data = data_enc.cuda(args.local_rank)
        labels = data_dec.cuda(args.local_rank)
        edges = edges.cuda(args.local_rank)
        hidden, cell = model.module.initialize(data[:, 0])
        output, (hidden, cell), shape = model.module.encode(data[:, :-1], hidden, cell)
        attention_score, weight = model.module.extract(output, shape, weight=(edges if args.gt else None))  # use final layer's hidden state
        data = data[:, -1]
        data_tmp = torch.cat((data_enc[:, -1].unsqueeze(1), data_dec), dim=1)
        label_diff_list = (data_tmp[:, 1:] - data_tmp[:, :-1]).cuda(args.local_rank)

        for n in range(step_num):
            label = labels[:, n]
            #label_diff = label - data
            label_diff = label_diff_list[:, n]
            (mu, sig), hidden, cell = model.module.decode(data, hidden, cell, weight)
            if args.use_sample:
                sample = sampler(mu, sig).cuda(args.local_rank)
            else:
                sample = mu.cuda(args.local_rank)

            if type(sample)==type(None):
                print("broke_train")
                sample = label_diff[:, n]
                break

            if args.ww:    
                test_loss = torch.mean(torch.sum(criterion(label_diff[:, :, :-1], mu, sig), dim=-1)) # W
            else:
                test_loss = torch.mean(torch.sum(criterion(label_diff, mu, sig), dim=-1))  # noW
            test_losses_list[n].update(test_loss.item(), data.size(0) * agent_num)
            test_losses_list[-1].update(test_loss.item(), data.size(0) * agent_num)

            if args.ww:
                next_data = sample + data[:, :, :-1]
                data = torch.cat((next_data, data[:, :, -1].unsqueeze(-1)), dim=-1)
            else:
                next_data = sample + data
                data = next_data

        del mu, sig, hidden, cell
        del test_loss, sample, next_data, data
        del label_diff, label_diff_list, data_tmp
        torch.cuda.empty_cache()

    return (
        [test_losses_list[i].avg for i in range(len(test_losses_list))],
        [test_losses_list[i].count for i in range(len(test_losses_list))],
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


if __name__ == "__main__":
    print("started!")  # For test
    main()
