import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

import pickle
import numpy as np
import networkx as nx
import ppo.core as core


def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)


class mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    y_pred, y_true = y_pred.flatten(), y_true.flatten()
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

class DataGen():
    def __init__(self, env, baseline_type, batch_size, batch_num):
        self.system = system(env, baseline_type, batch_size, batch_num)

    def run(self, file_name, total_size, batch_size, train_ratio):

        train_image = []
        train_label = []
        test_image = []
        test_label = []

        batch_num = int(total_size / batch_size)

        for i in range(batch_num):
            for j in range(batch_size):
                data, answer = next(self.system)
                if j < batch_size * train_ratio:
                    train_image.append(data.astype(float))
                    train_label.append(answer)
                else:
                    test_image.append(data.astype(float))
                    test_label.append(answer)

        train_output = {'Image': train_image, 'Label': train_label}
        test_output = {'Image': test_image, 'Label': test_label}

        # Output pickle
        with open('./data/supervised/' + file_name + '_train.pkl', 'wb') as f:
            pickle.dump(train_output, f)

        with open('./data/supervised/' + file_name + '_test.pkl', 'wb') as f:
            pickle.dump(test_output, f)


def system(env, baseline_type, batch_size, batch_num):
    while(True):
        ac_base = core.__dict__[baseline_type](env, env.action_type, env.extra_type, corr_type='TT')
        if baseline_type == 'FollowMajor':
            states = env.observation_space.sample()
            MOD_CONST = 20
            assert states.shape[1] % MOD_CONST == 0
            states = np.repeat(states[:, :int(states.shape[1] / MOD_CONST), :], MOD_CONST, axis=1)
            o, _ = env.reset(E=batch_num, states=states, base=True)
        else:
            o, _ = env.reset(E=batch_num, base=True)
        a = ac_base.step(o)
        yield o, a


def max_mean_clustering_network(n=100):
    assert n % 5 == 0
    s = int(n / 5)
    A = 1 - np.eye(s)
    B = np.zeros((s, s))
    C = np.block([[A, B, B, B, B],
                 [B, A, B, B, B],
                 [B, B, A, B, B],
                 [B, B, B, A, B],
                 [B, B, B, B, A]])
    for i in range(5):
        j = s * i
        C[j][j + s - 1] = 0
        C[j + s - 1][j] = 0
        C[j][j - 1] = 1
        C[j - 1][j] = 1

    G = nx.from_numpy_matrix(C)
    return G


def real_network(network_data, network_index=0):
    graph = nx.Graph() 
    graph.add_edges_from([tuple(x) for x in network_data[network_data['network_index'] == network_index]['edges_id'].values[0]]) # add weights to the edges
    graph = nx.k_core(graph, k=3)
    graph = nx.relabel.convert_node_labels_to_integers(graph, first_label=0, ordering='default')
    return graph


class RunningMeanStd():
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = 0.
        self.var = 1.
        self.count = epsilon
    
    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # pylint: disable=attribute-defined-outside-init
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RewardNormalizer:
    """
    Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
    section 9.3 (which is based on our Baselines code, haha)
    Motivation is that we'd rather normalize the returns = sum of future rewards,
    but we haven't seen the future yet. So we assume that the time-reversed rewards
    have similar statistics to the rewards, and normalize the time-reversed rewards.
    """

    def __init__(self, ensemble_num, agent_num, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False):
        ret_rms_shape = (ensemble_num, agent_num,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.ret = np.zeros((ensemble_num, agent_num))
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward, first=first, gamma=self.gamma
        )
        self.ret = rets[..., -1]
        self.ret_rms.update(rets if self.per_env else rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        return np.clip(
            reward / np.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )

def backward_discounted_sum(
    prevret,
    reward,
    first,
    gamma,
):
    _, _, nstep = reward.shape
    ret = np.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[..., t] = reward[..., t] + (1 - first[..., t]) * gamma * prevret
    return ret


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)