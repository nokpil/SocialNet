import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["mlp", "ds", "st"]


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=64, num_layer=2, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.act = activation
        layers = [nn.Linear(dim_in, dim_hidden), self.act]

        for i in range(num_layer - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(self.act)

        layers.append(nn.Linear(dim_hidden, dim_out))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        return self.fc(x.view(batch_size, -1))

class DeepSet(nn.Module):
    # Zaheer et al., NIPS (2017). https://arxiv.org/abs/1703.06114
    def __init__(self, dim_in, dim_out, dim_hidden=128, activation=nn.ReLU()):
        super(DeepSet, self).__init__()
        self.act = activation
        self.enc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        x = self.enc(x).mean(-2)
        x = self.dec(x)
        return x


class RNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer=1, dim_hidden=64, activation=nn.ReLU()):
        super(RNN, self).__init__()
        self.act = activation
        self.n_layer = n_layer
        self.dim_hidden = dim_hidden
        self.dim_rnn = dim_hidden // 2
        self.emb = nn.Embedding(2, self.dim_rnn // 2)
        self.rnn = nn.GRU(self.dim_rnn // 2, self.dim_rnn, n_layer)
        self.enc = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.dim_hidden + self.dim_rnn, self.dim_hidden),
            self.act,
            nn.Linear(self.dim_hidden, dim_out),
        )

    def forward(self, s):
        obs, hist_act = s

        hist_act = self.emb(hist_act)
        os, _ = self.rnn(hist_act)

        hist_act = os[-1]
        obs = self.enc(obs).mean(-2)

        x = torch.cat([obs, hist_act], dim=-1)
        output = self.dec(x)
        return output

class SetTransformer(nn.Module):
    # "Lee, J. et. al. ICML (2019, May), http://proceedings.mlr.press/v97/lee19d/lee19d.pdf "
    def __init__(self, dim_input, dim_out, num_outputs=1, num_inds=32, dim_hidden=128, num_heads=4, ln=False, activation=nn.ReLU):
        super(SetTransformer, self).__init__()
        self.dim_out = dim_out
        self.num_outputs = num_outputs
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln, activation=activation),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln, activation=activation))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln, activation=activation),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, activation=activation),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln, activation=activation),
                nn.Linear(dim_hidden, dim_out))

    def forward(self, X):
        if len(X.shape) > 3: #  Typical input would be (batch, ensemble, )
            input_shape = X.shape
            X = X.reshape(-1, *input_shape[-2:])
            output = self.dec(self.enc(X))
            output_shape = output.shape
            return output.view(*input_shape[:-2], *output_shape[-2:])
        return self.dec(self.enc(X))

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, activation=nn.ReLU()):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.activation = activation

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.activation(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, activation=nn.ReLU()):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, activation=activation)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, activation=nn.ReLU()):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, activation=activation)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, activation=activation)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, activation=nn.ReLU()):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, activation=activation)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


def mlp(obs_dim, dim_out, **kwargs):
    model = MLP(np.prod(obs_dim), dim_out, **kwargs)
    return model


def ds(obs_dim, dim_out, **kwargs):
    model = DeepSet(obs_dim[-1], dim_out, **kwargs)
    return model

def st(obs_dim, dim_out, **kwargs):
    model = SetTransformer(obs_dim[-1], dim_out, **kwargs)
    return model


class PNA(nn.Module):
    # Principal Neighbourhood Aggregation for Graph Nets, G. Corso et al. (2020) https://arxiv.org/pdf/2004.05718.pdf
    def __init__(self, dim_in, dim_out, dim_hidden=128, activation=nn.ReLU()):
        super(PNA, self).__init__()
        self.act = activation
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_hidden),
            self.act,
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        


        x = self.dec(x)
        return x