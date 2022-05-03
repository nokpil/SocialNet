import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import ppo.net as net


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return np.array(
        scipy.signal.lfilter([1], [1, float(-discount)], x[:, :, ::-1], axis=-1)[:, :, ::-1]
    )
    
class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class SocialActor(Actor):
    def __init__(self, obs_dim, act_dim, arch, **kwargs):
        super().__init__()
        self.act_dim = act_dim
        self.logits_net = net.__dict__[arch](obs_dim, np.prod(act_dim), **kwargs)

    def _distribution(self, obs):
        logits = self.logits_net(obs).view(*obs.shape[:-2], *self.act_dim)  # Typically, (batch, ensemble, agent, | N, 2)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class Critic(nn.Module):
    def __init__(self, obs_dim, arch, **kwargs):
        super().__init__()
        self.v_net = net.__dict__[arch](obs_dim, 1, **kwargs)

    def forward(self, obs):
        return self.v_net(obs).squeeze(-1).squeeze(-1)  # Critical to ensure v has right shape.


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, arch, **kwargs):
        super().__init__()
        
        # policy builder depends on action space
        self.pi = SocialActor(obs_dim, act_dim, arch, **kwargs)

        # build value function
        self.v = Critic(obs_dim, arch, **kwargs)

    def step(self, obs, return_pi=False):
        with torch.no_grad():
            pi = self.pi.module._distribution(obs)
            a = pi.sample()
            logp_a = self.pi.module._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        if return_pi:
            return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy(), pi
        else:
            return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class Baseline():
    def __init__(self, env, action_type, extra_type, corr_type):
        self.env = env
        self.action_type = action_type
        self.extra_type = extra_type
        self.extra_num = len(self.extra_type)
        self.state_correction = True if corr_type[0] == 'T' else False
        self.reward_correction = True if corr_type[1] == 'T' else False

    def step(self, obs):
        raise NotImplementedError

    def act(self, obs):
        return NotImplementedError

class FollowBest(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)
            
            stay = (1 - better_social)
            if self.state_correction:
                states = (better_social * states_social) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + stay * scores
            else:
                scores = scores_social

        return states

class FollowBest_indv(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)

            index_indv = np.zeros_like(states)
            np.put_along_axis(index_indv, np.random.randint(0, N, (E, M, 1)), 1, axis=-1)
            states_indv = (states + index_indv) % 2
            scores_indv = self.env.get_score(states=states_indv)

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowBest_random(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)
            
            states_indv = np.random.randint(0, 2, size=states.shape)
            scores_indv = self.env.get_score(states=states_indv)

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowBest_steepest(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)

            states = np.random.randint(0, 2, (E, M, N))
            states_indv_candidate = np.tile(np.expand_dims(states, axis=-2), (1, 1, N, 1))
            index_indv = np.zeros_like(states_indv_candidate)
            np.put_along_axis(index_indv, np.tile(np.arange(N)[np.newaxis, np.newaxis, :, np.newaxis], (E, M, 1, 1)), 1, axis=-1)
            states_indv_candidate = (states_indv_candidate + index_indv) % 2
            scores_indv_candidate = self.env.get_score(states=states_indv_candidate.reshape(-1, N, N)).reshape(E, M, N, 1)
            scores_indv = np.max(scores_indv_candidate, axis=-2)
            index_indv = np.zeros_like(states)
            np.put_along_axis(index_indv, np.argmax(scores_indv_candidate, axis=-2), 1, axis=-1)
            states_indv = (states + index_indv) % 2

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowBest_prob(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)

            index_indv = np.random.binomial(1, 1 / N, size=states.shape)
            states_indv = (states + index_indv) % 2
            scores_indv = self.env.get_score(states=states_indv)

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowMajor(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_social = np.zeros_like(states)
            for i in range(E):
                for j in range(M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.env.neighbor_num:  # At least one 'most frequent' state
                        states_most_frequent = states_unique[freq_states == freq_states.max()]
                        if len(states_most_frequent) == 1:  # Single 'most frequent' state
                            states_social[i][j] = states_most_frequent[0]
                        else:  # Multiple 'most frequent' solutions
                            states_social[i][j] = states_most_frequent[np.random.randint(len(states_most_frequent))]
                    else:  # No frequent state
                        states_social[i][j] = states[i][j]

            scores_social = self.env.get_score(states=states_social)
            better_social = (scores_social > scores).astype(np.long)

            if self.state_correction:
                states = better_social * states_social + (1 - better_social) * states
            else:
                states = states_social

            if self.reward_correction:
                scores = better_social * scores_social + (1 - better_social) * scores
            else:
                scores = scores_social
            
        return states
class FollowMajor_indv(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_social = np.zeros_like(states)

            for i in range(E):
                for j in range(M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.env.neighbor_num:  # At least one 'most frequent' state
                        states_most_frequent = states_unique[freq_states == freq_states.max()]
                        if len(states_most_frequent) == 1 :  # Single 'most frequent' state
                            states_social[i][j] = states_most_frequent[0]
                        else:  # Multiple 'most frequent' solutions
                            states_social[i][j] = states_most_frequent[np.random.randint(len(states_most_frequent))]
                    else:  # indv learning
                        states_social[i][j] = states[i][j]
                        
            scores_social = self.env.get_score(states=states_social)
            better_social = (scores_social > scores).astype(np.long)
            
            index_indv = np.zeros_like(states)
            np.put_along_axis(index_indv, np.random.randint(0, N, (E, M, 1)), 1, axis=-1)
            states_indv = (states + index_indv) % 2
            scores_indv = self.env.get_score(states=states_indv)
            
            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowMajor_random(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_social = np.zeros_like(states)

            for i in range(E):
                for j in range(M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.env.neighbor_num:  # At least one 'most frequent' state
                        states_most_frequent = states_unique[freq_states == freq_states.max()]
                        if len(states_most_frequent) == 1 :  # Single 'most frequent' state
                            states_social[i][j] = states_most_frequent[0]
                        else:  # Multiple 'most frequent' solutions
                            states_social[i][j] = states_most_frequent[np.random.randint(len(states_most_frequent))]
                    else:  # indv learning
                        states_social[i][j] = states[i][j]
                        
            scores_social = self.env.get_score(states=states_social)
            better_social = (scores_social > scores).astype(np.long)

            states_indv = np.random.randint(0, 2, size=states.shape)
            scores_indv = self.env.get_score(states=states_indv)
            
            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowMajor_steepest(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_social = np.zeros_like(states)

            for i in range(E):
                for j in range(M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.env.neighbor_num:  # At least one 'most frequent' state
                        states_most_frequent = states_unique[freq_states == freq_states.max()]
                        if len(states_most_frequent) == 1 :  # Single 'most frequent' state
                            states_social[i][j] = states_most_frequent[0]
                        else:  # Multiple 'most frequent' solutions
                            states_social[i][j] = states_most_frequent[np.random.randint(len(states_most_frequent))]
                    else:  # indv learning
                        states_social[i][j] = states[i][j]
                        
            scores_social = self.env.get_score(states=states_social)
            better_social = (scores_social > scores).astype(np.long)

            states = np.random.randint(0, 2, (E, M, N))
            states_indv_candidate = np.tile(np.expand_dims(states, axis=-2), (1, 1, N, 1))
            index_indv = np.zeros_like(states_indv_candidate)
            np.put_along_axis(index_indv, np.tile(np.arange(N)[np.newaxis, np.newaxis, :, np.newaxis], (E, M, 1, 1)), 1, axis=-1)
            states_indv_candidate = (states_indv_candidate + index_indv) % 2
            scores_indv_candidate = self.env.get_score(states=states_indv_candidate.reshape(-1, N, N)).reshape(E, M, N, 1)
            scores_indv = np.max(scores_indv_candidate, axis=-2)
            index_indv = np.zeros_like(states)
            np.put_along_axis(index_indv, np.argmax(scores_indv_candidate, axis=-2), 1, axis=-1)
            states_indv = (states + index_indv) % 2
            
            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowMajor_prob(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_social = np.zeros_like(states)

            for i in range(E):
                for j in range(M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.env.neighbor_num:  # At least one 'most frequent' state
                        states_most_frequent = states_unique[freq_states == freq_states.max()]
                        if len(states_most_frequent) == 1 :  # Single 'most frequent' state
                            states_social[i][j] = states_most_frequent[0]
                        else:  # Multiple 'most frequent' solutions
                            states_social[i][j] = states_most_frequent[np.random.randint(len(states_most_frequent))]
                    else:  # indv learning
                        states_social[i][j] = states[i][j]
                        
            scores_social = self.env.get_score(states=states_social)
            better_social = (scores_social > scores).astype(np.long)

            index_indv = np.random.binomial(1, 1 / N, size=states.shape)
            states_indv = (states + index_indv) % 2
            scores_indv = self.env.get_score(states=states_indv)
            
            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states


class IndvLearning(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_indv = np.zeros_like(states)
            for i in range(E):
                for j in range(M):
                    indv_index = np.random.randint(N)
                    states_indv[i][j] = states[i][j]
                    states_indv[i][j][indv_index] = (states_indv[i][j][indv_index] + 1) % 2

            scores_indv = self.env.get_score(states=states_indv)
            better_indv = (scores_indv > scores).astype(np.long)

            if self.state_correction:
                states = better_indv * states_indv + (1 - better_indv) * states
            else:
                states = states_indv

            if self.reward_correction:
                scores = better_indv * scores_indv + (1 - better_indv) * scores
            else:
                scores = scores_indv

        return states

class IndvRandom(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_indv = np.random.randint(0, 2, size=states.shape)
            scores_indv = self.env.get_score(states=states_indv)
            better_indv = (scores_indv > scores).astype(np.long)

            if self.state_correction:
                states = better_indv * states_indv + (1 - better_indv) * states
            else:
                states = states_indv

            if self.reward_correction:
                scores = better_indv * scores_indv + (1 - better_indv) * scores
            else:
                scores = scores_indv
        return states

class IndvSteepest(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states = np.random.randint(0, 2, (E, M, N))
            states_indv_candidate = np.tile(np.expand_dims(states, axis=-2), (1, 1, N, 1))
            index_indv = np.zeros_like(states_indv_candidate)
            np.put_along_axis(index_indv, np.tile(np.arange(N)[np.newaxis, np.newaxis, :, np.newaxis], (E, M, 1, 1)), 1, axis=-1)
            states_indv_candidate = (states_indv_candidate + index_indv) % 2
            scores_indv_candidate = self.env.get_score(states=states_indv_candidate.reshape(-1, N, N)).reshape(E, M, N, 1)
            scores_indv = np.max(scores_indv_candidate, axis=-2)
            index_indv = np.zeros_like(states)
            np.put_along_axis(index_indv, np.argmax(scores_indv_candidate, axis=-2), 1, axis=-1)
            states_indv = (states + index_indv) % 2
            better_indv = (scores_indv > scores).astype(np.long)

            if self.state_correction:
                states = better_indv * states_indv + (1 - better_indv) * states
            else:
                states = states_indv

            if self.reward_correction:
                scores = better_indv * scores_indv + (1 - better_indv) * scores
            else:
                scores = scores_indv

        return states

class IndvProb(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            index_indv = np.random.binomial(1, 1 / N, size=states.shape)
            states_indv = (states + index_indv) % 2
            scores_indv = self.env.get_score(states=states_indv)
            better_indv = (scores_indv > scores).astype(np.long)

            if self.state_correction:
                states = better_indv * states_indv + (1 - better_indv) * states
            else:
                states = states_indv

            if self.reward_correction:
                scores = better_indv * scores_indv + (1 - better_indv) * scores
            else:
                scores = scores_indv

        return states

class RandomCopy(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            random_position = np.expand_dims(np.random.randint(0, scores_neighbor.shape[-1], size=scores_neighbor.shape[:-1]), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(random_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, random_position, axis=-1)
            random_social = np.expand_dims((np.sum(scores_neighbor > scores, axis=-1) > 0).astype(np.long), axis=-1)

            stay = (1 - random_social)

            if self.state_correction:
                states = (random_social * states_social) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * random_social) + stay * scores
            else:
                scores = scores_social
 
        return states


class RL_Inspired_SLSs(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type, mod_type):
        # mod_type : 45ST80
        super().__init__(env, action_type, extra_type, corr_type)
        self.boundary = int(mod_type[:2])
        self.indv_type = mod_type[2]
        self.freeze = True if mod_type[3] == 'T' else False
        if len(mod_type) < 5:
            self.ceiling = 100
        else:
            self.ceiling = int(mod_type[4:])
        self.landscape = env.landscape
        self.score_max = env.score_max
        #print(self.boundary, self.indv_type, self.freeze, self.ceiling)

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            states_freeze = np.zeros_like(states).astype(np.long)
            scores_freeze = self.env.get_score(states=states_freeze)
            better_freeze = (scores_freeze > scores).astype(np.long)

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = (scores_social > scores).astype(np.long)

            if self.indv_type == 'S':
                index_indv = np.zeros_like(states)
                np.put_along_axis(index_indv, np.random.randint(0, N, (E, M, 1)), 1, axis=-1)
                states_indv = (states + index_indv) % 2
            elif self.indv_type == 'E':
                states_indv = np.random.randint(0, 2, size=states.shape)
            else:
                raise NotImplementedError

            scores_indv = self.env.get_score(states=states_indv)
            better_indv = (scores_indv > scores).astype(np.long)

            self_boundary = scores > self.boundary
            scores_second = np.expand_dims(np.sort(scores_neighbor.squeeze())[:, :, -2], axis=-1)
            second_boundary = np.logical_or((scores_second > self.boundary), (scores_social > self.ceiling))

            if self.freeze:
                select_freeze = self_boundary * better_freeze
                select_social = second_boundary * better_social * (1 - self_boundary)
                select_indv = better_indv * (1 - select_social) * (1 - self_boundary)
                stay = (1 - select_social) * (1 - select_indv) * (1 - select_freeze)

                assert ((states == 0) | (states == 1)).all()
                assert ((states_social == 0) | (states_social == 1)).all()
                assert ((states_indv == 0) | (states_indv == 1)).all()

                if self.state_correction:
                    states = (select_freeze * states_freeze) + (select_social * states_social) + (select_indv * states_indv) + stay * states
                else:
                    states = states_social

                if self.reward_correction:
                    scores = (select_freeze * scores_freeze) + (select_social * scores_social) + (select_indv * scores_indv) + stay * scores
                else:
                    scores = scores_social

            else:
                select_social = second_boundary * better_social
                select_indv = better_indv * (1 - select_social)
                stay = (1 - select_social) * (1 - select_indv)

                if self.state_correction:
                    states = (select_social * states_social) + (select_indv * states_indv) + stay * states
                else:
                    states = states_social

                if self.reward_correction:
                    scores = (select_social * scores_social) + (select_indv * scores_indv) + stay * scores
                else:
                    scores = scores_social

        return states

class FollowBest_p20(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type, score_offset):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max
        self.score_offset = score_offset

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > (scores + self.score_offset), axis=-1) > 0).astype(np.long), axis=-1)
            
            states_indv = np.random.randint(0, 2, size=states.shape)
            scores_indv = self.env.get_score(states=states_indv)

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

class FollowBest_p15(Baseline):
    def __init__(self, env, action_type, extra_type, corr_type):
        super().__init__(env, action_type, extra_type, corr_type)
        self.landscape = env.landscape
        self.score_max = env.score_max
        self.score_offset = 0.15

    def step(self, obs):
        with torch.no_grad():
            states_input = obs
            if self.action_type == 'total':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[3] - self.extra_num
                states = states_input[:, :, :, :N]
                states_neighbor = states[:, :, 1:, :]
                states = states[:, :, 0, :]
                scores = np.expand_dims(states_input[:, :, 0, N], axis=-1)
                scores_neighbor = states_input[:, :, 1:, N]
            elif self.action_type == 'split':
                E = obs.shape[0]
                M = obs.shape[1]
                N = obs.shape[2]
                states = states_input[:, :, :, :, 0]
                states_neighbor = states[:, :, :, 1:].transpose(0, 1, 3, 2)
                states = states[:, :, :, 0]
                scores = np.expand_dims(states_input[:, :, 0, 0, 1], axis=-1)
                scores_neighbor = states_input[:, :, 0, 1:, 1]
            else:
                raise NotImplementedError

            better_position = np.expand_dims(scores_neighbor.argmax(axis=-1), axis=-1)
            states_social = np.take_along_axis(states_neighbor, np.expand_dims(better_position, axis=-1).repeat(N, axis=-1), axis=-2).squeeze(-2)
            scores_social = np.take_along_axis(scores_neighbor, better_position, axis=-1)
            better_social = np.expand_dims((np.sum(scores_neighbor > (scores + self.score_offset), axis=-1) > 0).astype(np.long), axis=-1)
            
            states_indv = np.random.randint(0, 2, size=states.shape)
            scores_indv = self.env.get_score(states=states_indv)

            better_indv = (scores_indv > scores).astype(np.long) * (1 - better_social)  # not better social but better indv
            stay = (1 - better_social) * (1 - better_indv)
            if self.state_correction:
                states = (better_social * states_social) + (better_indv * states_indv) + stay * states
            else:
                states = states_social

            if self.reward_correction:
                scores = (scores_social * better_social) + (scores_indv * better_indv) + stay * scores
            else:
                scores = scores_social

        return states

