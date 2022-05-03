import numpy as np
import nkmodel as nk
from envs.base_env import Env, MultiBinary
from scipy.stats import rankdata


class base_landscape:
    def __init__(self, landscape):
        self.landscape = landscape

    def __getitem__(self, index):
        ## ignore index
        return self.landscape

    def __iter__(self):
        return self.landscape

    def __next__(self):
        pass

class SL_NK(Env):
    def __init__(self, E=16, M=100, N=15, K=7, exp=8, neighbor_num=3, graph_type=None, graph=None, graph_dict=None, reward_type=None, action_type=None, extra_type=None, corr_type=None, rescale=False, self_include=True, env_scheduler=None):
        
        # E: number of ensembles
        # M: number of agents (agent_num)
        # N: number of gene dimensions
        # K: system complexity
        # graph : networkx graph object (e.g., nx.complete_graph(), nx.barabasi_alberg_graph(), ...)
        # graph_dict : graph generation parameters (e.g., {'n': 10, 'm' : 3})

        self.E = E
        self.M = M
        self.N = N
        self.K = K
        self.exp = exp
        self.neighbor_num = neighbor_num
        self.graph_type = graph_type
        self.graph_generator = graph
        self.graph_dict = graph_dict
        reward_group_type, reward_calc_type, reward_supply_type = reward_type.split('_')
        self.reward_group_type = reward_group_type
        self.reward_calc_type = reward_calc_type
        self.reward_supply_type = reward_supply_type
        self.reward_constant = 100.
        self.action_type = action_type
        self.extra_type = extra_type
        self.extra_num = len(extra_type)
        self.state_correction = True if corr_type[0] == 'T' else False
        self.reward_correction = True if corr_type[1] == 'T' else False
        self.self_include = self_include
        if env_scheduler:
            self.env_scheduler = env_scheduler

        self.observation_space = MultiBinary((self.E, self.M, self.N))  # E ensembles of M agents with each N dimension vector
        self.action_space = MultiBinary((self.N, 2))  # 0 or 1 for each dimension

        self.landscape = None
        self.states = None
        self.scores = None
        self.score_max = None
        self.graph = None
        self.fixed_states = None
        self.rescale = rescale
        

    def get_obs(self):
        NotImplementedError

    def get_score(self, states=None):
        if states is None:
            states = self.states
        shape = states.shape
        scores = np.zeros((shape[0], shape[1], 1))
        assert ((states == 0) | (states == 1)).all()
        for i in range(shape[0]):
            for j in range(shape[1]):
                scores[i][j][0] = self.landscape.get_value(states[i][j]) / self.score_max
        if self.rescale:
            return scores * self.reward_constant
        else:
            return scores

    def step(self, action):
        raise NotImplementedError

    def reset(self, E=None, states=None, state_only=False, base=False):
        if E:
            self.E = E
            self.observation_space = MultiBinary((self.E, self.M, self.N))

        if states is None:
            self.states = self.observation_space.sample()
        else:
            self.states = states
            #state_only = True

        if not state_only:
            # reset landscape and graph structure
            if base:
                landscape = nk.NK(self.N, self.K, self.exp)
                self.landscape = landscape
            else:
                self.landscape = self.env_scheduler.get_landscape()
            self.graph = self.graph_generator(**self.graph_dict)

        assert ((self.states == 0) | (self.states == 1)).all()
        self.score_max = self.landscape.get_global_max()[1]
        self.scores = self.get_score()
        return self.get_obs(), self.states

    def seed(self, seed=None):
        np.random.seed(seed)
        self.seed = seed
        return seed


class SL_NK_total(SL_NK):
    def __init__(self, E=16, M=100, N=15, K=7, exp=8, neighbor_num=3, graph_type=None, graph=None, graph_dict=None, reward_type=None, action_type=None, extra_type=None, corr_type=None, rescale=False, self_include=True, env_scheduler=None):
        super().__init__(E, M, N, K, exp, neighbor_num, graph_type, graph, graph_dict, reward_type, action_type, extra_type, corr_type, rescale, self_include, env_scheduler)
     
    def get_obs(self):
        neighbors = [list(np.random.permutation(list(self.graph.neighbors(i))))[:self.neighbor_num] for i in range(self.M)]
        states_neighbor = self.states[:, neighbors, :]
        scores_neighbor = self.scores[:, neighbors]
        self_feature = [self.states]
        neighbor_feature = [states_neighbor]
  
        if 'S' in self.extra_type:
            self_feature.append(self.scores)
            neighbor_feature.append(scores_neighbor)
        if 'I' in self.extra_type:
            self_feature.append(np.ones_like(self.scores))
            neighbor_feature.append(np.zeros_like(scores_neighbor))
        if 'R' in self.extra_type:
            scores_concat = np.concatenate([self.scores, scores_neighbor.squeeze(-1)], axis=-1)
            score_rank = rankdata(scores_concat, axis=-1, method='min') / (self.neighbor_num + 1)
            self_feature.append(np.expand_dims(score_rank[..., 0], axis=-1))
            neighbor_feature.append(np.expand_dims(score_rank[..., 1:], axis=-1))
        if 'F' in self.extra_type:
            self_freq = np.zeros_like(self.scores)
            neighbor_freq = np.zeros_like(scores_neighbor)
            for i in range(self.E):
                for j in range(self.M):
                    states_unique, freq_states = np.unique(states_neighbor[i][j], axis=0, return_counts=True)
                    if len(freq_states) < self.neighbor_num:
                        for k in range(self.neighbor_num):  # At least one 'most frequent' state
                            neighbor_freq[i][j][k] = freq_states[np.all(np.equal(states_unique, states_neighbor[i][j][k]), axis=1)]
                    else:
                        neighbor_freq[i][j] = np.ones((self.neighbor_num, 1))

            self_feature.append(self_freq)
            neighbor_feature.append(neighbor_freq / self.neighbor_num)

        states_input = np.concatenate(
            [
                np.expand_dims(np.concatenate(self_feature, axis=-1), axis=-2),
                np.concatenate(neighbor_feature, axis=-1)
            ],
            axis=-2
        )
        return states_input
        
    def step(self, action):
        new_scores = self.get_score(action)
        better_states = (new_scores > self.scores).astype(np.long)
        if self.state_correction:
            self.states = better_states * action + (1 - better_states) * self.states
        else:
            self.states = action
        
        if self.reward_correction:
            new_scores = self.get_score()
        else:
            new_scores = self.get_score(action)
        
        if self.reward_group_type == 'indv':
            rew = new_scores
            now = self.scores
        elif self.reward_group_type == 'pop':
            rew = np.repeat(np.mean(new_scores, axis=-2, keepdims=True), new_scores.shape[-2], -2)
            now = np.mean(self.scores, axis=-2)
        else:
            raise NotImplementedError

        if self.reward_calc_type == 'raw':
            pass
        elif self.reward_calc_type == 'diff':
            rew = rew - now
        else:
            raise NotImplementedError

        if self.rescale:
            rew = rew.squeeze(-1) / self.reward_constant
        else:
            rew = rew.squeeze(-1)
        self.scores = self.get_score()  # Even if reward correction is True, this makes the state and 
        return self.get_obs(), rew, self.scores.squeeze(-1)

class SL_NK_split(SL_NK):
    def __init__(self, E=16, M=100, N=15, K=7, exp=8, neighbor_num=3, graph_type=None, graph=None, graph_dict=None, reward_type=None, action_type=None, extra_type=None, corr_type=None, rescale=False, self_include=True, env_scheduler=None):
        super().__init__(E, M, N, K, exp, neighbor_num, graph_type, graph, graph_dict, reward_type, action_type, extra_type, corr_type, rescale, self_include, env_scheduler)

    def get_obs(self):
        neighbors = [list(np.random.permutation(list(self.graph.neighbors(i))))[:self.neighbor_num] for i in range(self.M)]
        states_neighbor = self.states[:, neighbors, :]
        scores_neighbor = self.scores[:, neighbors].squeeze(-1)
        self_feature = [np.expand_dims(self.states, axis=-1)]
        neighbor_feature = [np.expand_dims(states_neighbor, axis=-1)]
        if 'S' in self.extra_type:
            self_feature.append(np.expand_dims(np.repeat(self.scores, self.N, axis=-1), axis=-1))
            neighbor_feature.append(np.expand_dims(np.repeat(np.expand_dims(scores_neighbor, axis=-1), self.N, axis=-1), axis=-1))
        if 'I' in self.extra_type:
            self_feature.append(np.expand_dims(np.ones_like(self.states), axis=-1))
            neighbor_feature.append(np.expand_dims(np.zeros_like(states_neighbor), axis=-1))
        if 'R' in self.extra_type:
            scores_concat = np.concatenate([self.scores, scores_neighbor], axis=-1)
            score_rank = rankdata(scores_concat, axis=-1, method='min') / (1 + self.neighbor_num)
            self_feature.append(np.repeat(np.expand_dims(score_rank[..., 0], axis=(-2, -1)), repeats=self.N, axis=-2))
            neighbor_feature.append(np.repeat(np.expand_dims(score_rank[..., 1:], axis=(-2, -1)), repeats=self.N, axis=-2))

        states_input = np.concatenate(
            [
                np.expand_dims(np.concatenate(self_feature, axis=-1), axis=-3),
                np.concatenate(neighbor_feature, axis=-1)
            ],
            axis=-3
        ).transpose(0, 1, 3, 2, 4)
        return states_input
        
    def step(self, action):
        
        new_scores = self.get_score(action)
        better_states = (new_scores > self.scores).astype(np.long)
        if self.state_correction:
            self.states = better_states * action + (1 - better_states) * self.states
        else:
            self.states = action
        
        if self.reward_correction:
            new_scores = self.get_score()
        else:
            new_scores = self.get_score(action)

        #print(f'before : {(new_scores-self.scores)[0].squeeze()}')

        if self.reward_group_type == 'indv':
            rew = new_scores
            now = self.scores
        elif self.reward_group_type == 'pop':
            rew = np.repeat(np.mean(new_scores, axis=-2, keepdims=True), repeats=self.M, axis=-2)
            now = np.repeat(np.mean(self.scores, axis=-2, keepdims=True), repeats=self.M, axis=-2)
        else:
            raise NotImplementedError

        if self.reward_calc_type == 'raw':
            pass
        elif self.reward_calc_type == 'diff':
            rew = rew - now
        else:
            raise NotImplementedError

        if self.rescale:
            rew = rew / self.reward_constant

        rew = np.repeat(rew, self.N, axis=-1)
        self.scores = new_scores

        return self.get_obs(), rew, self.scores.squeeze(-1)
