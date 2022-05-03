import numpy as np
import random
import itertools
import pickle
import nkmodel as nk
import os.path as osp

from user_config import (
    DEFAULT_DATA_DIR
)


class env_scheduler():
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        self.local_rank = local_rank
        self.E = E
        self.N = N
        self.K = K
        self.exp = exp
        self.NGPU = NGPU
        data_dir = data_dir or DEFAULT_DATA_DIR
        self.output_dir = osp.join(data_dir, exp_name, exp_name)
        self.landscape = None
        self.period = 20

    def initialize_landscape(self, value=True, fixed=True):
        landscape_list = None
        if fixed:
            landscape = nk.NK(self.N, self.K, self.exp)
            if value:
                _ = landscape.get_global_max()[1]
            landscape_list = [landscape for _ in range(self.NGPU)]
        else:
            landscape_list = [nk.NK(self.N, self.K, self.exp) for i in range(self.NGPU)]
            if value:
                for i in range(self.NGPU):
                    _ = landscape_list[i].get_global_max()[1]
        with open(self.output_dir + "_landscape_list.pkl", 'wb') as f:
            print('save new landscape list')
            pickle.dump(landscape_list, f)

    def get_landscape(self):
        try:
            with open(self.output_dir + "_landscape_list.pkl", 'rb') as f:
                landscape_list = pickle.load(f)
                self.landscape = landscape_list[self.local_rank]
        except FileNotFoundError as e:
            print(e)
        return self.landscape

    def step(self, epoch):
        raise NotImplementedError


class random_env_scheduler(env_scheduler):
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        super().__init__(local_rank, exp_name, E, N, K, exp, NGPU, data_dir)

    def step(self, epoch):
        print('random : pass')
        pass  # since initialize_landscape does not compute its value (random envsch -> value=False), get_landscape loads empty landscape and randomly initializes by nk_env.reset().


class multifixed_env_scheduler(env_scheduler):
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        super().__init__(local_rank, exp_name, E, N, K, exp, NGPU, data_dir)

    def step(self, epoch):
        print('multifixed : pass')
        pass  # since initialize_landscape does not compute its value (random envsch -> value=True), get_landscape loads pre-determined landscape.


class periodic_env_scheduler(env_scheduler):
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        super().__init__(local_rank, exp_name, E, N, K, exp, NGPU, data_dir)

    def step(self, epoch):
        try:
            print(epoch, self.period, epoch % self.period)
            if epoch % self.period == 0:
                print('env_step')
                self.initialize_landscape(fixed=False)
        except Exception as e:
            print(e)


class gradual_env_scheduler(env_scheduler):
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        super().__init__(local_rank, exp_name, E, N, K, exp, NGPU, data_dir)
        self.change_ratio = 0.01

    def step(self, epoch):
        try:
            #if epoch % self.period == 0:
            if True:
                print('env_step')
                with open(self.output_dir + "_landscape_list.pkl", 'rb') as f:
                    landscape_list = pickle.load(f)
                current_landscape = landscape_list[self.local_rank]

                dependence = current_landscape.dependence
                loci = current_landscape.loci
                values = current_landscape.values
                all_states = itertools.product((0, 1), repeat=current_landscape.N)
                for state in all_states:
                    for n in loci:
                        label = tuple([state[i] for i in dependence[n]])
                        if random.random() < self.change_ratio:
                            v = random.random()
                            values[n][label] = v

                landscape_list = [current_landscape for i in range(self.NGPU)]

                with open(self.output_dir + "_landscape_list.pkl", 'wb') as f:
                    pickle.dump(landscape_list, f)

        except Exception as e:
            print(e)


class roundabout_env_scheduler(env_scheduler):
    def __init__(self, local_rank, exp_name, E, N, K, exp, NGPU, data_dir=None):
        super().__init__(local_rank, exp_name, E, N, K, exp, NGPU, data_dir)

    def step(self, epoch):
        pass
