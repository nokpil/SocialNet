import numpy as np


class Env(object):
    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def seed(self, seed=None):
        return

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)


class Space(object):
    def sample(self):
        raise NotImplementedError


class MultiBinary(Space):
    '''
    An n-shape binary space. 
    The argument to MultiBinary defines n, which could be a number or a `list` of numbers.
    
    Example Usage:
    
    >> self.observation_space = spaces.MultiBinary(5)
    >> self.observation_space.sample()
        array([0,1,0,1,0], dtype =int8)
    >> self.observation_space = spaces.MultiBinary([3,2])
    >> self.observation_space.sample()
        array([[0, 0],
               [0, 1],   
               [1, 1]], dtype=int8)
    '''
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(low=0, high=2, size=self.n)

    def __repr__(self):
        return "MultiBinary({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, MultiBinary) and self.n == other.n