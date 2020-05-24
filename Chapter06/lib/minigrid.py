# -*- coding: utf-8 -*-
import gym
import numpy as np
import torch.nn as nn

DEFAULT_ENV_NAME = "MiniGrid-Empty-5x5-v0"
HIDDEN_SIZE = 16


class FlatteningObsWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(FlatteningObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 255,
            (np.product(env.observation_space['image'].shape[:-1]),),
            dtype='uint8')

    def observation(self, observation):
        return tuple(observation['image'][:, :, 0].reshape(-1,))


class ReducingActionSpaceWrapper(gym.ActionWrapper):
    """
    Reduce actions to:

    left = 0
    right = 1
    forward = 2
    """
    def __init__(self, env):
        super(ReducingActionSpaceWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, act):
        return act


class DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, n_actions):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


def make_env(env_name):
    env = gym.make(env_name)
    env = FlatteningObsWrapper(env)
    env = ReducingActionSpaceWrapper(env)
    return env
