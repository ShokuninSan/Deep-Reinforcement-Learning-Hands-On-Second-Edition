#!/usr/bin/env python3
import gym
import collections
from tensorboardX import SummaryWriter
import numpy as np
from gym.wrappers import Monitor
from gym_minigrid.wrappers import FullyObsWrapper

ENV_NAME = "MiniGrid-Empty-5x5-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class FlatteningFullyObsWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(FlatteningFullyObsWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0, 255,
            (np.product(env.observation_space['image'].shape[:-1]),),
            dtype='uint8')

    def observation(self, observation):
        return tuple(observation['image'][:, :, 0].reshape(-1,))


class ReducingActionSpaceWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super(ReducingActionSpaceWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(3)

    def action(self, act):
        return act


class Agent:

    def __init__(self):
        self.env = make_env(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1-ALPHA) + new_v * ALPHA

    def random_walk(self):
        self.value_update(*self.sample_env())

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


def make_env(env_name):
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FlatteningFullyObsWrapper(env)
    env = ReducingActionSpaceWrapper(env)
    return env


if __name__ == "__main__":
    test_env = make_env(ENV_NAME)
    test_env = Monitor(test_env, directory='mon', force=True)
    agent = Agent()
    writer = SummaryWriter(comment="-q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.random_walk()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        print(f'Got a reward of {reward} in playing episodes '
              f'for iteration {iter_no}...')
        writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (
                best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    writer.close()
