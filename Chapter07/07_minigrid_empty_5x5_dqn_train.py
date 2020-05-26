#!/usr/bin/env python3

import argparse
import numpy as np
import collections

import ptan
import gym
from gym.wrappers import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import gym_minigrid


DEFAULT_ENV_NAME = "MiniGrid-Empty-5x5-v0"
HIDDEN_SIZE = 16

MEAN_REWARD_BOUND = 0.95

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
TARGET_NET_SYNC = 1000
LEARNING_RATE = 1e-4
EPSILON_DECAY = 0.99
EPSILON_START = 1.0


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
        return self.net(x.float())


def make_env(env_name):
    env = gym.make(env_name)
    env = FlatteningObsWrapper(env)
    env = ReducingActionSpaceWrapper(env)
    return env


@torch.no_grad()
def unpack_batch(batch, net, gamma):
    states = []
    actions = []
    rewards = []
    done_masks = []
    last_states = []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        done_masks.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)

    states_v = torch.tensor(states)
    actions_v = torch.tensor(actions)
    rewards_v = torch.tensor(rewards)
    last_states_v = torch.tensor(last_states)
    last_state_q_v = net(last_states_v)
    best_last_q_v = torch.max(last_state_q_v, dim=1)[0]
    best_last_q_v[done_masks] = 0.0
    return states_v, actions_v, best_last_q_v * gamma + rewards_v


if __name__ == "__main__":

    device = torch.device("cpu")

    env = make_env(DEFAULT_ENV_NAME)
    env = Monitor(env, directory="mon", force=True)

    net = DQN(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    writer = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
    print(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    best_m_reward = None
    step = 0
    solved = False

    while True:

        step += 1
        buffer.populate(1)

        for reward, _ in exp_source.pop_rewards_steps():
            total_rewards.append(reward)
            m_reward = np.mean(total_rewards[-100:])

            writer.add_scalar("epsilon", selector.epsilon, step)
            writer.add_scalar("reward_100", m_reward, step)
            writer.add_scalar("reward", reward, step)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), DEFAULT_ENV_NAME + "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" %
                          (best_m_reward, m_reward))
                best_m_reward = m_reward

            solved = m_reward > MEAN_REWARD_BOUND

        if solved:
            print("Solved in %d steps!" % step)
            break

        # make sure that we have enough samples for subsequent steps
        if len(buffer) < REPLAY_SIZE:
            continue

        batch = buffer.sample(BATCH_SIZE)
        states_v, actions_v, tgt_q_v = \
            unpack_batch(batch, tgt_net.target_model, GAMMA)
        optimizer.zero_grad()
        q_v = net(states_v)
        q_v = q_v.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        loss_v = F.mse_loss(q_v, tgt_q_v)
        loss_v.backward()
        optimizer.step()
        selector.epsilon *= EPSILON_DECAY

        if step % TARGET_NET_SYNC == 0:
            tgt_net.sync()

    writer.close()
