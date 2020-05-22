#!/usr/bin/env python3

import argparse
import time
import numpy as np
import collections

import gym
from gym.wrappers import Monitor
from gym_minigrid.wrappers import FullyObsWrapper
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter


DEFAULT_ENV_NAME = "MiniGrid-Empty-5x5-v0"
MEAN_REWARD_BOUND = 0.8

HIDDEN_SIZE = 128
GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])


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


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # epsilon-greedy
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.FloatTensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    next_states_v = torch.FloatTensor(np.array(next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
                                   rewards_v
    return nn.MSELoss()(state_action_values,
                        expected_state_action_values)


def make_env(env_name):
    env = gym.make(env_name)
    env = FullyObsWrapper(env)
    env = FlatteningFullyObsWrapper(env)
    env = ReducingActionSpaceWrapper(env)
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default=" +
                             DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = make_env(args.env)

    net = DQN(env.observation_space.shape[0],
              HIDDEN_SIZE,
              env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape[0],
                  HIDDEN_SIZE,
                  env.action_space.n).to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    n_agent_steps = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    while True:
        n_agent_steps += 1

        reward = agent.play_step(net, epsilon, device=device)

        if reward is not None:
            total_rewards.append(reward)

            m_reward = np.mean(total_rewards[-100:])
            print("%d: steps done %d games, reward %.3f, "
                  "eps %.2f" % (
                      n_agent_steps, len(total_rewards), m_reward, epsilon,
            ))

            writer.add_scalar("epsilon", epsilon, n_agent_steps)
            writer.add_scalar("reward_100", m_reward, n_agent_steps)
            writer.add_scalar("reward", reward, n_agent_steps)

            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward

                tgt_net.load_state_dict(net.state_dict())

            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d steps!" % n_agent_steps)
                break

            epsilon = max(EPSILON_FINAL, EPSILON_START -
                          n_agent_steps / EPSILON_DECAY_LAST_FRAME)

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_t = calc_loss(batch, net, tgt_net, device=device)
            loss_t.backward()
            optimizer.step()
    writer.close()
