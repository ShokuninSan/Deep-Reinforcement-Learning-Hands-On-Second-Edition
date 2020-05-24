#!/usr/bin/env python3
import gym
import time
import argparse
import numpy as np

import torch
from gym.wrappers import Monitor
import gym_minigrid

from lib.minigrid import DQN, make_env, DEFAULT_ENV_NAME, HIDDEN_SIZE

import collections

FPS = 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default=" +
                             DEFAULT_ENV_NAME)
    parser.add_argument("-r", "--record", help="Directory for video")
    parser.add_argument("--no-vis", default=True, dest='vis',
                        help="Disable visualization",
                        action='store_false')
    args = parser.parse_args()

    env = make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)
    net = DQN(env.observation_space.shape[0], HIDDEN_SIZE, env.action_space.n)
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if args.vis:
            env.render()
        state_v = torch.FloatTensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if args.vis:
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    if args.record:
        env.env.close()

