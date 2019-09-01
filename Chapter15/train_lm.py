#!/usr/bin/env python3
import gym
import ptan
import pathlib
import argparse
import itertools
import numpy as np
from typing import List
import warnings
from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model, common

import torch
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine


GAMMA = 0.9
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
POLICY_BETA = 0.1

# have to be less or equal to env.action_space.max_length
LM_MAX_TOKENS = 4
LM_MAX_COMMANDS = 10
LM_STOP_AVG_REWARD = -1.0


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
    "policy_commands": True,
    "last_command": True,
}


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast], prep: preproc.Preprocessor):
    states = []
    rewards = []
    not_done_idx = []
    next_states = []

    for idx, exp in enumerate(batch):
        states.append(exp.state['obs'])
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            next_states.append(exp.last_state['obs'])
    return prep.encode_sequences(states)


def batch_generator(exp_source: ptan.experience.ExperienceSourceFirstLast,
                    batch_size: int):
    batch = []
    for exp in exp_source:
        batch.append(exp)
        if len(batch) == batch_size:
            yield batch
            batch.clear()


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    parser.add_argument("--params", choices=list(common.PARAMS.keys()),
                        help="Training params, could be one of %s" % (list(common.PARAMS.keys())))
    parser.add_argument("-s", "--suffices", type=int, default=1,
                        help="Count of game indices to use during training, default=1")
    parser.add_argument("-v", "--validation", default='-val',
                        help="Suffix for game used for validation, default=-val")
    parser.add_argument("--cuda", default=False, action='store_true',
                        help="Use cuda for training")
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    params = common.PARAMS[args.params]

    game_files = ["games/%s%s.ulx" % (args.game, s) for s in range(1, args.suffices+1)]
    if not all(map(lambda p: pathlib.Path(p).exists(), game_files)):
        raise RuntimeError(f"Some game files from {game_files} not found! Probably you need to run make_games.sh")
    env_id = register_games(game_files, request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Registered env %s for game files %s" % (env_id, game_files))
    val_game_file = "games/%s%s.ulx" % (args.game, args.validation)
    val_env_id = register_games([val_game_file], request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Game %s, with file %s will be used for validation" % (val_env_id, val_game_file))

    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env, use_admissible_commands=False,
                                   keep_admissible_commands=True,
                                   reward_wrong_last_command=-0.1)
    prep = preproc.Preprocessor(
        dict_size=env.observation_space.vocab_size,
        emb_size=params.embeddings, num_sequences=env.num_fields,
        enc_output_size=params.encoder_size).to(device)

    cmd = model.CommandModel(prep.obs_enc_size, env.observation_space.vocab_size, prep.emb,
                             max_tokens=LM_MAX_TOKENS,
                             max_commands=LM_MAX_COMMANDS,
                             start_token=env.action_space.BOS_id,
                             sep_token=env.action_space.EOS_id).to(device)
    if False:
        agent = model.CmdAgent(env, cmd, prep, device=device)
        exp_source = ptan.experience.ExperienceSourceFirstLast(
            env, agent, gamma=GAMMA, steps_count=1)
        buffer = ptan.experience.ExperienceReplayBuffer(
            exp_source, params.replay_size)

        optimizer = optim.RMSprop(itertools.chain(prep.parameters(), cmd.parameters()),
                                  lr=LEARNING_RATE, eps=1e-5)

        def process_batch(engine, batch):
            optimizer.zero_grad()
            obs_t = unpack_batch(batch, prep)

            commands = []

            for s in batch:
                cmds = []
                for c in s.state['admissible_commands']:
                    t = env.action_space.tokenize(c)
                    if len(t)-2 <= LM_MAX_TOKENS:
                        cmds.append(t)
                commands.append(cmds)

            loss_t = model.pretrain_policy_loss(cmd, commands, obs_t)
            loss_t.backward()
            optimizer.step()

            if engine.state.metrics.get('avg_reward', LM_STOP_AVG_REWARD) > LM_STOP_AVG_REWARD:
                print("Mean reward reached %.2f, stop pretraining" % LM_STOP_AVG_REWARD)
                engine.should_terminate = True
            return {
                "loss": loss_t.item(),
            }

        engine = Engine(process_batch)
        run_name = f"lm-{args.params}_{args.run}"
        common.setup_ignite(engine, exp_source, run_name)
        engine.run(common.batch_generator(buffer, BATCH_SIZE, BATCH_SIZE))

        torch.save(prep.state_dict(), "prep.dat")
        torch.save(cmd.state_dict(), "cmd.dat")
    prep.load_state_dict(torch.load("prep.dat"))
    cmd.load_state_dict(torch.load("cmd.dat"))

    # DQN training using Preprocessor and Command generator as part of the environment
    val_env = gym.make(val_env_id)
    val_env = preproc.TextWorldPreproc(val_env)

    net = model.DQNModel(obs_size=prep.obs_enc_size,
                         cmd_size=prep.obs_enc_size).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = model.CmdDQNAgent(env, net, cmd, prep, epsilon=1, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, params.replay_size)
    #
    # buffer.experience_source_iter = iter(exp_source)

    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_t = model.calc_loss_dqncmd(
            batch, prep, cmd, net, tgt_net.target_model, GAMMA,
            env, device)
        loss_t.backward()
        optimizer.step()
        eps = 1 - engine.state.iteration / params.epsilon_steps
        agent.epsilon = max(params.epsilon_final, eps)
        if engine.state.iteration % params.sync_nets == 0:
            tgt_net.sync()
        return {
            "loss": loss_t.item(),
            "epsilon": agent.epsilon,
        }

    engine = Engine(process_batch)
    run_name = f"dqn-{args.params}_{args.run}"
    save_path = pathlib.Path("saves") / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    common.setup_ignite(engine, exp_source, run_name,
                        extra_metrics=('val_reward', 'val_steps'))

    # @engine.on(ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)
    # def validate(engine):
    #     reward = 0.0
    #     steps = 0
    #
    #     obs = val_env.reset()
    #
    #     while True:
    #         obs_t = prep.encode_sequences([obs['obs']]).to(device)
    #         cmd_t = prep.encode_commands(obs['admissible_commands']).to(device)
    #         q_vals = net.q_values(obs_t, cmd_t)
    #         act = np.argmax(q_vals)
    #
    #         obs, r, is_done, _ = val_env.step(act)
    #         steps += 1
    #         reward += r
    #         if is_done:
    #             break
    #     engine.state.metrics['val_reward'] = reward
    #     engine.state.metrics['val_steps'] = steps
    #     print("Validation got %.3f reward in %d steps" % (reward, steps))
    #     best_val_reward = getattr(engine.state, "best_val_reward", None)
    #     if best_val_reward is None:
    #         engine.state.best_val_reward = reward
    #     elif best_val_reward < reward:
    #         print("Best validation reward updated: %s -> %s" % (best_val_reward, reward))
    #         save_prep_name = save_path / ("best_val_%.3f_p.dat" % reward)
    #         save_net_name = save_path / ("best_val_%.3f_n.dat" % reward)
    #         torch.save(prep.state_dict(), save_prep_name)
    #         torch.save(net.state_dict(), save_net_name)
    #         engine.state.best_val_reward = reward

    # @engine.on(ptan.ignite.EpisodeEvents.BEST_REWARD_REACHED)
    # def best_reward_updated(trainer: Engine):
    #     reward = trainer.state.metrics['avg_reward']
    #     if reward > 0:
    #         save_prep_name = save_path / ("best_train_%.3f_p.dat" % reward)
    #         save_net_name = save_path / ("best_train_%.3f_n.dat" % reward)
    #         torch.save(prep.state_dict(), save_prep_name)
    #         torch.save(net.state_dict(), save_net_name)
    #         print("%d: best avg training reward: %.3f, saved" % (
    #             trainer.state.iteration, reward))

    engine.run(common.batch_generator(buffer, 128, BATCH_SIZE))
