import argparse
import os
import random
import sys
import time
import yaml

import numpy as np
import torch
import dill
import wandb

from baselines import logger

from dcpg.algos import *
from dcpg.envs import make_envs
from dcpg.models import *
from dcpg.sample_utils import sample_episodes
from dcpg.storages import RolloutStorage
from dcpg.buffer import FIFOStateBuffer, RNDFIFOStateBuffer, RNDStateBuffer, LevelsRNDStateBuffer
from test import evaluate
from dcpg.render_utils import render_starting_states

DEBUG = False

def main(config):
    print('Starting!!')
    # Fix random seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # CUDA setting
    torch.set_num_threads(1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    if not config["debug"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

    # Create environments
    envs = make_envs(
        num_envs=config["num_processes"],
        env_name=config["env_name"],
        num_levels=config["num_levels"],
        start_level=config["start_level"],
        distribution_mode=config["distribution_mode"],
        normalize_reward=config["normalize_reward"],
        device=device,
    )
    obs_space = envs.observation_space
    action_space = envs.action_space

    # Create actor-critic
    actor_critic_class = getattr(sys.modules[__name__], config["actor_critic_class"])
    actor_critic_params = config["actor_critic_params"]
    actor_critic = actor_critic_class(
        obs_space.shape, action_space.n, **actor_critic_params
    )
    actor_critic.to(device)
    print("\nActor-Critic Network:", actor_critic)

    # Create rollout storage
    rollouts = RolloutStorage(
        config["num_steps"], config["num_processes"], obs_space.shape, action_space
    )
    rollouts.to(device)

    # Create state buffer
    if config["buffer_type"] == "LevelRND":
        rnd_config = dict()
        rnd_config['rnd_embed_dim'] = 512
        rnd_config['rnd_kwargs'] = dict(activation_fn = torch.nn.ReLU, net_arch=[1024], learning_rate=0.0001)
        rnd_config['rnd_flatten_input'] = True
        rnd_config['rnd_use_resnet'] = True
        rnd_config['rnd_normalize_images'] = False
        rnd_config['rnd_normalize_output'] = False
        config["rnd_config"] = rnd_config
        state_buffer = LevelsRNDStateBuffer(
            config["state_buffer_size"], device, obs_space, action_space, config['num_levels'], config['start_level'], rnd_config, epochs=config["rnd_epoch"], add_batch_size=config["add_batch_size"],
        )

    elif config["buffer_type"] == "FIFO":
        if config["use_rnd"]:
            rnd_config = dict()
            rnd_config['rnd_embed_dim'] = 512
            rnd_config['rnd_kwargs'] = dict(activation_fn = torch.nn.ReLU, net_arch=[1024], learning_rate=0.0001)
            rnd_config['rnd_flatten_input'] = True
            rnd_config['rnd_use_resnet'] = True
            rnd_config['rnd_normalize_images'] = False
            rnd_config['rnd_normalize_output'] = False
            config["rnd_config"] = rnd_config

            state_buffer = RNDFIFOStateBuffer(
                config["state_buffer_size"], device, obs_space, action_space, config['num_levels'], config['start_level'], rnd_config, rnd_epoch=config["rnd_epoch"],
            )
        else:
            state_buffer = FIFOStateBuffer(
                config["state_buffer_size"], device, obs_space, config['num_levels'], config['start_level'],
            )
    elif config["buffer_type"] == "RND":
        rnd_config = dict()
        rnd_config['rnd_embed_dim'] = 512
        rnd_config['rnd_kwargs'] = dict(activation_fn = torch.nn.ReLU, net_arch=[1024], learning_rate=0.0001)
        rnd_config['rnd_flatten_input'] = True
        rnd_config['rnd_use_resnet'] = True
        rnd_config['rnd_normalize_images'] = False
        rnd_config['rnd_normalize_output'] = False
        config["rnd_config"] = rnd_config
        state_buffer = RNDStateBuffer(
            config["state_buffer_size"], device, obs_space, action_space, config['num_levels'], config['start_level'], rnd_config, epochs=config["rnd_epoch"], add_batch_size=config["add_batch_size"],
        )

    # Create agent
    agent_class = getattr(sys.modules[__name__], config["agent_class"])
    agent_params = config["agent_params"]
    agent = agent_class(actor_critic, **agent_params, device=device)

    # Create logger
    log_file = "-{}-{}-s{}".format(
        config["env_name"], config["exp_name"], config["seed"]
    )
    if config["debug"]:
        log_file += "-debug"
    logger.configure(
        dir=config["log_dir"], format_strs=["csv", "wandb"], log_suffix=log_file,
        project_name=config['project_name'], model_name=config['env_name'] + " - " + config['model_name'], wandb_dir=config['wandb_dir'], args=config,
    )
    print("\nLog File:", log_file)

    # Initialize environments
    obs = envs.reset()
    *_, infos = envs.step_wait()
    levels = torch.LongTensor([info["level_seed"] for info in infos])
    states = envs.get_states()
    rollouts.obs[0].copy_(obs)
    rollouts.levels[0].copy_(levels)
    rollouts.states[0] = states

    # Train actor-critic
    num_env_steps_epoch = config["num_steps"] * config["num_processes"]
    num_updates = int(config["num_env_steps"]) // num_env_steps_epoch
    elapsed_time = 0

    for j in range(num_updates):
        # Start training
        start = time.time()

        # Set actor-critic to train mode
        actor_critic.train()

        # Sample episode
        prob_to_teleport = 0
        if j*num_env_steps_epoch > config["teleport_warmup"]:
            prob_to_teleport = config["prob_to_teleport"]
        sample_episodes(envs, rollouts, state_buffer, actor_critic, prob_to_teleport)

        # Add states to the state buffer
        state_buffer.add(rollouts["obs"], rollouts["states"], rollouts["levels"])

        # Compute return
        with torch.no_grad():
            next_critic_outputs = actor_critic.forward_critic(rollouts.obs[-1])
            next_value = next_critic_outputs["value"]
        rollouts.compute_returns(next_value, config["gamma"], config["gae_lambda"])
        rollouts.compute_advantages()

        # Update actor-critic
        train_statistics = agent.update(rollouts)

        # Reset rollout storage
        rollouts.after_update()

        # End training
        end = time.time()
        elapsed_time += end - start

        # Statistics
        if j % config["log_interval"] == 0:
            # Render a sample of states from the state buffer
            if (j // config["log_interval"]) % 10 == 0:
                render_starting_states(state_buffer)

            # Compute state buffer logging metrics
            fraction_of_unique_states, fraction_of_unique_obs, fraction_of_level_coverage, levels = state_buffer.compute_logging_metrics()
            logger.logkv("buffer/fraction_of_unique_states", fraction_of_unique_states)
            logger.logkv("buffer/fraction_of_unique_obs", fraction_of_unique_obs)
            logger.logkv("buffer/fracion_of_level_coverage", fraction_of_level_coverage)
            wandb.log({"buffer/level_distribution": wandb.Histogram(np_histogram=np.histogram(levels, bins=config["num_levels"], density=True))})

            # Train statistics
            total_num_steps = (j + 1) * config["num_processes"] * config["num_steps"]
            time_per_epoch = elapsed_time / (j + 1)

            print(
                "\nUpdate {}, step {}, time per epoch {:.2f} \n".format(
                    j, total_num_steps, time_per_epoch
                )
            )

            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/time_per_epoch", time_per_epoch)
            for key, val in train_statistics.items():
                logger.logkv("train/{}".format(key), val)

            # Fetch reward normalizing variables
            norm_infos = envs.normalization_infos()

            # Evaluate actor-critic on train environments
            train_eval_statistics, train_value_statistics = evaluate(
                config, actor_critic, device, test_envs=False, norm_infos=norm_infos
            )
            train_episode_rewards = train_eval_statistics["episode_rewards"]
            train_episode_steps = train_eval_statistics["episode_steps"]

            print(
                "Last {} training episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(train_episode_rewards),
                    np.mean(train_episode_rewards),
                    np.median(train_episode_rewards),
                    np.std(train_episode_rewards),
                    np.mean(train_episode_steps),
                    np.median(train_episode_steps),
                    np.std(train_episode_steps),
                )
            )

            logger.logkv("train/mean_episode_reward", np.mean(train_episode_rewards))
            logger.logkv("train/med_episode_reward", np.median(train_episode_rewards))
            logger.logkv("train/std_episode_reward", np.std(train_episode_rewards))
            logger.logkv("train/mean_episode_step", np.mean(train_episode_steps))
            logger.logkv("train/med_episode_step", np.median(train_episode_steps))
            logger.logkv("train/std_episode_step", np.std(train_episode_steps))

            for key, val in train_value_statistics.items():
                logger.logkv("train/{}".format(key), val)

            # Evaluate actor-critic on test environments
            test_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=True
            )
            test_episode_rewards = test_eval_statistics["episode_rewards"]
            test_episode_steps = test_eval_statistics["episode_steps"]

            print(
                "Last {} test episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(test_episode_rewards),
                    np.mean(test_episode_rewards),
                    np.median(test_episode_rewards),
                    np.std(test_episode_rewards),
                    np.mean(test_episode_steps),
                    np.median(test_episode_steps),
                    np.std(test_episode_steps),
                )
            )

            logger.logkv("test/mean_episode_reward", np.mean(test_episode_rewards))
            logger.logkv("test/med_episode_reward", np.median(test_episode_rewards))
            logger.logkv("test/std_episode_reward", np.std(test_episode_rewards))
            logger.logkv("test/mean_episode_step", np.mean(test_episode_steps))
            logger.logkv("test/med_episode_step", np.median(test_episode_steps))
            logger.logkv("test/std_episode_step", np.std(test_episode_steps))

            logger.dumpkvs()

        if j == num_updates - 1 and not config["debug"]:
            print("\nFinal evaluation \n")

            # Evaluate actor-critic on train environments
            train_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=False
            )
            train_episode_rewards = train_eval_statistics["episode_rewards"]
            train_episode_steps = train_eval_statistics["episode_steps"]

            print(
                "Last {} train episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(train_episode_rewards),
                    np.mean(train_episode_rewards),
                    np.median(train_episode_rewards),
                    np.std(train_episode_rewards),
                    np.mean(train_episode_steps),
                    np.median(train_episode_steps),
                    np.std(train_episode_steps),
                )
            )

            # Save train scores
            np.save(
                os.path.join(config["output_dir"], "scores-train{}.npy".format(log_file)),
                np.array(train_episode_rewards),
            )

            # Evaluate actor-critic on test environments
            test_eval_statistics, *_ = evaluate(
                config, actor_critic, device, test_envs=True
            )
            test_episode_rewards = test_eval_statistics["episode_rewards"]
            test_episode_steps = test_eval_statistics["episode_steps"]

            print(
                "Last {} test episodes: \n"
                "mean/med/std reward {:.2f}/{:.2f}/{:.2f}, "
                "mean/med/std step {:.2f}/{:.2f}/{:.2f} \n".format(
                    len(test_episode_rewards),
                    np.mean(test_episode_rewards),
                    np.median(test_episode_rewards),
                    np.std(test_episode_rewards),
                    np.mean(test_episode_steps),
                    np.median(test_episode_steps),
                    np.std(test_episode_steps),
                )
            )

            # Save test scores
            np.save(
                os.path.join(config["output_dir"], "scores-test{}.npy".format(log_file)),
                np.array(test_episode_rewards),
            )

            # Save checkpoint
            torch.save(
                actor_critic.state_dict(),
                os.path.join(config["save_dir"], "agent{}.pt".format(log_file)),
            )

            # Save states buffer
            with open(os.path.join(config["save_dir"], "state_buffer{}.pl".format(log_file)), 'wb') as file:
                dill.dump({"observations": state_buffer.obs.detach().cpu().numpy(), "states": np.array(state_buffer.states)}, file)


if __name__ == "__main__":
    print('starting!!')
    if not DEBUG:
        # Argument
        parser = argparse.ArgumentParser()

        parser.add_argument("--exp_name", type=str, required=True)
        parser.add_argument("--env_name", type=str, required=True)
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--debug", action="store_true")

        args = parser.parse_args()

    # Load config
    if DEBUG:
        # config_file = open("configs/{}.yaml".format('dcpg'), "r")
        config_file = open("configs/{}.yaml".format('ppo'), "r")
    else:
        #config_file = open("configs/{}.yaml".format(args.exp_name), "r")
        config_file = open(args.config, "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Update config
    if DEBUG:
        # config["exp_name"] = 'dcpg'
        config["exp_name"] = 'ppo'
        config["env_name"] = 'maze'
        config["seed"] = 1
        config["debug"] = False
        # config["project_name"] = "debugging"
        config["log_dir"] = config["log_dir"][1:]
        config["output_dir"] = config["output_dir"][1:]
        config["save_dir"] = config["save_dir"][1:]
        config["wandb_dir"] = config["wandb_dir"][1:]
    else:
        config["exp_name"] = args.exp_name
        config["env_name"] = args.env_name
        config["seed"] = args.seed
        config["debug"] = args.debug

        # NEW STUFF!
        config["log_dir"] = config["log_dir"][1:]
        config["output_dir"] = config["output_dir"][1:]
        config["save_dir"] = config["save_dir"][1:]
        config["wandb_dir"] = config["wandb_dir"][1:]

    # Run main
    main(config)
