import argparse
import os
import random
import sys
import time
import yaml

import numpy as np
import torch

from baselines import logger

from dcpg.algos import *
from dcpg.envs import make_envs
from dcpg.models import *
from dcpg.sample_utils import sample_episodes
from dcpg.storages import RolloutStorage
from dcpg.rnd import RandomNetworkDistillationState, RandomNetworkDistillationStateAction
from test import evaluate

DEBUG = False

def main(config):
    # Fix random seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

    # CUDA setting
    torch.set_num_threads(1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    # device = torch.device("cpu")

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    if not config["debug"]:
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

    # Create logger
    log_file = "-{}-{}-s{}".format(
        config["env_name"], config["exp_name"], config["seed"]
    )
    if config["debug"]:
        log_file += "-debug"
    # logger.configure(
    #     dir=config["log_dir"], format_strs=["csv", "stdout"], log_suffix=log_file
    # )
    logger.configure(
        dir=config["log_dir"], format_strs=["csv", "wandb"], log_suffix=log_file,
        project_name=config['project_name'], model_name=config['env_name'] + " - " + config['model_name'], args=config,
    )
    print("\nLog File:", log_file)

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

    # Create pure exploration actor-critic
    pure_actor_critic = actor_critic_class(
        obs_space.shape, action_space.n, **actor_critic_params
    )
    pure_actor_critic.to(device)
    print("\nPure Exploration Actor-Critic Network:", pure_actor_critic)

    # Create rollout storage
    rollouts = RolloutStorage(
        config["num_steps"], config["num_processes"], obs_space.shape, action_space
    )
    rollouts.to(device)

    # Create pure exploration rollout storage
    pure_rollouts = RolloutStorage(
        config["num_steps"], config["num_processes"], obs_space.shape, action_space
    )
    pure_rollouts.to(device)

    # Create agent
    agent_class = getattr(sys.modules[__name__], config["agent_class"])
    agent_params = config["agent_params"]
    agent = agent_class(actor_critic, **agent_params, device=device)

    # Create pure exploration agent
    pure_agent = agent_class(pure_actor_critic, **agent_params, device=device)

    # Create Random Network Distillation
    config['rnd_embed_dim'] = 512
    config['rnd_kwargs'] = dict(activation_fn = torch.nn.ReLU, net_arch=[1024], learning_rate=0.0001)
    config['rnd_flatten_input'] = True
    config['rnd_use_resnet'] = True
    config['rnd_normalize_images'] = False
    config['rnd_normalize_output'] = True
    config['rnd_next_state'] = True

    if config['rnd_next_state']:
        rnd = RandomNetworkDistillationState(
            envs.action_space,
            envs.observation_space,
            config['rnd_embed_dim'],
            config['rnd_kwargs'],
            device=device,
            flatten_input=config['rnd_flatten_input'],
            use_resnet=config['rnd_use_resnet'],
            normalize_images=config['rnd_normalize_images'],
            normalize_output=config['rnd_normalize_output'],
            )
    else:
        rnd = RandomNetworkDistillationStateAction(
            envs.action_space,
            envs.observation_space,
            config['rnd_embed_dim'],
            config['rnd_kwargs'],
            device=device,
            flatten_input=config['rnd_flatten_input'],
            use_resnet=config['rnd_use_resnet'],
            normalize_images=config['rnd_normalize_images'],
            normalize_output=config['rnd_normalize_output'],
            )

    # Initialize environments
    obs = envs.reset()
    *_, infos = envs.step_wait()
    levels = torch.LongTensor([info["level_seed"] for info in infos]).to(device)

    # Train actor-critic
    num_env_steps_epoch = config["num_steps"] * config["num_processes"]
    num_updates = int(config["num_env_steps"]) // num_env_steps_epoch
    elapsed_time = 0

    # Variables used for sampling
    num_pure_expl_steps = config["num_pure_expl_steps"]
    pure_expl_steps_per_env = np.random.randint(0, num_pure_expl_steps+1 , size=config["num_processes"])
    episode_steps = np.zeros(config["num_processes"])
    num_normal_steps = 0

    for j in range(num_updates):
        # Start training
        start = time.time()

        # Set actor-critic to train mode
        actor_critic.train()
        pure_actor_critic.train()

        # Sample episode
        normalise = False
        if j < 10:
            # normalise RND on the data of the first 10 rollouts
            normalise = True
        _, obs, levels, new_normal_steps = sample_episodes(
            envs, 
            rollouts, 
            pure_rollouts, 
            obs, 
            levels, 
            pure_expl_steps_per_env, 
            episode_steps, 
            actor_critic, 
            pure_actor_critic, 
            num_pure_expl_steps, 
            config['num_processes'], 
            rnd, 
            config['rnd_next_state'], 
            normalise
        )
        num_normal_steps += new_normal_steps

        # Compute return
        rollouts.compute_returns(actor_critic, config["gamma"], config["gae_lambda"])
        rollouts.compute_advantages()

        pure_rollouts.compute_returns(pure_actor_critic, config["gamma"], config["gae_lambda"])
        pure_rollouts.compute_advantages()

        # Update actor-critic
        train_statistics = agent.update(rollouts)
        pure_train_statistics = pure_agent.update(pure_rollouts)

        # Reset rollout storage
        rollouts.after_update()
        pure_rollouts.after_update()

        # End training
        end = time.time()
        elapsed_time += end - start

        # Statistics
        if j % config["log_interval"] == 0:
            # Train statistics
            total_num_steps = (j + 1) * config["num_processes"] * config["num_steps"]
            time_per_epoch = elapsed_time / (j + 1)

            print(
                "\nUpdate {}, step {}, time per epoch {:.2f} \n".format(
                    j, total_num_steps, time_per_epoch
                )
            )

            logger.logkv("train/total_num_steps", total_num_steps)
            logger.logkv("train/num_normal_steps", num_normal_steps)
            logger.logkv("train/time_per_epoch", time_per_epoch)
            for key, val in train_statistics.items():
                logger.logkv("train/{}".format(key), val)
            for key, val in pure_train_statistics.items():
                logger.logkv("pure_train/{}".format(key), val)

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

            # Evaluate pure actor-critic on train environments
            rendering = False
            if (j // config["log_interval"]) % 3 == 0:
                # only render every third eval
                rendering = config['render_pure_eval']
            render_filename = os.path.join(config["output_dir"], f"pure_eval_{log_file}_{int(total_num_steps / 100_000)}.gif")
            pure_train_eval_statistics, pure_train_value_statistics = evaluate(
                config, pure_actor_critic, device, test_envs=False, norm_infos=norm_infos, rendering=rendering, render_filename=render_filename,
            )
            pure_train_episode_rewards = pure_train_eval_statistics["episode_rewards"]
            pure_train_episode_steps = pure_train_eval_statistics["episode_steps"]

            logger.logkv("pure_train/mean_episode_reward", np.mean(pure_train_episode_rewards))
            logger.logkv("pure_train/med_episode_reward", np.median(pure_train_episode_rewards))
            logger.logkv("pure_train/std_episode_reward", np.std(pure_train_episode_rewards))
            logger.logkv("pure_train/mean_episode_step", np.mean(pure_train_episode_steps))
            logger.logkv("pure_train/med_episode_step", np.median(pure_train_episode_steps))
            logger.logkv("pure_train/std_episode_step", np.std(pure_train_episode_steps))

            for key, val in pure_train_value_statistics.items():
                logger.logkv("pure_train/{}".format(key), val) 

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


if __name__ == "__main__":
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
        config_file = open("configs/{}.yaml".format('dcpg'), "r")     
    else:
        # config_file = open("configs/{}.yaml".format(args.exp_name), "r")    
        config_file = open(args.config, "r")   
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Update config
    if DEBUG:
        config["exp_name"] = 'dcpg'
        config["env_name"] = 'bigfish'
        config["seed"] = 1
        config["debug"] = False
        config["project_name"] = "debugging"
        config["log_dir"] = config["log_dir"][1:]
        config["output_dir"] = config["output_dir"][1:]
        config["save_dir"] = config["save_dir"][1:]
    else:
        config["exp_name"] = args.exp_name
        config["env_name"] = args.env_name
        config["seed"] = args.seed
        config["debug"] = args.debug

    # Run main
    main(config)
