from typing import List, Tuple, Union

import torch
import numpy as np

from dcpg.envs import VecPyTorchProcgen
from dcpg.models import PPOModel
from dcpg.storages import RolloutStorage
from dcpg.rnd import RandomNetworkDistillationState, RandomNetworkDistillationStateAction


def sample_episodes(
    envs: VecPyTorchProcgen,
    rollouts: RolloutStorage,
    obs: torch.Tensor,
    levels: torch.Tensor,
    pure_expl_steps_per_env: np.ndarray,
    episode_steps: np.ndarray,
    ac: PPOModel,
    num_pure_expl_steps: int,
    num_processes: int,
) -> Tuple[List[float], torch.Tensor, torch.Tensor, int]:
    """
    Sample episodes
    """
    episode_rewards = []
    device = obs.device
    num_normal_steps = 0

    # Sample episodes
    for step in range(rollouts.num_steps):
        # Sample action
        pure_expl_inds = episode_steps < pure_expl_steps_per_env
        normal_inds = episode_steps >= pure_expl_steps_per_env

        with torch.no_grad():
            if sum(normal_inds) > 0:
                normal_action, normal_action_log_prob, normal_value = ac.act(obs[normal_inds])
                if sum(pure_expl_inds) > 0:
                    pure_action = torch.randint(0, envs.action_space.n, size=(obs[pure_expl_inds].shape[0],1), device=device)
                else:
                    # cannot call act with empty observation tensor
                    # so create our own empty results
                    pure_action = torch.zeros((0, *normal_action.shape[1:]), dtype=normal_action.dtype, device=device)
            else:
                pure_action = torch.randint(0, envs.action_space.n, size=(obs[pure_expl_inds].shape[0],1), device=device)
                # cannot call act with empty observation tensor
                # so create our own empty results
                normal_action = torch.zeros((0, *pure_action.shape[1:]), dtype=pure_action.dtype, device=device)
                normal_action_log_prob = torch.zeros((0, 1), dtype=torch.float32, device=device)
                normal_value = torch.zeros((0, 1), dtype=torch.float32, device=device)
        
        action = torch.zeros((num_processes, 1), dtype=normal_action.dtype, device=device)
        action[normal_inds] = normal_action
        action[pure_expl_inds] = pure_action

        # Interact with environment
        next_obs, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
        pure_masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
        # Set mask to 0 for the last pure exploration step of the episode
        pure_masks[episode_steps == pure_expl_steps_per_env-1] = torch.FloatTensor([0.0]).to(device)
        next_levels = torch.LongTensor([0 for info in infos]).to(device)
        reward = reward.to(device)

        # update episodic step count
        episode_steps += 1
        episode_steps[done] = 0
        
        # sample new number of pure exploration steps to perform
        pure_expl_steps_per_env[done] = np.random.randint(0, num_pure_expl_steps+1 , size=sum(done))

        # Insert obs, action and reward into rollout storage
        rollouts.insert(obs[normal_inds], next_obs[normal_inds], normal_action, normal_action_log_prob, reward[normal_inds], normal_value, masks[normal_inds], levels[normal_inds], next_levels[normal_inds], normal_inds)

        # Track episode info
        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

        obs = next_obs
        levels = next_levels
        num_normal_steps += sum(normal_inds)

    return episode_rewards, next_obs, next_levels, num_normal_steps
