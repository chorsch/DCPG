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
    pure_rollouts: RolloutStorage,
    full_rollouts: RolloutStorage,
    obs: torch.Tensor,
    levels: torch.Tensor,
    pure_expl_steps_per_env: np.ndarray,
    episode_steps: np.ndarray,
    ac: PPOModel,
    pure_expl_ac: PPOModel,
    num_pure_expl_steps: int,
    num_processes: int,
    rnd: Union[RandomNetworkDistillationState, RandomNetworkDistillationStateAction],
    rnd_next_state: bool,
    normalise: bool,
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
                    pure_action, pure_action_log_prob, pure_value = pure_expl_ac.act(obs[pure_expl_inds])
                else:
                    # cannot call act with empty observation tensor
                    # so create our own empty results
                    pure_action = torch.zeros((0, *normal_action.shape[1:]), dtype=normal_action.dtype, device=normal_action.device)
                    pure_action_log_prob = torch.zeros((0, *normal_action_log_prob.shape[1:]), dtype=normal_action_log_prob.dtype, device=normal_action_log_prob.device)
                    pure_value = torch.zeros((0, *normal_value.shape[1:]), dtype=normal_value.dtype, device=normal_value.device)
            else:
                pure_action, pure_action_log_prob, pure_value = pure_expl_ac.act(obs[pure_expl_inds])
                # cannot call act with empty observation tensor
                # so create our own empty results
                normal_action = torch.zeros((0, *pure_action.shape[1:]), dtype=pure_action.dtype, device=pure_action.device)
                normal_action_log_prob = torch.zeros((0, *pure_action_log_prob.shape[1:]), dtype=pure_action_log_prob.dtype, device=pure_action_log_prob.device)
                normal_value = torch.zeros((0, *pure_value.shape[1:]), dtype=pure_value.dtype, device=pure_value.device)
        
        action = torch.zeros((num_processes, 1), dtype=normal_action.dtype, device=device)
        action[normal_inds] = normal_action
        action[pure_expl_inds] = pure_action

        # Interact with environment
        next_obs, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)
        next_levels = torch.LongTensor([info["level_seed"] for info in infos]).to(device)
        reward = reward.to(device)

        # update episodic step count
        episode_steps += 1
        episode_steps[done] = 0
        
        # sample new number of pure exploration steps to perform
        pure_expl_steps_per_env[done] = np.random.randint(0, num_pure_expl_steps+1 , size=sum(done))

        # Train the RND loss
        if rnd_next_state:
            rnd.observe(obs[pure_expl_inds])
        else:
            rnd.observe(obs[pure_expl_inds], pure_action)

        # Insert obs, action and reward into rollout storage
        rollouts.insert(obs[normal_inds], next_obs[normal_inds], normal_action, normal_action_log_prob, reward[normal_inds], normal_value, masks[normal_inds], levels[normal_inds], next_levels[normal_inds], normal_inds)
        pure_rollouts.insert(obs[pure_expl_inds], next_obs[pure_expl_inds], pure_action, pure_action_log_prob, reward[pure_expl_inds], pure_value, masks[pure_expl_inds], levels[pure_expl_inds], next_levels[pure_expl_inds], pure_expl_inds)

        value = torch.zeros((num_processes, 1), dtype=normal_value.dtype, device=device)
        value[normal_inds] = normal_value
        value[pure_expl_inds] = pure_value
        action_log_prob = torch.zeros((num_processes, 1), dtype=normal_action_log_prob.dtype, device=device)
        action_log_prob[normal_inds] = normal_action_log_prob
        action_log_prob[pure_expl_inds] = pure_action_log_prob
        full_rollouts.insert(obs, next_obs, action, action_log_prob, reward, value, masks, levels, next_levels, np.ones(normal_inds.shape, dtype=np.bool_))

        # Track episode info
        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

        obs = next_obs
        levels = next_levels
        num_normal_steps += sum(normal_inds)

    
    # Update reward to include the intrinsic reward
    pure_rollouts.update_rewards(rnd, rnd_next_state, normalise)

    return episode_rewards, next_obs, next_levels, num_normal_steps
