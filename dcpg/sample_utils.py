from typing import List

import torch

from dcpg.envs import VecPyTorchProcgen
from dcpg.models import PPOModel
from dcpg.storages import RolloutStorage
from dcpg.rnd import RandomNetworkDistillation


def sample_episodes(
    envs: VecPyTorchProcgen,
    rollouts: RolloutStorage,
    actor_critic: PPOModel,
    rnd: RandomNetworkDistillation,
    use_rnd: bool,
    beta: float,
    normalise: bool,
) -> List[float]:
    """
    Sample episodes
    """
    episode_rewards = []

    # Sample episodes
    for step in range(rollouts.num_steps):
        # Sample action
        with torch.no_grad():
            action, action_log_prob, value = actor_critic.act(rollouts.obs[step])

        # Interact with environment
        obs, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        levels = torch.LongTensor([info["level_seed"] for info in infos])

        if use_rnd:
            # Train the RND loss
            rnd.observe(rollouts.obs[step], action)

        # Insert obs, action and reward into rollout storage
        rollouts.insert(obs, action, action_log_prob, reward, value, masks, levels)

        # Track episode info
        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

    if use_rnd:
        # Update reward to include the intrinsic reward
        rollouts.update_rewards(rnd, beta, normalise)

    return episode_rewards
