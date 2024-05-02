from typing import List

import torch
import numpy as np

from dcpg.envs import VecPyTorchProcgen
from dcpg.models import PPOModel
from dcpg.storages import RolloutStorage
from dcpg.buffer import StateBuffer


def sample_episodes(
    envs: VecPyTorchProcgen,
    rollouts: RolloutStorage,
    state_buffer: StateBuffer,
    actor_critic: PPOModel,
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
        states = envs.get_states()

        if np.any(done):
            if state_buffer.full:
                # Teleport to state from the state buffer
                num_of_dones = sum(done)
                teleport_obs, teleport_states = state_buffer.sample(num_of_dones)

                for c, v in enumerate(np.where(done)[0]):
                    states[v] = teleport_states[c]
                    obs[v] = teleport_obs[c]

                envs.set_states(teleport_states, np.where(done)[0])

        # Insert obs, action and reward into rollout storage
        rollouts.insert(obs, action, action_log_prob, reward, value, masks, levels, states)

        # Track episode info
        for info in infos:
            if "episode" in info.keys():
                episode_rewards.append(info["episode"]["r"])

    return episode_rewards
