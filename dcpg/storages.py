from typing import Iterator, Sequence, Tuple

import torch
from torch import device, Tensor
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler
import numpy as np

from gym.spaces import Space

from dcpg.models import PPOModel

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = float(self.count)
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RolloutStorage(object):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: Sequence[int],
        action_space: Space,
    ):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.levels = torch.LongTensor(num_steps + 1, num_processes).fill_(0)

        self.num_steps = num_steps
        self.step = torch.zeros(num_processes, dtype=torch.long)

        self.num_processes = num_processes
        self.obs_shape = obs_shape
        self.action_space = action_space

    def compute_intrinsic_rewards(self, rnd, rnd_next_state, normalise):
        for i in range(self.num_processes):
            num_steps = self.step[i]
            if num_steps == 0:
                # never added any data for this process
                continue

            # Calculate RND
            with torch.no_grad():
                if rnd_next_state:
                    # we calculate RND of the next state, even when done (which would be the starting state in the next episode)
                    intrinsic_rewards = rnd(self.obs[1:num_steps+1, i], update_rms=normalise)
                    intrinsic_rewards *= self.masks[1:num_steps+1, i].squeeze(-1)
                else:
                    intrinsic_rewards = rnd(self.obs[:num_steps, i], self.actions[:num_steps, i], update_rms=normalise)
            self.rewards[:num_steps, i] = intrinsic_rewards.unsqueeze(-1)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def to(self, device: device):
        self.obs = self.obs.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        self.levels = self.levels.to(device)

    def insert(
        self,
        obs: Tensor,
        next_obs: Tensor,
        actions: Tensor,
        action_log_probs: Tensor,
        rewards,
        value_preds: Tensor,
        masks: Tensor,
        levels: Tensor,
        next_levels: Tensor,
        indices: np.ndarray,
    ):
        subset_step = self.step[indices]
        column_vector = torch.arange(self.num_processes)[indices]
        self.obs[subset_step, column_vector] = obs
        self.obs[subset_step + 1, column_vector] = next_obs
        self.actions[subset_step, column_vector] = actions
        self.action_log_probs[subset_step, column_vector] = action_log_probs
        self.rewards[subset_step, column_vector] = rewards
        self.value_preds[subset_step, column_vector] = value_preds
        self.masks[subset_step + 1, column_vector] = masks
        self.levels[subset_step, column_vector] = levels
        self.levels[subset_step + 1, column_vector] = next_levels

        self.step[indices] = (self.step[indices] + 1)

    def after_update(self):
        device = self.obs.device
        self.obs[0].copy_(self.obs[self.step, torch.arange(self.num_processes)])
        self.obs[1:].copy_(torch.zeros(self.num_steps, self.num_processes, *self.obs_shape, device=device))
        self.masks[0].copy_(self.masks[self.step, torch.arange(self.num_processes)])
        self.masks[1:].copy_(torch.ones(self.num_steps, self.num_processes, 1, device=device))
        self.levels[0].copy_(self.levels[self.step, torch.arange(self.num_processes)])
        self.levels[1:].copy_(torch.LongTensor(self.num_steps, self.num_processes).fill_(0).to(device))

        self.step = torch.zeros(self.num_processes, dtype=torch.long)
        if self.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = self.action_space.shape[0]
        self.actions = torch.zeros(self.num_steps, self.num_processes, action_shape, device=device)
        if self.action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
        self.action_log_probs = torch.zeros(self.num_steps, self.num_processes, 1, device=device)
        self.rewards = torch.zeros(self.num_steps, self.num_processes, 1, device=device)
        self.value_preds = torch.zeros(self.num_steps + 1, self.num_processes, 1, device=device)
        self.returns = torch.zeros(self.num_steps + 1, self.num_processes, 1, device=device)

    def compute_returns(self, ac: PPOModel, gamma: float, gae_lambda: float):
        with torch.no_grad():
            next_critic_outputs = ac.forward_critic(self.obs[self.step, torch.arange(self.num_processes)])
            next_value = next_critic_outputs["value"]

        for i in range(self.num_processes):
            num_steps = self.step[i]
            if num_steps == 0:
                # never added any data for this process
                continue

            self.value_preds[num_steps, i] = next_value[i]
            gae = 0
            for step in reversed(range(num_steps)):
                delta = (
                    self.rewards[step, i]
                    + gamma * self.value_preds[step + 1, i] * self.masks[step + 1, i]
                    - self.value_preds[step, i]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1, i] * gae
                self.returns[step, i] = gae + self.value_preds[step, i]

    def compute_advantages(self):
        self.advantages = torch.zeros(self.returns[:-1].shape, device=self.obs.device)
        non_zero_advantages = []
        for i in range(self.num_processes):
            num_steps = self.step[i]
            if num_steps == 0:
                # never added any data for this process
                continue

            self.advantages[:num_steps,i] = self.returns[:num_steps, i] - self.value_preds[:num_steps, i]
            non_zero_advantages.append(self.advantages[:num_steps,i])

        if len(non_zero_advantages) > 0:
            non_zero_advantages = torch.cat(non_zero_advantages)
            mean = non_zero_advantages.mean()
            std = non_zero_advantages.std()
            self.advantages = (self.advantages - mean) / (std + 1e-5)

    def feed_forward_generator(
        self,
        num_mini_batch: int = None,
        mini_batch_size: int = None,
    ) -> Iterator[Tuple[Tensor, ...]]:
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, "[ERROR] num mini batch is too large"
            mini_batch_size = batch_size // num_mini_batch

        indices_to_sample = np.zeros(batch_size, dtype=np.float32)
        for i in range(num_processes):
            step = self.step[i]
            indices_to_sample[np.arange(0, num_processes*step, num_processes) + i] = 1
            
        if self.step.sum().item() == 0:
            # nothing was added to the rollout so just yield nothing
            yield from ()
        else:
            sampler = BatchSampler(
                WeightedRandomSampler(indices_to_sample, self.step.sum().item(), replacement=False),
                mini_batch_size,
                drop_last=True,
            )

            for indices in sampler:
                obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
                next_obs_batch = self.obs[1:].view(-1, *self.obs.size()[2:])[indices]
                actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
                old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
                rewards_batch = self.rewards.view(-1, 1)[indices]
                value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
                returns_batch = self.returns[:-1].view(-1, 1)[indices]
                masks_batch = self.masks[1:].view(-1, 1)[indices]
                levels_batch = self.levels[:-1].view(-1, 1)[indices]

                if hasattr(self, "advantages"):
                    adv_targs = self.advantages.view(-1, 1)[indices]
                else:
                    adv_targs = None

                yield (
                    obs_batch,
                    next_obs_batch,
                    actions_batch,
                    old_action_log_probs_batch,
                    rewards_batch,
                    value_preds_batch,
                    returns_batch,
                    masks_batch,
                    adv_targs,
                    levels_batch,
                )


class E3BRolloutStorage(RolloutStorage):
    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: Sequence[int],
        action_space: Space,
        encoder_hidden_dim: int,
        device,
        e3b_ridge: float = 0.1,
    ):
        super().__init__(
            num_steps=num_steps,
            num_processes=num_processes,
            obs_shape=obs_shape,
            action_space=action_space,
        )

        self.device = device
        self.inverse_cov_init = torch.eye(encoder_hidden_dim) * 1./e3b_ridge
        self.inverse_cov = self.inverse_cov_init.repeat(num_processes, 1, 1).to(device)
        self.intrinsic_reward_rms = RunningMeanStd(shape=())
        self.norm_epsilon = 1e-12

    def compute_intrinsic_rewards(self, rnd, rnd_next_state, normalise, feature_encoder):
        for i in range(self.num_processes):
            num_steps = self.step[i]
            if num_steps == 0:
                # never added any data for this process
                continue

            # Calculate RND
            with torch.no_grad():
                if rnd_next_state:
                    # we calculate RND of the next state, even when done (which would be the starting state in the next episode)
                    global_bonus = rnd(self.obs[1:num_steps+1, i], update_rms=normalise)
                    global_bonus *= self.masks[1:num_steps+1, i].squeeze(-1)
                else:
                    global_bonus = rnd(self.obs[:num_steps, i], self.actions[:num_steps, i], update_rms=normalise)

            # Calculate E3B
            with torch.no_grad():
                encoder_input = self.obs[:num_steps, i]
                phi = feature_encoder(encoder_input)
                # phi = phi.view(B, T, -1)
                done_masks = self.masks[1:num_steps+1, i]
                # dones = buff['dones'].view(B, T)
                # env_id = buff['env_id'].long()
                # # make sure all trajectories are from distinct envs
                # unique_envs = torch.unique(env_id)
                # if torch.numel(env_id) != torch.numel(unique_envs):
                #     print(torch.sort(env_id))
                # #assert torch.numel(env_id) == torch.numel(unique_envs)
                
                # # make sure each trajectory comes after previous ones
                # start_steps = self.start_step[env_id]
                # assert torch.all(buff['start_step'] > start_steps).item()
                    
                batch_inverse_covs = self.inverse_cov[i]
                episodic_bonus = torch.zeros(num_steps).to(self.device)
                    
                for t in range(num_steps):
                    phi_t = phi[t]
                    # u = torch.bmm(phi_t.unsqueeze(1), batch_inverse_covs)
                    u = torch.matmul(batch_inverse_covs, phi_t)
                    # elliptical_bonus = torch.bmm(u, phi_t.unsqueeze(2))
                    elliptical_bonus = torch.matmul(phi_t, u)
                    episodic_bonus[t].copy_(elliptical_bonus)
                    # torch.bmm(u.permute(0, 2, 1), u, out=self.outer_product_buffers)
                    outer_product_u = torch.outer(u, u)
                    outer_product_u = outer_product_u * -1./(1. + elliptical_bonus)
                    # torch.mul(self.outer_product_buffers, -1./(1. + elliptical_bonus), out=self.outer_product_buffers)
                    # torch.add(batch_inverse_covs, self.outer_product_buffers, out=batch_inverse_covs)
                    batch_inverse_covs = batch_inverse_covs + outer_product_u
                    # if any episodes are done, reset the (inverse) covariance matrix
                    # if torch.any(dones[:, t]).item():
                    if (done_masks[t].squeeze(-1) == 0).item():                                                                 
                        batch_inverse_covs.copy_(self.inverse_cov_init)
                                                    
            self.inverse_cov[i] = batch_inverse_covs
            self.rewards[:num_steps, i] = (episodic_bonus * global_bonus).unsqueeze(-1)
            self.intrinsic_reward_rms.update((episodic_bonus * global_bonus).detach().cpu().numpy())

        # normalise the intrinsic rewards
        for i in range(self.num_processes):
            num_steps = self.step[i]
            if num_steps == 0:
                # never added any data for this process
                continue
            self.rewards[:num_steps, i] = self.rewards[:num_steps, i] / torch.sqrt(torch.as_tensor(self.intrinsic_reward_rms.var, device=self.device) + self.norm_epsilon)
