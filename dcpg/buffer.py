import torch
from torch.distributions import kl_divergence
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from dcpg.rnd import RandomNetworkDistillationState

import numpy as np

class FIFOStateBuffer:
    def __init__(self, buffer_size, device, obs_space, store_unnormalised_obs=True):
        self.num_states_added = 0
        self.full = False
        self.size = buffer_size
        self.store_unnormalised_obs = store_unnormalised_obs
        self.device = device
        self.obs_space = obs_space
        self.states = [None]*buffer_size
        if store_unnormalised_obs:
            self.obs = torch.zeros(buffer_size, *obs_space.shape, dtype=torch.uint8)
        else:
            self.obs = torch.zeros(buffer_size, *obs_space.shape)

    def add(self, observations, states):
        if self.store_unnormalised_obs:
            observations = (observations*255).to(torch.uint8)
        observations = observations.cpu()
        
        num_states = observations.shape[0]*observations.shape[1]    # num_steps * num_processes

        # Add the most recent states in FIFO order
        self.states = self.states[num_states:] + list(np.concatenate(states).flatten()[-self.size:])
        self.obs[:max(0, self.size-num_states)] = self.obs[num_states:]
        self.obs[-num_states:] = observations.view(-1, *self.obs.size()[1:])[-self.size:]

        if not self.full:
            self.num_states_added += num_states
        if self.num_states_added >= self.size:
            self.full = True

    def sample(self, batch_size):
        sample_max = self.size if self.full else self.num_states_added
        batch_inds = np.random.randint(self.size-sample_max, self.size, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds):
        obs_batch = []
        states_batch = []
        for i in batch_inds:
            obs_batch.append(self.obs[i])
            states_batch.append(self.states[i])

        obs_batch = torch.stack(obs_batch, dim=0).to(self.device)

        if self.store_unnormalised_obs:
            obs_batch = obs_batch.to(torch.float32) / 255.

        return obs_batch, states_batch
    

class RNDFIFOStateBuffer(FIFOStateBuffer):
    def __init__(self, config, buffer_size, device, obs_space, action_space, store_unnormalised_obs=True, num_mini_batch=8, rnd_epoch=1):
        super().__init__(buffer_size, device, obs_space, store_unnormalised_obs=store_unnormalised_obs)

        self.action_space = action_space
        self.rnd_num_mini_batch = num_mini_batch
        self.rnd_epoch = rnd_epoch
        self.rnd_values = np.zeros(buffer_size)

        self.rnd = RandomNetworkDistillationState(
            action_space,
            obs_space,
            config['rnd_embed_dim'],
            config['rnd_kwargs'],
            device=device,
            flatten_input=config['rnd_flatten_input'],
            use_resnet=config['rnd_use_resnet'],
            normalize_images=config['rnd_normalize_images'],
            normalize_output=config['rnd_normalize_output'],
            )

        self.config = config

    def add(self, observations, states):
        super().add(observations, states)

        # Train RND on the buffer
        # The buffer has changed so much we need to reinitialise the RND networks
        self.rnd = RandomNetworkDistillationState(
            self.action_space,
            self.obs_space,
            self.config['rnd_embed_dim'],
            self.config['rnd_kwargs'],
            device=self.device,
            flatten_input=self.config['rnd_flatten_input'],
            use_resnet=self.config['rnd_use_resnet'],
            normalize_images=self.config['rnd_normalize_images'],
            normalize_output=self.config['rnd_normalize_output'],
            )

        sample_max = self.size if self.full else self.num_states_added
        for _ in range(max(self.rnd_epoch, 1)):
            # if rnd_epoch > 1 assume its integer and perform several epochs
            sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.size // self.rnd_num_mini_batch, drop_last=True
            )

            max_num_updates = self.rnd_num_mini_batch * self.rnd_epoch
            for count, indices in enumerate(sampler):
                if count == max_num_updates:
                    # don't perform anymore RND updates
                    # this can only trigger if self.rnd_epoch < 1
                    break

                obs_batch = self.obs[indices].to(self.device)
                if self.store_unnormalised_obs:
                    obs_batch = obs_batch.to(torch.float32) / 255.
                self.rnd.observe(obs_batch)



        # update the RND values of the states in the buffer
        sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.size // self.rnd_num_mini_batch, drop_last=False
            )
        for indices in sampler:
            obs_batch = self.obs[indices].to(self.device)
            if self.store_unnormalised_obs:
                obs_batch = obs_batch.to(torch.float32) / 255.
            with torch.no_grad():
                self.rnd_values[indices] = self.rnd(obs_batch).cpu().numpy()



    def sample(self, batch_size):
        sample_max = self.size if self.full else self.num_states_added

        # sample uniformly from the 5*batch_size states with highest RND
        highest_rnd_indices = self.rnd_values[:sample_max].argsort()[-5*batch_size:]
        batch_inds = np.random.choice(highest_rnd_indices, size=batch_size, replace=False)


        return super()._get_samples(batch_inds)


    # def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
    #     num_processes = self.segs[0]["obs"].size(1)
    #     num_segs = len(self.segs)
    #     batch_size = num_processes * num_segs

    #     if mini_batch_size is None:
    #         mini_batch_size = num_processes // num_mini_batch

    #     sampler = BatchSampler(
    #         SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
    #     )

    #     for indices in sampler:
    #         obs = []
    #         returns = []

    #         for idx in indices:
    #             process_idx = idx // num_segs
    #             seg_idx = idx % num_segs

    #             seg = self.segs[seg_idx]
    #             obs.append(seg["obs"][:, process_idx])
    #             returns.append(seg["returns"][:, process_idx])

    #         obs = torch.stack(obs, dim=1).to(self.device)
    #         returns = torch.stack(returns, dim=1).to(self.device)

    #         obs_batch = obs[:-1].view(-1, *obs.size()[2:])
    #         returns_batch = returns[:-1].view(-1, 1)

    #         if self.store_unnormalised_obs:
    #             obs_batch = obs_batch.to(torch.float32) / 255.

    #         yield obs_batch, returns_batch