import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from dcpg.rnd import RandomNetworkDistillationState

import numpy as np

class FIFOStateBuffer:
    def __init__(self, buffer_size, device, obs_space, num_levels, start_level, store_unnormalised_obs=True):
        self.full = False
        self.size = buffer_size
        self.store_unnormalised_obs = store_unnormalised_obs
        self.device = device
        self.obs_space = obs_space
        self.num_levels = num_levels
        self.start_level = start_level
        self.states = [None]*buffer_size
        self.levels = np.zeros(buffer_size, dtype=np.int32)
        if store_unnormalised_obs:
            self.obs = torch.zeros(buffer_size, *obs_space.shape, dtype=torch.uint8)
        else:
            self.obs = torch.zeros(buffer_size, *obs_space.shape)


        self.pos = 0

    def add(self, observations, states, levels, num_states=None):
        if self.store_unnormalised_obs:
            observations = (observations*255).to(torch.uint8)

        if num_states is None:
            observations = observations.cpu()
            levels = levels.to(torch.int32).cpu().numpy()
            num_states = observations.shape[0]*observations.shape[1]    # num_steps * num_processes
        else:
            num_states = num_states

        shuffled_indices = np.random.permutation(num_states)

        # TODO: Try to make more general...
        if len(observations.shape) == 5:
            observations = observations.view(-1, *self.obs.size()[1:])[shuffled_indices]
            states = np.concatenate(states).flatten()[shuffled_indices]
            levels = levels.flatten()[shuffled_indices]
        else:
            # print('levels: ', levels.shape)
            # print('levels: ', levels)
            # print('states: ', states.shape)
            observations = observations[shuffled_indices]
            states = states[shuffled_indices]
            levels = levels[shuffled_indices]

        # Add the most recent states in FIFO order
        # this only works for num_states < 2*self.size
        if self.pos + num_states > self.size:
            self.states[self.pos:] = list(states[:(self.size-self.pos)])
            self.states[:(num_states-self.size+self.pos)] = list(states[(self.size-self.pos):])

            # print('obs shape: ', self.obs.shape)
            # print('observation shape: ', observations.shape)
            # print('pos: ', self.pos)
            # print('size: ', self.size)

            self.obs[self.pos:] = observations[:(self.size-self.pos)]
            self.obs[:(num_states-self.size+self.pos)] = observations[(self.size-self.pos):]

            self.levels[self.pos:] = levels[:(self.size-self.pos)]
            self.levels[:(num_states-self.size+self.pos)] = levels[(self.size-self.pos):]
        else:
            self.states[self.pos:self.pos+num_states] = list(states)
            self.obs[self.pos:self.pos+num_states] = observations
            self.levels[self.pos:self.pos+num_states] = levels

        if self.pos + num_states >= self.size:
            self.full = True
        self.pos = (self.pos + num_states) % self.size

    def sample(self, batch_size):
        sample_max = self.size if self.full else self.pos
        batch_inds = np.random.randint(0, sample_max, size=batch_size)
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

    def compute_logging_metrics(self):
        num_states_in_buffer = self.size if self.full else self.pos
        fraction_of_unique_states = len(set(self.states[:num_states_in_buffer])) / num_states_in_buffer
        obs_hashes = [hash(o.numpy().data.tobytes()) for o in self.obs[:num_states_in_buffer]]
        fraction_of_unique_obs = len(set(obs_hashes)) / num_states_in_buffer
        level_distribution = np.bincount(self.levels[:num_states_in_buffer] - self.start_level, minlength=self.num_levels)
        fraction_of_level_coverage = np.count_nonzero(level_distribution) / self.num_levels


        return fraction_of_unique_states, fraction_of_unique_obs, fraction_of_level_coverage, (self.levels[:num_states_in_buffer] - self.start_level)


class RNDStateBuffer(FIFOStateBuffer):
    def __init__(self, buffer_size, device, obs_space, action_space, num_levels, start_level, rnd_config, store_unnormalised_obs=True, num_mini_batch=8, epochs=1, add_batch_size=514):
        super().__init__(buffer_size, device, obs_space, num_levels, start_level, store_unnormalised_obs=store_unnormalised_obs)

        self.num_mini_batches = num_mini_batch
        self.add_batch_size = add_batch_size
        self.epochs = epochs
        self.rnd_values = np.zeros(buffer_size)

        self.rnd = RandomNetworkDistillationState(
            action_space,
            obs_space,
            rnd_config['rnd_embed_dim'],
            rnd_config['rnd_kwargs'],
            device=device,
            flatten_input=rnd_config['rnd_flatten_input'],
            use_resnet=rnd_config['rnd_use_resnet'],
            normalize_images=rnd_config['rnd_normalize_images'],
            normalize_output=rnd_config['rnd_normalize_output'],
            )

    def add(self, observations, states, levels, num_states=None):
        if not self.full:
            super().add(observations, states, levels, num_states)
        else:
            if num_states == None:
                observations = observations.view(-1, *self.obs.size()[1:])
                states = np.concatenate(states).flatten()
                levels = levels.to(torch.int32).flatten().cpu().numpy()

            num_states = observations.shape[0]    # num_steps * num_processes

            # Add the new states in minibatches
            sampler = BatchSampler(
                SubsetRandomSampler(range(num_states)), num_states, drop_last=False
            )
            for indices in sampler:
                obs_batch = observations[indices]
                states_batch = states[indices]
                levels_batch = levels[indices]

                with torch.no_grad():
                    rnd_values_batch = self.rnd(obs_batch).cpu().numpy()

                # replace the states with lowest RND
                lowest_rnd_indices = self.rnd_values.argsort()[:num_states]

                # Add observations
                if self.store_unnormalised_obs:
                    obs_batch = (obs_batch*255).to(torch.uint8).cpu()
                self.obs[lowest_rnd_indices] = obs_batch

                # Add levels
                self.levels[lowest_rnd_indices] = levels_batch

                # Add states
                for c,v in enumerate(lowest_rnd_indices):
                    self.states[v] = states_batch[c]

                # Update RND values
                self.rnd_values[lowest_rnd_indices] = rnd_values_batch


        # Train RND on the buffer
        sample_max = self.size if self.full else self.pos
        for _ in range(max(self.epochs, 1)):
            # if epochs > 1 assume its integer and perform several epochs
            sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.size // self.num_mini_batches, drop_last=True
            )

            max_num_updates = self.num_mini_batches * self.epochs
            for count, indices in enumerate(sampler):
                if count == max_num_updates:
                    # don't perform anymore RND updates
                    # this can only trigger if self.epochs < 1
                    break

                obs_batch = self.obs[indices].to(self.device)
                if self.store_unnormalised_obs:
                    obs_batch = obs_batch.to(torch.float32) / 255.
                self.rnd.observe(obs_batch)



        # update the RND values of the states in the buffer
        sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.size // self.num_mini_batches, drop_last=False
            )
        for indices in sampler:
            obs_batch = self.obs[indices].to(self.device)
            if self.store_unnormalised_obs:
                obs_batch = obs_batch.to(torch.float32) / 255.
            with torch.no_grad():
                self.rnd_values[indices] = self.rnd(obs_batch).cpu().numpy()



class RNDFIFOStateBuffer(FIFOStateBuffer):
    def __init__(self, buffer_size, device, obs_space, action_space, num_levels, start_level, rnd_config, store_unnormalised_obs=True, mini_batch_size=2056, rnd_epoch=1, tournament_size=500):
        super().__init__(buffer_size, device, obs_space, num_levels, start_level, store_unnormalised_obs=store_unnormalised_obs)

        self.action_space = action_space
        self.rnd_mini_batch_size = mini_batch_size
        self.rnd_epoch = rnd_epoch
        self.tournament_size = tournament_size
        self.rnd_values = np.zeros(buffer_size)

        self.rnd = RandomNetworkDistillationState(
            action_space,
            obs_space,
            rnd_config['rnd_embed_dim'],
            rnd_config['rnd_kwargs'],
            device=device,
            flatten_input=rnd_config['rnd_flatten_input'],
            use_resnet=rnd_config['rnd_use_resnet'],
            normalize_images=rnd_config['rnd_normalize_images'],
            normalize_output=rnd_config['rnd_normalize_output'],
            )

        self.rnd_config = rnd_config

    def add(self, observations, states, levels, num_states=None):
        super().add(observations, states, levels, num_states)

        # The buffer has changed so much we need to reinitialise the RND networks
        # self.rnd = RandomNetworkDistillationState(
        #     self.action_space,
        #     self.obs_space,
        #     self.rnd_config['rnd_embed_dim'],
        #     self.rnd_config['rnd_kwargs'],
        #     device=self.device,
        #     flatten_input=self.rnd_config['rnd_flatten_input'],
        #     use_resnet=self.rnd_config['rnd_use_resnet'],
        #     normalize_images=self.rnd_config['rnd_normalize_images'],
        #     normalize_output=self.rnd_config['rnd_normalize_output'],
        #     )

        # Train RND on the buffer
        sample_max = self.size if self.full else self.pos
        for _ in range(max(self.rnd_epoch, 1)):
            # if rnd_epoch > 1 assume its integer and perform several epochs
            sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.rnd_mini_batch_size, drop_last=True
            )

            max_num_updates = (self.size / self.rnd_mini_batch_size) * self.rnd_epoch
            for count, indices in enumerate(sampler):
                if count >= max_num_updates:
                    # don't perform anymore RND updates
                    # this can only trigger if self.rnd_epoch < 1
                    break

                obs_batch = self.obs[indices].to(self.device)
                if self.store_unnormalised_obs:
                    obs_batch = obs_batch.to(torch.float32) / 255.
                self.rnd.observe(obs_batch)



        # update the RND values of the states in the buffer
        sampler = BatchSampler(
                SubsetRandomSampler(range(sample_max)), self.rnd_mini_batch_size, drop_last=False
            )
        max_num_updates = (self.size / self.rnd_mini_batch_size) * self.rnd_epoch
        for count, indices in enumerate(sampler):
            if count >= max_num_updates:
                # don't perform anymore RND updates
                # this can only trigger if self.rnd_epoch < 1
                break
            obs_batch = self.obs[indices].to(self.device)
            if self.store_unnormalised_obs:
                obs_batch = obs_batch.to(torch.float32) / 255.
            with torch.no_grad():
                self.rnd_values[indices] = self.rnd(obs_batch).cpu().numpy()



    def sample(self, batch_size):
        # sample_max = self.size if self.full else self.pos

        # # sample uniformly from the 5*batch_size states with highest RND
        # highest_rnd_indices = self.rnd_values[:sample_max].argsort()[-5*batch_size:]
        # batch_inds = np.random.choice(highest_rnd_indices, size=batch_size, replace=False)

        # sample uniformly self.tournament_size states
        sample_max = self.size if self.full else self.pos
        tournament_inds = np.random.randint(0, sample_max, size=self.tournament_size)

        # take the batch_size states with highest novelty in the tournament
        tournament_values = self.rnd_values[tournament_inds]
        highest_rnd_indices = tournament_values.argsort()[-batch_size:]


        return super()._get_samples(tournament_inds[highest_rnd_indices])

class LevelsRNDStateBuffer(FIFOStateBuffer):
    # Keep one buffer for each level
    # Sample uniformly from levels, then
    def __init__(self, buffer_size, device, obs_space, action_space, num_levels, start_level, rnd_config, store_unnormalised_obs=True, num_mini_batch=8, epochs=1, add_batch_size=514):
        super().__init__(buffer_size, device, obs_space, num_levels, start_level, store_unnormalised_obs=store_unnormalised_obs)

        # adjust buffer size per level
        buffer_size_level = buffer_size//num_levels
        buffer_size_level = buffer_size
        self.level_rnds = [RNDStateBuffer(buffer_size_level, device, obs_space, action_space,
                            num_levels, start_level, rnd_config, store_unnormalised_obs,
                            num_mini_batch, epochs, add_batch_size) for _ in range(num_levels)]
        self.used_levels = set([])

    def add(self, observations, states, levels):
        # print(f'observations shape before: {observations.shape}')
        # print(f'levels before: {levels}')
        # print(f'levels shape: {levels.shape}')
        observations = observations.view(-1, *self.obs.size()[1:])
        # print(f'observations shape after: {observations.shape}')
        states = np.concatenate(states).flatten()
        levels = levels.to(torch.int32).flatten().cpu().numpy()
        # print(f'levels after: {levels}')
        # print(f'levels after shape: {levels.shape}')

        cur_size = 0
        for level in range(max(levels)):
            indices = np.where(levels == level)
            obs_level = observations[indices]
            states_level = states[indices]
            levels_level = levels[indices]
            if len(obs_level) > 0:
                self.used_levels.add(level)
                self.level_rnds[level].add(obs_level,
                                            states_level,
                                            torch.LongTensor(levels_level),
                                            num_states=obs_level.shape[0])
            #update the current total buffer size
            cur_size += self.level_rnds[level].size if self.level_rnds[level].full else self.level_rnds[level].pos
        self.cur_size = cur_size

    def sample(self, batch_size):
        sample_max = self.size if self.full else self.cur_size
        # uniformly sample from the levels
        levels = np.random.choice(list(self.used_levels), size=batch_size)
        # print('Levels selected: ', levels)
        obs_batch = []
        states_batch = []
        for level in levels:
            # print(f'level: {level}, pos: {self.level_rnds[level].pos}')
            obs, state = self.level_rnds[level].sample(1)
            obs_batch.append(obs[0])
            states_batch.append(state[0])

        obs_batch = torch.stack(obs_batch, dim=0).to(self.device)

        if self.store_unnormalised_obs:
            obs_batch = obs_batch.to(torch.float32) / 255.

        return obs_batch, states_batch

    def compute_logging_metrics(self):
        total_unique_states = 0.
        total_unique_obs = 0.
        level_dist = []
        total_size = 0
        for level in self.used_levels:
            buffer = self.level_rnds[level]
            num_states_in_buffer = buffer.size if buffer.full else buffer.pos
            total_size += num_states_in_buffer
            unique_states = len(set(buffer.states[:num_states_in_buffer]))
            obs_hashes = [hash(o.numpy().data.tobytes()) for o in buffer.obs[:num_states_in_buffer]]
            unique_obs = len(set(obs_hashes))


            total_unique_states += unique_states
            total_unique_obs += unique_obs
            level_dist += [level-self.start_level]*num_states_in_buffer
        return total_unique_states/total_size, total_unique_obs/total_size, len(self.used_levels), level_dist
