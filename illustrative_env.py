import numpy as np
from numpy import ndarray
from typing import Dict, Union

import gym
from gym import spaces

import random

import torch
from torch import device, Tensor

from baselines.common.vec_env import (
    VecEnv,
    VecEnvWrapper,
    VecMonitor,
)
from gymnasium.core import Wrapper

TIMEOUT_STEPS = 20

class IllustrativeCMDP(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, arm_length=1, tasks=[((255,255,0), 'left'), ((0,255,0), 'right'), ((0,0,255), 'top'), ((255,255,255), 'bottom')]):
        self.observation_space = spaces.Box(0,1, shape=(3,(2*arm_length + 1),(2*arm_length + 1)), dtype=float)
        self.size = (2*arm_length + 1)
        self.arm_length = arm_length

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),    # right
            1: np.array([0, 1]),    # down
            2: np.array([-1, 0]),   # left
            3: np.array([0, -1]),   # up
        }

        self.tasks = tasks
        self._current_task_id = 0
        self.num_tasks = len(tasks)
        self._target_location = np.array([self.arm_length,self.arm_length])

    def seed(self, seed):
        super().seed(seed)
        self._shuffle_tasks(seed)

    def _shuffle_tasks(self, seed):
        if seed is not None:
            random.seed(seed)
            random.shuffle(self.tasks)

    def _get_obs(self):
        obs = np.tile(self._current_background_color, (self.size, self.size, 1)).transpose(2,0,1)

        # # Turn arms white (too easy)
        # obs[:, :, self.arm_length] = np.tile(np.array([255,255,255]), (self.size, 1)).transpose(1,0)
        # obs[:, self.arm_length, :] = np.tile(np.array([255,255,255]), (self.size, 1)).transpose(1,0)

        # set agent as a dark red square
        # agent_color = np.array([0,0,0], dtype=np.float64)
        # i = np.random.randint(0,3)
        # agent_color[i] = 1
        obs[:, self._agent_location[0], self._agent_location[1]] = np.array([125,0,0]) / 255.
        # obs[:, self._agent_location[0], self._agent_location[1]] = agent_color

        # set goal as dark green square
        obs[:, self._target_location[0], self._target_location[1]] = np.array([0,125,0]) / 255.

        return obs

    def reset(self):
        self._current_task_id = (self._current_task_id + 1) % self.num_tasks
        self._current_background_color = np.array(self.tasks[self._current_task_id][0]) / 255.
        if self.tasks[self._current_task_id][1] == 'right':
            self._agent_location = np.array([self.size-1, self.arm_length])
        if self.tasks[self._current_task_id][1] == 'left':
            self._agent_location = np.array([0, self.arm_length])
        if self.tasks[self._current_task_id][1] == 'top':
            self._agent_location = np.array([self.arm_length, 0])
        if self.tasks[self._current_task_id][1] == 'bottom':
            self._agent_location = np.array([self.arm_length, self.size-1])

        observation = self._get_obs()

        self.step_counter = 0

        return observation
    
    def _move_agent(self, action):
        if self._agent_location[1] == self.arm_length:
            # we are in the right or left arm
            if action == 3:
                if self._agent_location[0] == 0 or self._agent_location[0] == self.size-1:
                    # move to top arm
                    self._agent_location = np.array([self.arm_length, 0])
            elif action == 1:
                if self._agent_location[0] == 0 or self._agent_location[0] == self.size-1:
                    # move to bottom arm
                    self._agent_location = np.array([self.arm_length, self.size-1])
            else:
                self._agent_location = np.clip(self._agent_location + self._action_to_direction[action], 0, self.size-1)
        elif self._agent_location[0] == self.arm_length:
            # we are in the top or bottom arm
            if action == 0:
                if self._agent_location[1] == 0 or self._agent_location[1] == self.size-1:
                    # move to right arm
                    self._agent_location = np.array([self.size-1, self.arm_length])
            elif action == 2:
                if self._agent_location[1] == 0 or self._agent_location[1] == self.size-1:
                    # move to left arm
                    self._agent_location = np.array([0, self.arm_length])
            else:
                self._agent_location = np.clip(self._agent_location + self._action_to_direction[action], 0, self.size-1)

    def get_agent_location(self):
        return self._agent_location
    
    def step(self, action):
        self._move_agent(action)

        reward = 0
        terminated = False
        if tuple(self._agent_location) == tuple(self._target_location):
            reward = 1
            terminated = True

        obs = self._get_obs()

        self.step_counter += 1

        truncated = False
        if self.step_counter >= TIMEOUT_STEPS:
            truncated = True

        done = terminated or truncated
        info = {"TimeLimit.truncated": True} if truncated else {}
        info["level_seed"] = self._current_task_id

        return obs, reward, done, info

class VecPytorchIllustrative(VecEnvWrapper):
    def __init__(self, venv: VecEnv, device: device):
        super(VecPytorchIllustrative, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs
    
    def step_async(self, actions: Tensor):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
    
    def normalization_infos(self) -> Dict[str, Union[float, ndarray]]:
        var = self.venv.ret_rms.var
        eps = self.venv.epsilon
        cliprew = self.venv.cliprew
        norm_infos = {
            "var": var,
            "eps": eps,
            "cliprew": cliprew,
        }
        return norm_infos
