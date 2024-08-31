# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RoboBase vectorized environment.

The following example shows how to wrap an environment for RoboBase:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.robobase import RobobaseVecEnvWrapper

    env = RobobaseVecEnvWrapper(env)

"""

# needed to import for allowing type-hinting: torch.Tensor | dict[str, torch.Tensor]
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn  # noqa: F401
from typing import Any, Tuple

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv


class RobobaseVecEnvWrapper(gym.ActionWrapper):
    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv) -> None:
        """Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RoboBase vectorized environment.

        Args:
            env: The environment to wrap.

        """
        super().__init__(env)

        self._obs_dict = None
        self.reset_once = True
        self.device = env.device

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        return (
            self._obs_dict,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if self.reset_once:
            self._obs_dict, info = self.env.reset()
            self.reset_once = False
        return self._obs_dict, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        pass

    def close(self) -> None:
        """Close the environment"""
        self.env.close()
