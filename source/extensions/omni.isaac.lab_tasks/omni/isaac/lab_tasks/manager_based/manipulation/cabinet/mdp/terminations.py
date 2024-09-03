# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

"""
MDP terminations.
"""


def object_away_from_goal(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object"), threshold: float = 0.3
) -> torch.Tensor:
    """Terminate when the object is away from the goal."""
    drawer_pos: RigidObject = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    return drawer_pos > threshold
