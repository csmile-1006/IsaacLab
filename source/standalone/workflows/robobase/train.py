# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with Robobase.

Visit the robobase documentation (https://github.com/robobase-org/robobase) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime

from omegaconf import DictConfig
from robobase.isaaclab_workspace import IsaacLabWorkspace

from omni.isaac.lab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.robobase import RobobaseVecEnvWrapper


@hydra_task_config(args_cli.task, "robobase_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: dict):
    """Train with robobase agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = DictConfig(agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.num_train_envs = env_cfg.scene.num_envs
    # max iterations for training
    if args_cli.max_iterations is not None:
        agent_cfg.num_train_frames = (
            args_cli.max_iterations * agent_cfg.action_repeat * env_cfg.scene.num_envs * agent_cfg.action_sequence
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "robobase", agent_cfg.experiment.directory)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.experiment.experiment_name:
        log_dir += f"_{agent_cfg.experiment.experiment_name}"
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # override configurations with non-hydra CLI arguments & environment cfgs
    agent_cfg.tb.log_dir = log_dir
    agent_cfg.num_explore_steps = agent_cfg.num_explore_steps // env_cfg.scene.num_envs

    # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), dict(agent_cfg))
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), dict(agent_cfg))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for robobase
    # env = RobobaseVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaaclab")`

    # load workspace of Robobase
    workspace = IsaacLabWorkspace(cfg=agent_cfg, env=env, env_factory=None, work_dir=log_dir)
    workspace.train_envs = RobobaseVecEnvWrapper(workspace.train_envs)

    # train the agent
    workspace.train()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
