"""Opt lab 2 training entry point: the stock ``train`` CLI with our reward weights.

Every Pupper task ships with all reward weights at zero. This applies
``REWARD_WEIGHTS`` to the task's defaults before tyro parses the command line,
so every ``train`` flag still works and still overrides:

  python scripts/opt_lab2/train.py Mjlab-MixedGaits-Flat-Pupper-v3 \\
      --agent.run-name baseline-trot

  # Warm-start a bootstrapping round from a previous run:
  python scripts/opt_lab2/train.py Mjlab-MixedGaits-Flat-Pupper-v3 \\
      --agent.resume True --wandb-run-path <entity>/mjlab/<run-id>

The weights start from our lab 4 StableGait-Bumpy notebook. Terms a task doesn't
have are skipped; terms not listed here keep the task default (zero).
"""

import sys

import tyro

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import list_tasks

REWARD_WEIGHTS: dict[str, float] = {
  # Objectives.
  "track_linear_velocity": 10.0,
  "track_yaw_velocity": 5.0,
  "upright": 4.0,
  "air_time": 0.01,
  "base_height": 0.0,
  "pose": 3.0,  # VelocityFS only.
  "gait_tracking": 0.5,  # Must be nonzero: it's all the airborne phase learns.
  "track_angular_velocity": 0.0,
  # Penalties.
  "termination": -10.0,
  "orientation_l2": 0.0,
  "lin_vel_z_l2": -0.01,
  "ang_vel_xy_l2": -0.01,
  "foot_slip": -0.01,
  "action_rate_l2": -0.002,
  "joint_torques_l2": 0.0,
  "joint_acc_l2": -1e-7,
  "stand_still_pose": -0.01,
  "stand_still_joint_velocity": -0.001,
  "knee_ground_contact": -2.0,
  "abduction_angle": 0.0,
  "self_collision_l": -1.0,
  "self_collision_r": -1.0,
  "roll_asymmetry": 0.0,  # MixedGaits/Mystery only; untuned.
}
NUM_ENVS = 4096
MAX_ITERATIONS = 4000


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  default = TrainConfig.from_task(task)
  for name, weight in REWARD_WEIGHTS.items():
    if name in default.env.rewards:
      default.env.rewards[name].weight = weight
  default.env.scene.num_envs = NUM_ENVS
  default.agent.max_iterations = MAX_ITERATIONS

  args = tyro.cli(
    TrainConfig,
    args=remaining,
    default=default,
    prog=f"{sys.argv[0]} {task}",
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(task_id=task, args=args)


if __name__ == "__main__":
  main()
