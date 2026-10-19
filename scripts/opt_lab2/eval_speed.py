"""Fixed-command evaluation: tracking, effort, and slip conditioned on the command.

Training metrics average over every command the curriculum samples. Comparing
fast references needs them per command, so this rolls a checkpoint out with each
env pinned to one command and reports steady-state numbers per command:

  python scripts/opt_lab2/eval_speed.py Mjlab-MixedGaits-Flat-Pupper-v3 \\
      --wandb-run-path <entity>/mjlab/<run-id> [--checkpoint model_4000.pt]

Each env holds its command for the whole rollout (play mode: no timeouts, no
pushes; reset-time DR still applies). Samples within ``--warmup-s`` of an env's
start or of a fall are excluded. The heading hold stays on, as on the robot.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.utils.os import get_wandb_checkpoint_path

# (vx, vy, yaw): the fast band both ways, the trot band, and the lift modes.
DEFAULT_COMMANDS: tuple[tuple[float, float, float], ...] = (
  (-1.5, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
  (-0.5, 0.0, 0.0),
  (0.3, 0.0, 0.0),
  (0.5, 0.0, 0.0),
  (1.0, 0.0, 0.0),
  (1.25, 0.0, 0.0),
  (1.5, 0.0, 0.0),
  (0.0, 0.0, 1.0),
  (0.0, 0.3, 0.0),
)
FOOT_SITES = (
  "leg_front_r_3_foot_site",
  "leg_front_l_3_foot_site",
  "leg_back_r_3_foot_site",
  "leg_back_l_3_foot_site",
)
METRICS = ("vx", "err_xy", "err_yaw", "torque_sq", "slip")


def _parse_commands(text: str | None) -> tuple[tuple[float, float, float], ...]:
  if text is None:
    return DEFAULT_COMMANDS
  cmds = []
  for item in text.split(";"):
    vx, vy, yaw = (float(v) for v in item.split(","))
    cmds.append((vx, vy, yaw))
  return tuple(cmds)


def main() -> None:
  parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  parser.add_argument("task")
  parser.add_argument("--wandb-run-path", required=True)
  parser.add_argument("--checkpoint", default=None, help="e.g. model_4000.pt")
  parser.add_argument(
    "--commands", default=None, help='"vx,vy,yaw;vx,vy,yaw;..." (default: a sweep)'
  )
  parser.add_argument("--envs-per-command", type=int, default=64)
  parser.add_argument("--duration-s", type=float, default=12.0)
  parser.add_argument("--warmup-s", type=float, default=2.0)
  parser.add_argument("--out", default=None, help="write results as JSON here")
  args = parser.parse_args()

  commands = _parse_commands(args.commands)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(args.task, play=True)
  twist = env_cfg.commands["twist"]
  # Pin commands: no resampling, no standing/forward/turn special envs. The loop
  # below rewrites the raw command every step, so even a fall's reset-time
  # resample is overwritten before the next policy step.
  twist.resampling_time_range = (1e9, 1e9)
  for attr in ("rel_standing_envs", "rel_forward_envs", "rel_turn_envs"):
    setattr(twist, attr, 0.0)
  env_cfg.scene.num_envs = len(commands) * args.envs_per_command
  agent_cfg = load_rl_cfg(args.task)

  ckpt, _ = get_wandb_checkpoint_path(
    Path("logs/rsl_rl") / agent_cfg.experiment_name,
    Path(args.wandb_run_path),
    args.checkpoint,
  )
  print(f"[eval] checkpoint: {ckpt}")

  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=device),
    clip_actions=agent_cfg.clip_actions,
  )
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  base = env.unwrapped
  robot = base.scene["robot"]
  contact = base.scene["feet_ground_contact"]
  site_ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  cmd_term = base.command_manager.get_term("twist")
  assert isinstance(cmd_term, UniformVelocityCommand)
  target = torch.tensor(commands, device=device).repeat_interleave(
    args.envs_per_command, dim=0
  )
  group = torch.arange(len(commands), device=device).repeat_interleave(
    args.envs_per_command
  )

  dt = base.step_dt
  warmup_steps = int(args.warmup_s / dt)
  n_steps = int(args.duration_s / dt)
  since_reset = torch.zeros(base.num_envs, device=device, dtype=torch.long)
  sums = {m: torch.zeros(len(commands), device=device) for m in METRICS}
  counts = torch.zeros(len(commands), device=device)
  slip_counts = torch.zeros(len(commands), device=device)
  falls = torch.zeros(len(commands), device=device)

  cmd_term.vel_command_b[:] = target
  obs = env.get_observations()
  with torch.no_grad():
    for _ in range(n_steps):
      obs, _, dones, _ = env.step(policy(obs))
      since_reset = torch.where(dones.bool(), 0, since_reset + 1)
      falls.index_add_(0, group, dones.float())
      cmd_term.vel_command_b[:] = target

      live = (since_reset > warmup_steps).float()
      v_b = robot.data.root_link_lin_vel_b
      w_z = robot.data.root_link_ang_vel_b[:, 2]
      in_contact = (contact.data.found > 0).float()
      foot_speed = torch.norm(robot.data.site_lin_vel_w[:, site_ids, :2], dim=-1)
      per_env = {
        "vx": v_b[:, 0],
        "err_xy": torch.norm(target[:, :2] - v_b[:, :2], dim=-1),
        "err_yaw": torch.abs(target[:, 2] - w_z),
        "torque_sq": torch.sum(torch.square(robot.data.actuator_force), dim=1),
      }
      for m, value in per_env.items():
        sums[m].index_add_(0, group, value * live)
      counts.index_add_(0, group, live)
      sums["slip"].index_add_(
        0, group, torch.sum(foot_speed * in_contact, dim=1) * live
      )
      slip_counts.index_add_(0, group, torch.sum(in_contact, dim=1) * live)

  env_minutes = args.envs_per_command * n_steps * dt / 60.0  # per command
  rows = []
  for i, cmd in enumerate(commands):
    n = max(counts[i].item(), 1.0)
    row = {"command": list(cmd)}
    for m in METRICS[:-1]:
      row[m] = sums[m][i].item() / n
    row["slip"] = sums["slip"][i].item() / max(slip_counts[i].item(), 1.0)
    row["falls_per_env_min"] = falls[i].item() / env_minutes
    rows.append(row)

  header = (
    f"{'command (vx, vy, yaw)':>24} {'vx':>7} {'err_xy':>7} {'err_yaw':>7} "
    f"{'torque2':>8} {'slip':>6} {'falls/min':>9}"
  )
  print(header)
  for r in rows:
    vx, vy, yaw = r["command"]
    print(
      f"{f'({vx:+.2f}, {vy:+.2f}, {yaw:+.2f})':>24} {r['vx']:7.3f} "
      f"{r['err_xy']:7.3f} {r['err_yaw']:7.3f} {r['torque_sq']:8.2f} "
      f"{r['slip']:6.3f} {r['falls_per_env_min']:9.3f}"
    )
  if args.out:
    Path(args.out).write_text(
      json.dumps({"checkpoint": str(ckpt), "rows": rows}, indent=2)
    )
    print(f"[eval] wrote {args.out}")
  env.close()


if __name__ == "__main__":
  main()
