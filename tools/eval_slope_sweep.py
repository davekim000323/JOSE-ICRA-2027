"""Evaluate the teacher and JOSE estimators on terrain of one fixed slope.

Every tile is a pyramid or inverted pyramid of exactly ``--slope`` (rise over
run), so the result is a function of slope alone, not of where a curriculum
happened to leave the robots. One process evaluates one slope, because Isaac Sim
builds the scene once per process. Loop over slopes from a shell, e.g.

    for s in 0.0 0.1 0.2 0.3 0.4 0.5; do
      python tools/eval_slope_sweep.py --slope $s --headless \\
        --teacher-checkpoint <model.pt> \\
        --estimator-checkpoints logs/jose_g1/estimators/slope_*_seed42/best_estimator.pt \\
        --output logs/jose_g1/slope_sweep/slope_$s.json
    done

For each estimator it reports the closed-loop episode metrics (death rate,
episode length, tilt error) and the command-grid metrics (tracking error,
survival, gravity RMSE, tilt error). The teacher on ground-truth state is
reported on the same grid as the reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JOSE fixed-slope evaluation")
parser.add_argument("--slope", type=float, required=True, help="Slope ratio (rise over run), e.g. 0.3")
parser.add_argument(
    "--no-terrain-override", dest="no_terrain_override", action="store_true",
    help="Keep the task's own terrain and curriculum (diagnostic: reproduces the training-time evaluation)",
)
parser.add_argument("--task", default="Isaac-G1-PPO-Walk-Slope-Estimator-JOSE-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--teacher-checkpoint", dest="teacher_checkpoint", required=True)
parser.add_argument("--estimator-checkpoints", dest="estimator_checkpoints", nargs="*", default=[])
parser.add_argument("--num-envs", dest="num_envs", type=int, default=256)
parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=200)
parser.add_argument("--max-episode-steps", dest="max_episode_steps", type=int, default=1000)
parser.add_argument("--grid-settle-s", dest="grid_settle_s", type=float, default=1.0)
parser.add_argument("--grid-measure-s", dest="grid_measure_s", type=float, default=4.0)
parser.add_argument("--seed", type=int, default=10042)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401

from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.pipeline import (  # noqa: E402
    adapter_input_options,
    evaluate_estimator_closed_loop,
    evaluate_locomotion_grid,
    load_estimator,
)
from jose.ppo_walk.terrain_env_cfg import fixed_slope_terrain_cfg  # noqa: E402
from jose.teacher_setup import build_env_and_teacher  # noqa: E402


def _summary(closed: dict | None, grid: dict) -> dict:
    keys = (
        "death_rate", "episode_length_mean", "rmse", "target_rmse",
        "tilt_error_deg_mean", "tilt_error_deg_rmse",
    )
    out = {key: closed[key] for key in keys if closed and key in closed}
    for key, value in grid.items():
        if isinstance(value, (int, float, str, bool)) or key in ("grid_target_rmse",):
            out[f"grid/{key}" if not key.startswith("grid") else key] = value
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    if not args_cli.no_terrain_override:
        env_cfg.scene.terrain.terrain_generator = fixed_slope_terrain_cfg(args_cli.slope)
        env_cfg.scene.terrain.max_init_terrain_level = None
        if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

    env, teacher_agent = build_env_and_teacher(
        args_cli.task, "ppo_walk", env_cfg, agent_cfg,
        str(Path(args_cli.teacher_checkpoint).resolve()), args_cli.device, seed=args_cli.seed,
    )
    report = {
        "slope_ratio": None if args_cli.no_terrain_override else args_cli.slope,
        "terrain_override": not args_cli.no_terrain_override,
        "slope_deg": None if args_cli.no_terrain_override else math.degrees(math.atan(args_cli.slope)),
        "task": args_cli.task,
        "teacher_checkpoint": str(Path(args_cli.teacher_checkpoint).resolve()),
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "results": {},
    }

    base_adapter = make_policy_adapter("ppo_walk", env, "all")
    teacher_grid = evaluate_locomotion_grid(
        env, base_adapter, teacher_agent, None, "LSTM", 1,
        settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s, seed=args_cli.seed,
    )
    report["results"]["teacher_ground_truth"] = _summary(None, teacher_grid)
    print(f"[slope {args_cli.slope}] teacher: {json.dumps(report['results']['teacher_ground_truth'])}", flush=True)

    for path in args_cli.estimator_checkpoints:
        estimator, payload = load_estimator(path, args_cli.device)
        adapter = make_policy_adapter("ppo_walk", env, payload["joint_preset"], **adapter_input_options(payload))
        estimator_type = payload["model_config"]["type"]
        window = payload["window"]
        closed = evaluate_estimator_closed_loop(
            env, adapter, teacher_agent, estimator, estimator_type, window,
            args_cli.eval_episodes, args_cli.max_episode_steps, args_cli.seed,
        )
        grid = evaluate_locomotion_grid(
            env, adapter, teacher_agent, estimator, estimator_type, window,
            settle_s=args_cli.grid_settle_s, measure_s=args_cli.grid_measure_s, seed=args_cli.seed,
        )
        name = Path(path).parent.name
        report["results"][name] = {"checkpoint": str(Path(path).resolve()), **adapter.input_config(), **_summary(closed, grid)}
        print(f"[slope {args_cli.slope}] {name}: {json.dumps(report['results'][name])}", flush=True)

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
