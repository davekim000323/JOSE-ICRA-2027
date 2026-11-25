"""MDP terms for the G1 bimanual box-carry task.

The robot is told where a box is and where it must go. It walks to the box,
squeezes it between both hands (palm on one side face each, no gripper), lifts it
off a pedestal, carries it to the goal and holds it there.

Geometry conventions

* The box is close to a cube in x and y, so whichever pair of opposite side faces
  points most nearly along the robot's lateral axis is the grasp pair. This
  removes any dependence on the approach direction.
* The hand point is ``HAND_OFFSET`` along the ``*_wrist_yaw_link`` x axis, the
  palm centre of the bundled rubber hand (``left_hand_palm_joint`` origin 0.0415
  plus half the palm).
* Every per-episode quantity (goal, spawn height, curriculum level) lives on the
  env as a lazily created buffer, so the terms need no custom env subclass.

Everything here is plain torch on ``env.scene`` and is unit-tested with a fake
scene by ``tools/smoke_box_carry.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, yaw_quat
except ImportError:  # Isaac Lab < 2.1
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, yaw_quat
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------

BOX_SIZE = (0.28, 0.28, 0.24)
PEDESTAL_SIZE = (0.36, 0.36, 0.70)
PEDESTAL_TOP = PEDESTAL_SIZE[2]
BOX_SPAWN_Z = PEDESTAL_TOP + 0.5 * BOX_SIZE[2] + 0.002
HAND_OFFSET = 0.08
#: Contact force normaliser, N. 50 N per hand squeezes a 3 kg box at mu = 0.5.
CONTACT_FORCE_SCALE = 50.0
CONTACT_THRESHOLD_N = 2.0
LIFT_HEIGHT = 0.10
#: Below this the box is on the floor or falling there.
BOX_DROP_Z = 0.45
#: A box within this xy distance of the goal and lifted counts as delivered.
GOAL_RADIUS = 0.15
NUM_LEVELS = 10

HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


# ---------------------------------------------------------------------------
# Per-env task buffers
# ---------------------------------------------------------------------------


def _buf(env, name: str, shape: tuple, dtype=torch.float32, fill: float = 0.0) -> torch.Tensor:
    value = getattr(env, name, None)
    if value is None:
        value = torch.full((env.num_envs, *shape), fill, device=env.device, dtype=dtype)
        setattr(env, name, value)
    return value


def goal_pos_w(env) -> torch.Tensor:
    """Goal position of the box centre, world frame, (N, 3)."""
    return _buf(env, "_box_goal_w", (3,))


def curriculum_level(env) -> torch.Tensor:
    return _buf(env, "_box_level", (), dtype=torch.long, fill=0)


def _hand_ids(env, asset_cfg: SceneEntityCfg) -> list[int]:
    ids = getattr(env, "_box_hand_ids", None)
    if ids is None:
        robot = env.scene[asset_cfg.name]
        ids = [robot.find_bodies(name)[0][0] for name in HAND_BODIES]
        env._box_hand_ids = ids
    return ids


def hand_points_w(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Palm centres of both hands, world frame, (N, 2, 3)."""
    robot = env.scene[asset_cfg.name]
    ids = _hand_ids(env, asset_cfg)
    pos = robot.data.body_pos_w[:, ids]
    quat = robot.data.body_quat_w[:, ids]
    offset = torch.tensor((HAND_OFFSET, 0.0, 0.0), device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + quat_apply(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape(pos.shape)


def grasp_points_w(env, robot_cfg=SceneEntityCfg("robot"), box_cfg=SceneEntityCfg("box"), margin: float = 0.01):
    """Target palm positions on the two side faces that face the robot's left and right, (N, 2, 3)."""
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    n = box.data.root_pos_w.shape[0]
    q = box.data.root_quat_w
    ex = quat_apply(q, torch.tensor((1.0, 0.0, 0.0), device=q.device).expand(n, 3))
    ey = quat_apply(q, torch.tensor((0.0, 1.0, 0.0), device=q.device).expand(n, 3))
    left = quat_apply(robot.data.root_quat_w, torch.tensor((0.0, 1.0, 0.0), device=q.device).expand(n, 3))
    # Candidate normals +-ex, +-ey; pick the one most aligned with robot left.
    normals = torch.stack((ex, -ex, ey, -ey), dim=1)  # (N, 4, 3)
    half = torch.tensor((BOX_SIZE[0], BOX_SIZE[0], BOX_SIZE[1], BOX_SIZE[1]), device=q.device) * 0.5
    best = torch.argmax((normals * left.unsqueeze(1)).sum(-1), dim=1)
    idx = torch.arange(n, device=q.device)
    normal = normals[idx, best]
    reach = (half[best] + margin).unsqueeze(-1)
    centre = box.data.root_pos_w
    return torch.stack((centre + normal * reach, centre - normal * reach), dim=1)


def hand_contact_forces(env, sensor_names=("left_hand_contact", "right_hand_contact")) -> torch.Tensor:
    """Magnitude of the box-filtered contact force on each hand, N, (N, 2)."""
    out = []
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        matrix = sensor.data.force_matrix_w  # (N, bodies, filters, 3)
        out.append(matrix.sum(dim=(1, 2)).norm(dim=-1) if matrix is not None else torch.zeros(env.num_envs, device=env.device))
    return torch.stack(out, dim=-1)


def _grasped(env) -> torch.Tensor:
    return (hand_contact_forces(env) > CONTACT_THRESHOLD_N).all(dim=-1)


def _lifted(env, box_cfg=SceneEntityCfg("box")) -> torch.Tensor:
    box = env.scene[box_cfg.name]
    return _grasped(env) & (box.data.root_pos_w[:, 2] > BOX_SPAWN_Z + 0.5 * LIFT_HEIGHT)


def _payload_mass(env, box_cfg=SceneEntityCfg("box")) -> torch.Tensor:
    """Box mass per env, kg, (N,). Read once from PhysX after the startup randomisation."""
    mass = getattr(env, "_box_mass", None)
    if mass is None:
        box = env.scene[box_cfg.name]
        mass = box.root_physx_view.get_masses().reshape(env.num_envs, -1).sum(-1).to(env.device)
        env._box_mass = mass
    return mass


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _to_base(env, vec_w: torch.Tensor, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, vec_w - robot.data.root_pos_w)


def box_pos_b(env, box_cfg: SceneEntityCfg = SceneEntityCfg("box")) -> torch.Tensor:
    """Box centre in the pelvis frame. Deployment source is external perception."""
    return _to_base(env, env.scene[box_cfg.name].data.root_pos_w)


def box_axis_b(env, box_cfg: SceneEntityCfg = SceneEntityCfg("box")) -> torch.Tensor:
    """Box x axis in the pelvis frame, the box orientation a detector reports."""
    robot = env.scene["robot"]
    q = env.scene[box_cfg.name].data.root_quat_w
    ex = quat_apply(q, torch.tensor((1.0, 0.0, 0.0), device=q.device).expand(q.shape[0], 3))
    return quat_apply_inverse(robot.data.root_quat_w, ex)


def goal_pos_b(env) -> torch.Tensor:
    """Goal box position in the pelvis frame. Deployment source is the operator."""
    return _to_base(env, goal_pos_w(env))


def hand_contact_force_obs(env) -> torch.Tensor:
    """Privileged. Per-hand box contact force in units of CONTACT_FORCE_SCALE, clipped at 2."""
    return (hand_contact_forces(env) / CONTACT_FORCE_SCALE).clamp(max=2.0)


def payload_mass_obs(env) -> torch.Tensor:
    """Privileged. Box mass in kg, (N, 1)."""
    return _payload_mass(env).unsqueeze(-1)


def box_lin_vel_b(env, box_cfg: SceneEntityCfg = SceneEntityCfg("box")) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, env.scene[box_cfg.name].data.root_lin_vel_w)


def hand_to_grasp_b(env) -> torch.Tensor:
    """Critic only. Palm-to-grasp-point vectors in the pelvis frame, (N, 6)."""
    robot = env.scene["robot"]
    delta = grasp_points_w(env) - hand_points_w(env)
    q = robot.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4)
    return quat_apply_inverse(q, delta.reshape(-1, 3)).reshape(env.num_envs, 6)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def approach_box(env, standoff: float = 0.35, std: float = 0.6) -> torch.Tensor:
    """Pelvis to within ``standoff`` of the box in xy. Active until the box is lifted."""
    robot = env.scene["robot"]
    box = env.scene["box"]
    d = (box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]).norm(dim=-1)
    r = torch.exp(-((d - standoff).clamp(min=0.0) / std) ** 2)
    return r * (~_lifted(env)).float()


def face_target(env) -> torch.Tensor:
    """Pelvis heading toward the box before the lift and toward the goal after, in [0, 1]."""
    robot = env.scene["robot"]
    box = env.scene["box"]
    lifted = _lifted(env).unsqueeze(-1)
    target = torch.where(lifted, goal_pos_w(env), box.data.root_pos_w)
    to = target[:, :2] - robot.data.root_pos_w[:, :2]
    dist = to.norm(dim=-1)
    to = to / dist.clamp(min=1e-6).unsqueeze(-1)
    n = to.shape[0]
    fwd = quat_apply(robot.data.root_quat_w, torch.tensor((1.0, 0.0, 0.0), device=to.device).expand(n, 3))[:, :2]
    fwd = fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r = 0.5 * (1.0 + (fwd * to).sum(-1))
    # Once on top of the target the heading is undefined; pay full.
    return torch.where(dist < 0.3, torch.ones_like(r), r)


def reach_grasp_points(env, std: float = 0.15) -> torch.Tensor:
    """Mean over hands of exp(-|palm - grasp point|^2 / std^2)."""
    d2 = (grasp_points_w(env) - hand_points_w(env)).square().sum(-1)
    return torch.exp(-d2 / std**2).mean(-1)


def squeeze(env, saturation_n: float = 30.0) -> torch.Tensor:
    """Weaker hand's contact force, saturating, paid only when both hands touch."""
    f = hand_contact_forces(env)
    return (f.min(dim=-1).values / saturation_n).clamp(max=1.0) * _grasped(env).float()


def lift_box(env) -> torch.Tensor:
    """Box height above its spawn, saturating at LIFT_HEIGHT, while grasped."""
    box = env.scene["box"]
    h = ((box.data.root_pos_w[:, 2] - BOX_SPAWN_Z) / LIFT_HEIGHT).clamp(0.0, 1.0)
    return h * _grasped(env).float()


def carry_to_goal(env, std: float = 1.0) -> torch.Tensor:
    """Coarse xy pull of the box toward the goal, while lifted."""
    box = env.scene["box"]
    d2 = (box.data.root_pos_w[:, :2] - goal_pos_w(env)[:, :2]).square().sum(-1)
    return torch.exp(-d2 / std**2) * _lifted(env).float()


def hold_at_goal(env, std: float = 0.12) -> torch.Tensor:
    """Fine 3-D term, box centre at the goal point while lifted."""
    box = env.scene["box"]
    d2 = (box.data.root_pos_w - goal_pos_w(env)).square().sum(-1)
    return torch.exp(-d2 / std**2) * _lifted(env).float()


def box_upright(env) -> torch.Tensor:
    """1 - z component of the box up axis, 0 when level. Use with a negative weight."""
    box = env.scene["box"]
    q = box.data.root_quat_w
    up = quat_apply(q, torch.tensor((0.0, 0.0, 1.0), device=q.device).expand(q.shape[0], 3))
    return 1.0 - up[:, 2]


def feet_air_time_when_far(
    env, sensor_cfg: SceneEntityCfg, threshold: float = 0.4, far: float = 0.6
) -> torch.Tensor:
    """``feet_air_time_positive_biped`` gated by distance to the current target, not a velocity command.

    The walk task gates on |cmd| > 0.1. Here there is no command, so the gate is
    whether the pelvis still has more than ``far`` metres to cover to the box
    (before the lift) or the box to the goal (after).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    air = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact > 0.0
    mode_time = torch.where(in_contact, contact, air)
    single_stance = in_contact.int().sum(dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), mode_time, 0.0), dim=1)[0]
    reward = reward.clamp(max=threshold)
    robot = env.scene["robot"]
    box = env.scene["box"]
    lifted = _lifted(env)
    d_box = (box.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2]).norm(dim=-1)
    d_goal = (box.data.root_pos_w[:, :2] - goal_pos_w(env)[:, :2]).norm(dim=-1)
    dist = torch.where(lifted, d_goal, d_box - 0.35)
    return reward * (dist > far).float()


def delivered(env) -> torch.Tensor:
    """1 while the lifted box is inside GOAL_RADIUS of the goal in xy. Also the success metric."""
    box = env.scene["box"]
    d = (box.data.root_pos_w[:, :2] - goal_pos_w(env)[:, :2]).norm(dim=-1)
    return ((d < GOAL_RADIUS) & _lifted(env)).float()


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def box_dropped(env, minimum_height: float = BOX_DROP_Z) -> torch.Tensor:
    return env.scene["box"].data.root_pos_w[:, 2] < minimum_height


# ---------------------------------------------------------------------------
# Events and curriculum
# ---------------------------------------------------------------------------


def level_ranges(level: torch.Tensor) -> dict[str, torch.Tensor]:
    """Spawn ranges per curriculum level, 0 .. NUM_LEVELS-1, linear in the level.

    level 0   box 0.40-0.50 m dead ahead, within arm reach, goal 0.30 m from the box
    level 9   box 0.40-2.50 m within +-90 deg, goal 0.30-3.00 m any direction
    """
    s = level.float() / (NUM_LEVELS - 1)
    return {
        "box_dist_max": 0.50 + 2.00 * s,
        "box_bearing_max": (math.pi / 2) * s,
        "goal_dist_max": 0.30 + 2.70 * s,
    }


def reset_box_scene(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
    pedestal_cfg: SceneEntityCfg = SceneEntityCfg("pedestal"),
):
    """Place pedestal and box in front of the reset robot and sample a goal.

    Must run after ``reset_base`` so the robot root is already at its new pose.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    robot = env.scene[robot_cfg.name]
    box = env.scene[box_cfg.name]
    pedestal = env.scene[pedestal_cfg.name]
    n = len(env_ids)
    dev = env.device
    rng = level_ranges(curriculum_level(env)[env_ids])

    root = robot.data.root_pos_w[env_ids]
    yaw_q = yaw_quat(robot.data.root_quat_w[env_ids])
    fwd = quat_apply(yaw_q, torch.tensor((1.0, 0.0, 0.0), device=dev).expand(n, 3))
    heading = torch.atan2(fwd[:, 1], fwd[:, 0])

    u = torch.rand(n, device=dev)
    dist = 0.40 + u * (rng["box_dist_max"] - 0.40)
    bearing = heading + (2.0 * torch.rand(n, device=dev) - 1.0) * rng["box_bearing_max"]
    xy = root[:, :2] + torch.stack((torch.cos(bearing), torch.sin(bearing)), -1) * dist.unsqueeze(-1)

    zeros = torch.zeros(n, device=dev)
    box_yaw = 2.0 * math.pi * torch.rand(n, device=dev)
    ped_pose = torch.zeros(n, 7, device=dev)
    ped_pose[:, :2] = xy
    ped_pose[:, 2] = 0.5 * PEDESTAL_SIZE[2]
    ped_pose[:, 3:] = quat_from_euler_xyz(zeros, zeros, box_yaw)
    pedestal.write_root_pose_to_sim(ped_pose, env_ids=env_ids)

    box_pose = ped_pose.clone()
    box_pose[:, 2] = BOX_SPAWN_Z
    box.write_root_pose_to_sim(box_pose, env_ids=env_ids)
    box.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    # Goal: any direction from the box, 0.30 m or more from the pedestal centre,
    # and LIFT_HEIGHT above the spawn so it cannot be met by a box on the pedestal.
    gd = 0.30 + torch.rand(n, device=dev) * (rng["goal_dist_max"] - 0.30)
    ga = 2.0 * math.pi * torch.rand(n, device=dev)
    goal = torch.zeros(n, 3, device=dev)
    goal[:, :2] = xy + torch.stack((torch.cos(ga), torch.sin(ga)), -1) * gd.unsqueeze(-1)
    goal[:, 2] = BOX_SPAWN_Z + LIFT_HEIGHT
    goal_pos_w(env)[env_ids] = goal


def box_level_curriculum(env, env_ids: torch.Tensor) -> torch.Tensor:
    """Raise the level of envs that delivered the box at episode end, lower it for drops and falls.

    Runs at reset before the events, so the box and termination state still
    describe the episode that just ended. Returns the mean level for logging.
    """
    level = curriculum_level(env)
    if env_ids is None or len(env_ids) == 0:
        return level.float().mean()
    if env.common_step_counter > 0:
        success = delivered(env)[env_ids] > 0.5
        tm = env.termination_manager
        failed = tm.terminated[env_ids]
        level[env_ids] = torch.where(success, level[env_ids] + 1, level[env_ids])
        level[env_ids] = torch.where(failed & ~success, level[env_ids] - 1, level[env_ids])
        level.clamp_(0, NUM_LEVELS - 1)
    return level.float().mean()
