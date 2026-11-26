"""MDP terms for the G1 standing pick-and-lift task with a 1-DOF jaw gripper.

The robot stands at a table. It is told where a small cube is and where to take
it, reaches with the right arm, closes the jaw on the cube, lifts it and holds
it at the target, while the legs keep balance against the moving arm and payload.

The reward follows the Isaac Lab Franka lift recipe (reach, lift, coarse and
fine goal tracking) plus a two-jaw contact term and the walk task's posture
regularisers. All terms are plain torch on ``env.scene`` and are covered by
``tools/smoke_gripper_lift.py``.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

try:
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
except ImportError:  # Isaac Lab < 2.1
    from isaaclab.utils.math import quat_apply
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse


OBJECT_SIZE = 0.045
TABLE_SIZE = (0.30, 0.50, 0.78)
TABLE_POS = (0.35, -0.12, 0.5 * TABLE_SIZE[2])
TABLE_TOP = TABLE_SIZE[2]
OBJECT_SPAWN = (0.30, -0.15, TABLE_TOP + 0.5 * OBJECT_SIZE + 0.002)
#: Object centre this far above its resting height counts as lifted.
LIFT_MARGIN = 0.04
#: Below this the object has fallen off the table.
OBJECT_DROP_Z = TABLE_TOP - 0.10
#: Jaw centre in the wrist-yaw link frame. The jaws sit at x 0.065..0.135 and
#: y -0.04 / +0.04 when open (tools/make_gripper_urdf.py).
GRIP_CENTER = (0.10, 0.0, 0.0)
#: 20 N is the jaw drive effort limit.
GRIP_FORCE_SCALE = 20.0
CONTACT_THRESHOLD_N = 1.0
#: Delivered when the object is within this distance of the target.
SUCCESS_RADIUS = 0.05

WRIST_BODY = "right_wrist_yaw_link"


def _wrist_id(env) -> int:
    idx = getattr(env, "_grip_wrist_id", None)
    if idx is None:
        idx = env.scene["robot"].find_bodies(WRIST_BODY)[0][0]
        env._grip_wrist_id = idx
    return idx


def grip_center_w(env) -> torch.Tensor:
    """Midpoint between the open jaws, world frame, (N, 3)."""
    robot = env.scene["robot"]
    i = _wrist_id(env)
    pos = robot.data.body_pos_w[:, i]
    quat = robot.data.body_quat_w[:, i]
    off = torch.tensor(GRIP_CENTER, device=pos.device, dtype=pos.dtype).expand_as(pos)
    return pos + quat_apply(quat, off)


def jaw_forces(env, sensors=("finger_contact", "palm_contact")) -> torch.Tensor:
    """Object-filtered contact force magnitude on the moving and the fixed jaw, N, (N, 2)."""
    out = []
    for name in sensors:
        matrix = env.scene.sensors[name].data.force_matrix_w
        out.append(matrix.sum(dim=(1, 2)).norm(dim=-1))
    return torch.stack(out, dim=-1)


def _grasped(env) -> torch.Tensor:
    return (jaw_forces(env) > CONTACT_THRESHOLD_N).all(dim=-1)


def object_height(env) -> torch.Tensor:
    return env.scene["object"].data.root_pos_w[:, 2]


def _lifted(env) -> torch.Tensor:
    return object_height(env) > OBJECT_SPAWN[2] + LIFT_MARGIN


def target_pos_w(env, command_name: str = "object_pose") -> torch.Tensor:
    """Target object position, world frame. The command is stored in the pelvis frame."""
    robot = env.scene["robot"]
    cmd_b = env.command_manager.get_command(command_name)[:, :3]
    return robot.data.root_pos_w + quat_apply(robot.data.root_quat_w, cmd_b)


def _payload_mass(env) -> torch.Tensor:
    mass = getattr(env, "_object_mass", None)
    if mass is None:
        obj = env.scene["object"]
        mass = obj.root_physx_view.get_masses().reshape(env.num_envs, -1).sum(-1).to(env.device)
        env._object_mass = mass
    return mass


# ---------------------------------------------------------------- observations


def object_pos_b(env) -> torch.Tensor:
    """Object centre in the pelvis frame. Deployment source is an external detector."""
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, env.scene["object"].data.root_pos_w - robot.data.root_pos_w)


def object_axis_b(env) -> torch.Tensor:
    """Object x axis in the pelvis frame, the yaw a detector reports."""
    robot = env.scene["robot"]
    q = env.scene["object"].data.root_quat_w
    ex = quat_apply(q, torch.tensor((1.0, 0.0, 0.0), device=q.device).expand(q.shape[0], 3))
    return quat_apply_inverse(robot.data.root_quat_w, ex)


def target_pos_b(env, command_name: str = "object_pose") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, :3]


def grip_force_obs(env) -> torch.Tensor:
    """Privileged. Moving-jaw force on the object over 20 N, clipped at 1.5, (N, 1)."""
    return (jaw_forces(env)[:, :1] / GRIP_FORCE_SCALE).clamp(max=1.5)


def payload_mass_obs(env) -> torch.Tensor:
    """Privileged. Object mass, kg, (N, 1)."""
    return _payload_mass(env).unsqueeze(-1)


def object_lin_vel_b(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, env.scene["object"].data.root_lin_vel_w)


def grip_to_object_b(env) -> torch.Tensor:
    """Critic only. Jaw centre to object, pelvis frame, (N, 3)."""
    robot = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, env.scene["object"].data.root_pos_w - grip_center_w(env))


# ---------------------------------------------------------------- rewards


def reach_object(env, std: float = 0.1) -> torch.Tensor:
    d = (env.scene["object"].data.root_pos_w - grip_center_w(env)).norm(dim=-1)
    return 1.0 - torch.tanh(d / std)


def jaw_contact(env) -> torch.Tensor:
    """1 when both jaws press the object."""
    return _grasped(env).float()


def object_lifted(env) -> torch.Tensor:
    return _lifted(env).float()


def object_goal_tracking(env, std: float, command_name: str = "object_pose") -> torch.Tensor:
    d = (env.scene["object"].data.root_pos_w - target_pos_w(env, command_name)).norm(dim=-1)
    return (1.0 - torch.tanh(d / std)) * _lifted(env).float()


def base_xy_vel_l2(env) -> torch.Tensor:
    """Standing task, the pelvis should not drift."""
    return env.scene["robot"].data.root_lin_vel_b[:, :2].square().sum(-1)


def task_success(env, command_name: str = "object_pose") -> torch.Tensor:
    d = (env.scene["object"].data.root_pos_w - target_pos_w(env, command_name)).norm(dim=-1)
    return ((d < SUCCESS_RADIUS) & _lifted(env) & _grasped(env)).float()


# ---------------------------------------------------------------- terminations


def object_dropped(env, minimum_height: float = OBJECT_DROP_Z) -> torch.Tensor:
    return object_height(env) < minimum_height
