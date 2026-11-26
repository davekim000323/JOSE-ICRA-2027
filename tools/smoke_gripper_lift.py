"""CPU smoke test of the gripper-lift task, no Isaac Lab needed.

Run ``python tools/smoke_gripper_lift.py`` from the repository root. Checks the
gripper URDF (30 joints, jaw geometry), grip centre, reward gates, success,
the 550-D schema and the JOSE injection with the all_gripper joint preset.
"""
import os
import sys
import types
import xml.etree.ElementTree as ET

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import smoke_box_carry as stubs  # noqa: E402  (installs the Isaac Lab stubs, runs the box checks)

import jose.ppo_walk.mdp.gripper_lift_mdp as grip  # noqa: E402
import jose.schema as schema  # noqa: E402
from jose.estimator.adapters import make_policy_adapter  # noqa: E402
from jose.estimator.torque import torque_noise_preset  # noqa: E402

N = stubs.N
close = stubs.close

# 1. URDF
root = ET.parse(os.path.join(ROOT, "usd", "g1_29dof_gripper.urdf")).getroot()
moving = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
assert len(moving) == 30 and moving[-1] == schema.G1_GRIPPER_JOINT_NAME
assert set(moving[:29]) == set(schema.G1_JOINT_NAMES)
assert "right_rubber_hand" not in [l.get("name") for l in root.findall("link")]
jaw = [j for j in root.findall("joint") if j.get("name") == "right_gripper_joint"][0]
assert jaw.find("axis").get("xyz") == "0 -1 0" and float(jaw.find("limit").get("upper")) == 0.06
print("gripper urdf ok")


class Robot(stubs.Robot):
    BODIES = ["pelvis", "right_wrist_yaw_link", "right_gripper_finger"]

    def __init__(self):
        super().__init__()
        self.data.joint_names = list(schema.JOINT_PRESETS["all_gripper"])
        self.data.joint_pos = torch.zeros(N, 30); self.data.joint_vel = torch.zeros(N, 30)
        self.data.applied_torque = torch.zeros(N, 30)
        self.data.joint_effort_limits = torch.full((N, 30), 20.0)
        self.data.root_lin_vel_b = torch.zeros(N, 3)


class Cmd:
    def __init__(self):
        self.cmd = torch.zeros(N, 7)

    def get_command(self, name):
        return self.cmd


class Env(stubs.Env):
    def __init__(self):
        super().__init__()
        self.scene["robot"] = Robot()
        self.robot = self.scene["robot"]
        self.scene["object"] = stubs.Rigid(torch.tensor(grip.OBJECT_SPAWN).repeat(N, 1))
        self.scene.sensors = {"finger_contact": stubs.Contact(), "palm_contact": stubs.Contact()}
        self.command_manager = Cmd()
        self.action_manager.action = torch.zeros(N, 30)

    def get_estimator_target(self):
        return torch.randn(N, 11)


env = Env()
robot, obj = env.scene["robot"], env.scene["object"]
F, P = env.scene.sensors["finger_contact"], env.scene.sensors["palm_contact"]

# 2. grip centre is 0.10 m along the wrist x axis
robot.data.body_pos_w[:, 1] = torch.tensor([0.2, -0.1, 0.9])
assert close(grip.grip_center_w(env), torch.tensor([0.3, -0.1, 0.9]))
robot.data.body_pos_w[:, 1] = obj.data.root_pos_w - torch.tensor([0.10, 0, 0])
assert close(grip.reach_object(env), 1.0)
print("grip centre and reach ok")

# 3. gates
F.data.force_matrix_w[:, 0, 0] = torch.tensor([0, 15.0, 0])
assert close(grip.jaw_contact(env), 0.0)
P.data.force_matrix_w[:, 0, 0] = torch.tensor([0, -15.0, 0])
assert close(grip.jaw_contact(env), 1.0) and close(grip.grip_force_obs(env), 0.75)
assert close(grip.object_lifted(env), 0.0) and close(grip.object_goal_tracking(env, 0.3), 0.0)
obj.data.root_pos_w[:, 2] += grip.LIFT_MARGIN + 0.01
assert close(grip.object_lifted(env), 1.0)
env.command_manager.cmd[:, :3] = obj.data.root_pos_w - robot.data.root_pos_w  # identity root quat
assert close(grip.object_goal_tracking(env, 0.05), 1.0) and close(grip.task_success(env), 1.0)
assert close(grip.target_pos_w(env), obj.data.root_pos_w)
P.data.force_matrix_w.zero_()
assert close(grip.task_success(env), 0.0), "success needs both jaws"
obj.data.root_pos_w[:, 2] = grip.OBJECT_DROP_Z - 0.01
assert grip.object_dropped(env).all()
assert close(grip.payload_mass_obs(env).squeeze(-1), torch.linspace(0.5, 3.0, N))
print("reward gates ok")

# 4. schema and injection, jaw joint in the estimator input
sch = schema.GRIPPER_LIFT_OBSERVATION_SCHEMA
assert sch.policy_dim == 550 and sch.estimator_target_dim == 11 and sch.action_dim == 30
hist = schema.gripper_lift_history_target_indices()
assert len(hist) == 55 and len(set(hist)) == 55 and max(hist) < 55
for arm in (dict(), dict(use_torque=True, torque_noise=torque_noise_preset("default")), dict(use_action=True)):
    ad = make_policy_adapter("gripper_lift", env, "all_gripper", **arm)
    width = 60 + (30 if arm else 0)
    assert ad.input_dim == width and ad.estimator_input().shape == (N, width), (arm, ad.input_dim)
    out = ad.inject_estimate(torch.zeros(N, 550), torch.arange(11.0).repeat(N, 1))
    assert close(out[:, list(sch.estimator_target_indices)], torch.arange(11.0) * torch.tensor(schema.gripper_lift_target_scales()))
    assert close(out[:, 55:], 0.0)
print("schema and injection ok")
print("ALL GRIPPER-LIFT SMOKE CHECKS PASSED")
