"""G1 standing pick and lift with a 1-DOF jaw gripper on the right wrist.

Short-horizon companion of ``box_carry_env_cfg.py``, sized to train in under a
day on one 8 GB GPU. The robot stands at a 0.78 m table, a 45 mm cube of mass
U[0.05, 0.5] kg spawns on it within reach of the right hand, and a target point
0.17 to 0.32 m above the pelvis is sampled per episode. The policy reaches,
closes the jaw, lifts and holds the cube at the target while the legs keep
balance.

Task inputs at deployment: object position and x axis and target position in
the pelvis frame, joint encoders, the last action. Privileged state the teacher
reads and JOSE reconstructs from encoders plus joint torque: base linear and
angular velocity, projected gravity, the jaw grip force and the payload mass.

Asset is ``G1_GRIPPER_CFG`` (30 joints). The jaw is a binary action, open or
close, so the action is 29 joint offsets plus one gripper command (30-D).
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs.mdp import UniformPoseCommandCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .g1_asset import G1_GRIPPER_CFG, GRIPPER_CLOSE, GRIPPER_JOINT, GRIPPER_OPEN
from .mdp import gripper_lift_mdp as grip
from .walk_env_cfg import FEET_BODIES, EventCfg, G1WalkEnvCfg, RobotSceneCfg

#: Every body joint, so the gripper joint is excluded from the offset action.
BODY_JOINTS = [
    ".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "waist_.*",
    ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_.*",
]


@configclass
class GripperLiftSceneCfg(RobotSceneCfg):
    robot: ArticulationCfg = G1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=grip.TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=grip.TABLE_POS),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(grip.OBJECT_SIZE,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.8)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=grip.OBJECT_SPAWN),
    )

    finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_finger",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=1,
    )
    palm_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + grip.WRIST_BODY,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        history_length=1,
    )


@configclass
class GripperLiftCommandsCfg:
    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="pelvis",
        resampling_time_range=(10.0, 10.0),  # one target per 8 s episode
        debug_vis=False,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.20, 0.40), pos_y=(-0.35, -0.05), pos_z=(0.17, 0.32),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class GripperLiftActionsCfg:
    body = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=BODY_JOINTS,
        # Keys must not overlap: Isaac Lab rejects a joint matched by two keys.
        scale={
            ".*_hip_.*": 0.25, ".*_knee_joint": 0.25, ".*_ankle_.*": 0.25, "waist_.*": 0.25,
            ".*_shoulder_.*": 0.5, ".*_elbow_joint": 0.5, ".*_wrist_.*": 0.5,
        },
        use_default_offset=True,
        preserve_order=True,
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=[GRIPPER_JOINT],
        open_command_expr={GRIPPER_JOINT: GRIPPER_OPEN},
        close_command_expr={GRIPPER_JOINT: GRIPPER_CLOSE},
    )


@configclass
class GripperLiftObservationsCfg:
    """Do not reorder the policy group. ``schema.GRIPPER_LIFT_TERM_LAYOUT`` hardcodes it (550-D)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        grip_force = ObsTerm(func=grip.grip_force_obs, noise=Unoise(n_min=-0.05, n_max=0.05))
        payload_mass = ObsTerm(func=grip.payload_mass_obs, noise=Unoise(n_min=-0.05, n_max=0.05))
        object_pos_b = ObsTerm(func=grip.object_pos_b, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_axis_b = ObsTerm(func=grip.object_axis_b, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_pos_b = ObsTerm(func=grip.target_pos_b)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        grip_force = ObsTerm(func=grip.grip_force_obs)
        payload_mass = ObsTerm(func=grip.payload_mass_obs)
        object_pos_b = ObsTerm(func=grip.object_pos_b)
        object_axis_b = ObsTerm(func=grip.object_axis_b)
        target_pos_b = ObsTerm(func=grip.target_pos_b)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        object_lin_vel_b = ObsTerm(func=grip.object_lin_vel_b)
        grip_to_object_b = ObsTerm(func=grip.grip_to_object_b)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class GripperLiftRewardsCfg:
    # -- task, Isaac Lab Franka lift weights
    reach = RewTerm(func=grip.reach_object, weight=1.0, params={"std": 0.1})
    jaw_contact = RewTerm(func=grip.jaw_contact, weight=1.0)
    lift = RewTerm(func=grip.object_lifted, weight=15.0)
    goal_coarse = RewTerm(func=grip.object_goal_tracking, weight=16.0, params={"std": 0.3})
    goal_fine = RewTerm(func=grip.object_goal_tracking, weight=5.0, params={"std": 0.05})

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- standing and smoothness, walk weights unless noted
    base_xy_velocity = RewTerm(func=grip.base_xy_vel_l2, weight=-0.5)
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.76})
    # All leg joints, stronger than walk: this task wants a fixed stance.
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    joint_deviation_left_arm = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["left_shoulder_.*", "left_elbow_joint", "left_wrist_.*"])},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["(?!.*(ankle|right_wrist|right_gripper).*).*"]
            ),
        },
    )


@configclass
class GripperLiftTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.6})
    object_dropped = DoneTerm(func=grip.object_dropped)


@configclass
class GripperLiftEventCfg(EventCfg):
    object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"), "mass_distribution_params": (0.05, 0.5), "operation": "abs"},
    )
    object_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.5, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.10, 0.10), "yaw": (-math.pi / 4, math.pi / 4)},
            "velocity_range": {},
        },
    )

    def __post_init__(self):
        # Standing at a fixed table. Small placement noise only.
        self.reset_base.params["pose_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "yaw": (-0.1, 0.1)}
        self.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)


@configclass
class G1GripperLiftEnvCfg(G1WalkEnvCfg):
    scene: GripperLiftSceneCfg = GripperLiftSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: GripperLiftObservationsCfg = GripperLiftObservationsCfg()
    actions: GripperLiftActionsCfg = GripperLiftActionsCfg()
    commands: GripperLiftCommandsCfg = GripperLiftCommandsCfg()
    rewards: GripperLiftRewardsCfg = GripperLiftRewardsCfg()
    terminations: GripperLiftTerminationsCfg = GripperLiftTerminationsCfg()
    events: GripperLiftEventCfg = GripperLiftEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 8.0
        self.scene.finger_contact.update_period = self.sim.dt
        self.scene.palm_contact.update_period = self.sim.dt


@configclass
class G1GripperLiftPlayEnvCfg(G1GripperLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class G1GripperLiftEstimatorEnvCfg(G1GripperLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class G1GripperLiftEstimatorPlayEnvCfg(G1GripperLiftPlayEnvCfg):
    pass
