"""G1 bimanual box carry: walk to a box, squeeze it between both palms, lift, carry, hold at a goal.

Task inputs the policy receives at deployment
  * box position and x axis in the pelvis frame (an external detector)
  * goal position in the pelvis frame (the operator)
  * joint encoders and the last action
Privileged state the teacher reads and JOSE reconstructs from joint encoders
plus joint torque (motor current), never from an IMU or a force sensor
  * base linear velocity, base angular velocity, projected gravity (as in walk)
  * per-hand box contact force, payload mass

No gripper. The 29-DOF G1 asset ends in a rigid rubber hand, so the grasp is a
bimanual squeeze on two opposite side faces, which also makes the contact force
and the payload visible in the arm torques. That is the point of the task for
the JOSE torque claim.

The scene, actuators, action scale on the legs, PPO hyper-parameters and the
locomotion regularisers are the walk task's (``walk_env_cfg.py``). New here are
the box, the pedestal, two box-filtered hand contact sensors, the staged task
reward in ``mdp/box_carry_mdp.py`` and a per-env distance curriculum.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from .mdp import box_carry_mdp as box
from .walk_env_cfg import FEET_BODIES, EventCfg, G1WalkEnvCfg, RobotSceneCfg

#: Bodies allowed to touch the box. The palm is the wrist-yaw link collider, the
#: forearm links may brace it.
HAND_CONTACT_BODIES_L = "left_wrist_.*"
HAND_CONTACT_BODIES_R = "right_wrist_.*"


@configclass
class BoxCarrySceneCfg(RobotSceneCfg):
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=box.BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # Cardboard on rubber, roughly. Randomised at startup below.
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.55, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, box.BOX_SPAWN_Z)),
    )

    pedestal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=box.PEDESTAL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, 0.5 * box.PEDESTAL_SIZE[2])),
    )

    left_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + HAND_CONTACT_BODIES_L,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
        history_length=1,
    )
    right_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + HAND_CONTACT_BODIES_R,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
        history_length=1,
    )


@configclass
class BoxCarryEventCfg(EventCfg):
    """Walk events plus box mass and friction randomisation and the box scene reset.

    ``reset_box_scene`` is declared after the inherited ``reset_base`` so it reads
    the new robot pose. Event terms run in declaration order.
    """

    box_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "mass_distribution_params": (0.5, 3.0),
            "operation": "abs",
        },
    )
    box_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    reset_box = EventTerm(func=box.reset_box_scene, mode="reset")

    def __post_init__(self):
        # Facing roughly forward keeps the level-0 box in reach. The curriculum
        # spreads the bearing to +-90 deg, so heading variety comes from there.
        self.reset_base.params["pose_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14, 3.14)}


@configclass
class BoxCarryActionsCfg:
    """Joint position targets. Arms get twice the walk task's 0.25 rad scale to reach the box sides."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        # Keys must not overlap: Isaac Lab rejects a joint matched by two keys.
        scale={
            ".*_hip_.*": 0.25, ".*_knee_joint": 0.25, ".*_ankle_.*": 0.25, "waist_.*": 0.25,
            ".*_shoulder_.*": 0.5, ".*_elbow_joint": 0.5, ".*_wrist_.*": 0.5,
        },
        use_default_offset=True,
    )


@configclass
class BoxCarryObservationsCfg:
    """Do not reorder the policy group. ``schema.BOX_CARRY_TERM_LAYOUT`` hardcodes it (540-D)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        hand_contact_force = ObsTerm(func=box.hand_contact_force_obs, noise=Unoise(n_min=-0.05, n_max=0.05))
        payload_mass = ObsTerm(func=box.payload_mass_obs, noise=Unoise(n_min=-0.2, n_max=0.2))
        box_pos_b = ObsTerm(func=box.box_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02))
        box_axis_b = ObsTerm(func=box.box_axis_b, noise=Unoise(n_min=-0.05, n_max=0.05))
        goal_pos_b = ObsTerm(func=box.goal_pos_b, noise=Unoise(n_min=-0.02, n_max=0.02))
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
        hand_contact_force = ObsTerm(func=box.hand_contact_force_obs)
        payload_mass = ObsTerm(func=box.payload_mass_obs)
        box_pos_b = ObsTerm(func=box.box_pos_b)
        box_axis_b = ObsTerm(func=box.box_axis_b)
        goal_pos_b = ObsTerm(func=box.goal_pos_b)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        # critic-only extras
        box_lin_vel_b = ObsTerm(func=box.box_lin_vel_b)
        hand_to_grasp_b = ObsTerm(func=box.hand_to_grasp_b)

        def __post_init__(self):
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BoxCarryRewardsCfg:
    """Staged task reward plus the walk task's regularisers at the walk task's weights.

    Stages are gated, not sequenced. ``approach_box`` switches off and
    ``carry_to_goal`` on at the moment both palms hold the box above the spawn,
    so there is no phase variable the policy cannot observe.
    """

    # -- task, in the order the robot meets them
    approach_box = RewTerm(func=box.approach_box, weight=1.0)
    face_target = RewTerm(func=box.face_target, weight=0.3)
    reach = RewTerm(func=box.reach_grasp_points, weight=1.5)
    squeeze = RewTerm(func=box.squeeze, weight=2.0)
    lift = RewTerm(func=box.lift_box, weight=3.0)
    carry = RewTerm(func=box.carry_to_goal, weight=3.0)
    hold = RewTerm(func=box.hold_at_goal, weight=4.0)
    delivered = RewTerm(func=box.delivered, weight=2.0)
    box_upright = RewTerm(func=box.box_upright, weight=-1.0)

    feet_air_time = RewTerm(
        func=box.feet_air_time_when_far,
        weight=0.75,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES), "threshold": 0.4},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # -- regularisers, from walk_env_cfg.RewardsCfg
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.76})
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FEET_BODIES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FEET_BODIES),
        },
    )
    # Anything but feet, forearms and hands touching anything is penalised. That
    # covers the torso hugging the box and the knees hitting the pedestal.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*(ankle|wrist|rubber_hand).*).*"]),
        },
    )


@configclass
class BoxCarryTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    box_dropped = DoneTerm(func=box.box_dropped)


@configclass
class BoxCarryCurriculumCfg:
    box_level = CurrTerm(func=box.box_level_curriculum)


@configclass
class G1BoxCarryEnvCfg(G1WalkEnvCfg):
    scene: BoxCarrySceneCfg = BoxCarrySceneCfg(num_envs=4096, env_spacing=8.0)
    observations: BoxCarryObservationsCfg = BoxCarryObservationsCfg()
    actions: BoxCarryActionsCfg = BoxCarryActionsCfg()
    rewards: BoxCarryRewardsCfg = BoxCarryRewardsCfg()
    terminations: BoxCarryTerminationsCfg = BoxCarryTerminationsCfg()
    events: BoxCarryEventCfg = BoxCarryEventCfg()
    curriculum: BoxCarryCurriculumCfg = BoxCarryCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # No velocity command in this task. The base class declares one; drop it
        # so nothing reads it by accident.
        self.commands = None
        self.scene.left_hand_contact.update_period = self.sim.dt
        self.scene.right_hand_contact.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class G1BoxCarryPlayEnvCfg(G1BoxCarryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None


@configclass
class G1BoxCarryEstimatorEnvCfg(G1BoxCarryEnvCfg):
    """Same task, policy corruption off, so teacher and JOSE read identical observations (see walk_estimator_env_cfg)."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class G1BoxCarryEstimatorPlayEnvCfg(G1BoxCarryPlayEnvCfg):
    pass
