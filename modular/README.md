# Modular JOSE

JOSE replaces the IMU. Every learned humanoid controller checked for this module
reads the same two inertial terms, projected gravity and base angular velocity,
and several humanoid VLAs read torso roll and pitch as state. Modular JOSE keeps
that sensor contract and swaps the sensor. The VLA above and the controller below
stay frozen.

```
joints (q, dq, tau_est or current)
   │
   ▼
FeatureBuilder ──► JoseRuntime ──► StateEstimate (pelvis frame)
 native|canonical   window, model     │
                                      ├──► port ──► frozen low-level controller
                                      │    unitree_rl_gym, booster_t1,
                                      │    jose_ppo_walk, jose_box_carry, imu_message
                                      │
                                      └──► HighLevelAdapter.encode_state ──► VLA
VLA action chunk ──► HighLevelAdapter.decode_action ──► WholeBodyCommand ──► controller
```

Nothing here imports Isaac Lab. It needs `torch`, `numpy` and `pyyaml`.

## What is configurable

| Axis | File | Shipped |
|---|---|---|
| Robot | `configs/embodiments/*.yaml` | `g1_29dof`, `g1_23dof`, `g1_29dof_gripper`, `h1`, `h1_2`, `t1` |
| Low-level controller | `configs/ports/*.yaml` | `unitree_rl_gym`, `booster_t1`, `jose_ppo_walk`, `jose_box_carry`, and `imu_message` |
| High-level policy | `configs/highlevel/*.yaml` | `openpi_flat`, `groot_dict`, `openhlm_body`, `psi0_gripper`, `openvla_tokens`, `sonic_latent`, `wholebodyvla_flags` |

Joint names, order and effort limits of H1, H1-2 and T1 are taken from the vendor
URDFs (`unitree_ros`, `booster_gym`). The `unitree_rl_gym` and `booster_t1` port
layouts are taken from the vendors' deploy scripts. The VLA presets are written
from each paper's text and are style templates. Check them against the
checkpoint's own modality config before running a released model.

The gripper joint of `g1_29dof_gripper` is a placeholder. Set its name, range and
effort limit to the 1-DoF gripper actually mounted.

## Usage

```python
from modular import FeatureBuilder, JoseRuntime, JointReading, build_stack

def runtime(spec):
    return JoseRuntime.from_checkpoint("jose.pt", FeatureBuilder(spec, "native", use_torque=True))

stack = build_stack("g1_29dof_gripper", "imu_message", runtime, high_level="openpi_flat")

# every control tick (50 Hz)
reading = JointReading(q, dq, tau_est)
low_state = stack.tick(reading, low_state)       # IMU fields now hold the JOSE estimate
vla_state = stack.high_level_state(reading)      # torso roll and pitch come from JOSE

# every VLA call
for cmd in stack.command(action_chunk, reading):
    ...                                          # joint targets, gripper, base velocity
```

## Two feature layouts

`native` is the input the current G1 checkpoints were trained on. With
`g1_29dof` it is column-identical to `estimator/adapters.py`, so those weights
load through `JoseRuntime.from_checkpoint` without Isaac Lab.

`canonical` maps any robot onto 33 humanoid joint slots, each with position
offset, velocity, torque over effort limit and a presence mask. One network then
reads every shipped robot. Training that network needs the Isaac Lab rollouts of
each robot and is not part of this change.

## Frames

JOSE estimates in the pelvis. `imu.chain` in an embodiment YAML lists the joints
between the pelvis and the IMU. The H1 and H1-2 IMU sits above `torso_joint`, so
the `imu_message` port writes a torso-frame message and Unitree's own
`transform_imu_data` recovers the pelvis estimate. `tests/modular/test_ports.py`
checks this against a port of the Unitree code.

## Limits

* Yaw is integrated from the estimated yaw rate and drifts.
* The accelerometer field of `imu_message` is quasi-static, minus g times
  projected gravity, without linear acceleration.
* The `current` torque source needs a per-joint torque constant identified on the
  robot. The shipped robots use the vendor `tau_est`.

## Tests

```
python -m pytest tests/modular
```
