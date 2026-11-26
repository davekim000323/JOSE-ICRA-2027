"""Modular JOSE, an IMU-free estimator that plugs under any high-level policy and any humanoid.

The package has no Isaac Lab dependency. It needs only ``torch``, ``numpy`` and
``pyyaml``, so it runs on the robot computer and in CPU tests.

Layers, from the hardware up.

* ``embodiment``  what a robot is. Joint names, IMU mounting, torque source, grippers.
* ``frames``      moves gravity and angular velocity between the pelvis and the IMU frame.
* ``features``    turns one joint reading into the estimator input, native or canonical.
* ``runtime``     history window, model call, output parsing, yaw integration.
* ``ports``       writes the estimate where a frozen low-level controller reads its IMU.
* ``highlevel``   encodes proprioceptive state for a VLA and decodes its action chunk.
* ``stack``       wires the layers into one object per robot and policy pair.

The estimator never sees which VLA runs above it. The VLA never sees that the IMU
is gone. That separation is the point of the design.
"""

from .embodiment import EmbodimentSpec, GripperSpec, load_embodiment, list_embodiments
from .signals import JointReading, StateEstimate, WholeBodyCommand
from .features import FeatureBuilder, CANONICAL_SLOTS
from .frames import SensorChain
from .runtime import JoseRuntime
from .ports import make_port, list_port_presets
from .highlevel import make_high_level_adapter, list_high_level_presets
from .stack import JoseStack, build_stack

__all__ = [
    "EmbodimentSpec",
    "GripperSpec",
    "load_embodiment",
    "list_embodiments",
    "JointReading",
    "StateEstimate",
    "WholeBodyCommand",
    "FeatureBuilder",
    "CANONICAL_SLOTS",
    "SensorChain",
    "JoseRuntime",
    "make_port",
    "list_port_presets",
    "make_high_level_adapter",
    "list_high_level_presets",
    "JoseStack",
    "build_stack",
]
