"""One object per (robot, low-level controller, high-level policy) triple.

    joints ──► JoseRuntime ──► StateEstimate ──┬──► low-level port (replaces the IMU)
                                               └──► high-level state (VLA proprio)
    VLA action chunk ──► HighLevelAdapter ──► WholeBodyCommand ──► low-level controller

Swapping the VLA changes only the high-level YAML. Swapping the robot changes the
embodiment YAML, the port preset and the estimator weights. Nothing else moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .embodiment import EmbodimentSpec, load_embodiment
from .highlevel import HighLevelAdapter, make_high_level_adapter
from .ports import make_port
from .runtime import JoseRuntime
from .signals import JointReading, StateEstimate, WholeBodyCommand


@dataclass
class JoseStack:
    spec: EmbodimentSpec
    runtime: JoseRuntime
    port: object
    high_level: HighLevelAdapter | None = None
    last_estimate: StateEstimate | None = None

    def reset(self, mask=None, yaw: float = 0.0) -> None:
        self.runtime.reset(mask, yaw)
        self.port.reset(mask)
        self.last_estimate = None

    def tick(self, reading: JointReading, low_level_input):
        """Run the estimator once and write it into the controller's input.

        ``low_level_input`` is the flat observation for an ``ObservationPort`` or
        the low-state message for an ``ImuMessagePort``.
        """
        self.last_estimate = self.runtime.step(reading)
        return self.port.write(low_level_input, self.last_estimate, reading)

    def high_level_state(self, reading: JointReading):
        if self.high_level is None:
            raise RuntimeError("No high-level adapter configured")
        return self.high_level.encode_state(reading, self.last_estimate)

    def command(self, raw_action, reading: JointReading | None = None) -> list[WholeBodyCommand]:
        if self.high_level is None:
            raise RuntimeError("No high-level adapter configured")
        return self.high_level.decode_action(raw_action, reading)


def build_stack(
    embodiment: str | Path | EmbodimentSpec,
    port: str | dict,
    runtime_factory,
    high_level: str | dict | None = None,
) -> JoseStack:
    """``runtime_factory(spec) -> JoseRuntime`` keeps weight loading out of this module."""
    spec = embodiment if isinstance(embodiment, EmbodimentSpec) else load_embodiment(embodiment)
    runtime = runtime_factory(spec)
    hl = make_high_level_adapter(spec, high_level) if high_level is not None else None
    return JoseStack(spec, runtime, make_port(spec, port), hl)
