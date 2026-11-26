"""Deployment runtime. Joint readings in, a parsed ``StateEstimate`` out."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Sequence

import torch

from .features import FeatureBuilder
from .frames import euler_yaw_rate
from .signals import JointReading, StateEstimate

_AXES = ("x", "y", "z")
_VECTOR_FIELDS = ("projected_gravity", "base_ang_vel", "base_lin_vel")


def _parse_names(names: Sequence[str]) -> dict:
    """Locate each known field in the model output by name."""
    names = list(names)
    index = {name: i for i, name in enumerate(names)}
    layout: dict = {"vectors": {}, "contact": {}, "payload": None}
    for field in _VECTOR_FIELDS:
        cols = [f"{field}_{a}" for a in _AXES]
        if all(c in index for c in cols):
            layout["vectors"][field] = [index[c] for c in cols]
    for name, i in index.items():
        if name.endswith("_contact_force") or name.endswith("_contact"):
            layout["contact"][name.replace("_contact_force", "").replace("_contact", "")] = i
    if "payload_mass" in index:
        layout["payload"] = index["payload_mass"]
    for required in ("projected_gravity", "base_ang_vel"):
        if required not in layout["vectors"]:
            raise ValueError(
                f"Estimator outputs {names} lack {required}. The IMU ports need projected gravity "
                "and base angular velocity."
            )
    return layout


class JoseRuntime:
    """Holds the history window and turns each reading into a ``StateEstimate``.

    ``model`` is any callable mapping a (B, window, input_dim) history to
    (B, n_outputs) in physical units. For a JOSE ``NormalizedEstimator`` pass its
    ``predict`` method, or use ``from_checkpoint``.

    The window starts at zeros after ``reset``, which is how
    ``estimator/pipeline.py:HistoryBuffer`` fills it in training. A different
    fill at deployment would show the model inputs it never saw.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        features: FeatureBuilder,
        output_names: Sequence[str],
        window: int,
        batch: int = 1,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.features = features
        self.output_names = list(output_names)
        self.layout = _parse_names(self.output_names)
        self.window = int(window)
        self.device = torch.device(device)
        self.dt = features.spec.control_dt
        self._history = torch.zeros((batch, self.window, features.input_dim), device=self.device)
        self._yaw = torch.zeros(batch, device=self.device)

    def reset(self, mask: torch.Tensor | None = None, yaw: torch.Tensor | float = 0.0) -> None:
        if mask is None:
            self._history.zero_()
            self._yaw[:] = yaw
        else:
            self._history[mask] = 0.0
            self._yaw[mask] = yaw if isinstance(yaw, float) else yaw[mask]

    @torch.no_grad()
    def step(self, reading: JointReading) -> StateEstimate:
        frame = self.features(reading).to(self.device, self._history.dtype)
        if frame.shape[0] != self._history.shape[0]:
            self._history = torch.zeros(
                (frame.shape[0], self.window, frame.shape[-1]), device=self.device, dtype=frame.dtype
            )
            self._yaw = torch.zeros(frame.shape[0], device=self.device)
        self._history = torch.roll(self._history, -1, dims=1)
        self._history[:, -1] = frame
        out = self.model(self._history)
        if out.shape[-1] != len(self.output_names):
            raise RuntimeError(f"Model returned {out.shape[-1]} outputs, names list has {len(self.output_names)}")
        vectors = {k: out[:, ids] for k, ids in self.layout["vectors"].items()}
        gravity = torch.nn.functional.normalize(vectors["projected_gravity"], dim=-1)
        ang_vel = vectors["base_ang_vel"]
        # Yaw is unobservable from gravity loading. It is integrated from the
        # estimated rate and drifts, which the paper must report per episode length.
        self._yaw = self._yaw + self.dt * euler_yaw_rate(gravity, ang_vel)
        payload = self.layout["payload"]
        return StateEstimate(
            projected_gravity=gravity,
            base_ang_vel=ang_vel,
            base_lin_vel=vectors.get("base_lin_vel"),
            yaw=self._yaw.clone(),
            contact={name: out[:, i] for name, i in self.layout["contact"].items()},
            payload=out[:, payload] if payload is not None else None,
            raw=out,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, features: FeatureBuilder, device: str = "cpu", batch: int = 1):
        """Load a checkpoint written by ``estimator/pipeline.py:save_jose_checkpoint``.

        Only ``estimator/models.py`` is imported, by file path, so this works
        without Isaac Lab or skrl on the robot computer.
        """
        models_path = Path(__file__).resolve().parent.parent / "estimator" / "models.py"
        spec = importlib.util.spec_from_file_location("_jose_estimator_models", models_path)
        models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("kind") != "estimator":
            raise ValueError("Checkpoint is not a JOSE state estimator")
        if payload.get("imu_input"):
            raise ValueError("Checkpoint reads the IMU. Modular JOSE deploys IMU-free estimators only")
        if bool(payload.get("torque_input")) != features.use_torque:
            raise ValueError("Checkpoint torque input does not match the feature builder")
        if payload.get("action_input"):
            raise ValueError("Action-input checkpoints are an ablation arm, not a deployable estimator")
        config = payload["model_config"]
        if config["input_dim"] != features.input_dim:
            raise ValueError(f"Checkpoint input {config['input_dim']} != features {features.input_dim}")
        model = models.build_estimator(
            config["type"], config["input_dim"], config["output_dim"],
            config.get("hidden_size", 256), config.get("num_layers", 2),
            tuple(config.get("channels", (64, 128, 128))), config.get("window", payload.get("window", 50)),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        names = payload["observation_schema"]["estimator_target_names"]
        return cls(model.predict, features, names, payload["window"], batch=batch, device=device)
