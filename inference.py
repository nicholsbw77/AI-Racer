"""
agent/inference.py

Loads a trained DrivingPolicyNet checkpoint and runs inference.
Includes EMA output smoothing for jitter-free control at 360Hz.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from model import DrivingPolicyNet

logger = logging.getLogger(__name__)


class DrivingAgent:
    """
    Wraps a trained DrivingPolicyNet for live inference.

    Features:
      - Loads checkpoint by track/car combo name
      - EMA smoothing on all outputs
      - Output clamping with configurable limits
      - Fallback to safe state if model not loaded
    """

    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.inf_cfg = cfg.get("inference", {})

        # Force CPU for inference — the model is a small MLP where GPU transfer
        # overhead far exceeds compute savings.  Running on CUDA at 360Hz
        # hammers the driver's memory allocator and can trigger TDR crashes,
        # especially with Studio drivers.
        self.device = torch.device("cpu")

        self.model: Optional[DrivingPolicyNet] = None
        self.current_combo: Optional[str] = None

        # Normalization constants (loaded from checkpoint)
        self.norm: dict = {}

        # Pre-allocated input tensor buffer (resized on checkpoint load)
        self._input_buf: Optional[torch.Tensor] = None

        # EMA state
        alpha = self.inf_cfg.get("ema_alpha", 0.20)
        self.ema_alpha = alpha
        self.ema_throttle = 0.0
        self.ema_brake = 0.0
        self.ema_steering = 0.0

        # Output limits
        self.throttle_min = self.inf_cfg.get("throttle_min", 0.0)
        self.throttle_max = self.inf_cfg.get("throttle_max", 1.0)
        self.brake_min = self.inf_cfg.get("brake_min", 0.0)
        self.brake_max = self.inf_cfg.get("brake_max", 1.0)
        self.steering_min = self.inf_cfg.get("steering_min", -1.0)
        self.steering_max = self.inf_cfg.get("steering_max", 1.0)

        self.min_speed_cutoff = self.inf_cfg.get("min_speed_cutoff", 2.0)

        logger.info(f"DrivingAgent initialized on CPU (EMA α={alpha})")

    def load_checkpoint(self, combo_name: str) -> bool:
        """
        Load model checkpoint for the given track/car combo.
        Returns True if successful.
        """
        checkpoint_dir = Path(self.cfg["paths"]["checkpoints"]) / combo_name
        best_path = checkpoint_dir / "best.pt"

        if not best_path.exists():
            logger.warning(f"No checkpoint found for {combo_name} at {best_path}")
            return False

        try:
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            model_cfg = ckpt["cfg"]["model"]

            input_dim = ckpt["input_dim"]
            self.model = DrivingPolicyNet(
                input_dim=input_dim,
                hidden_dims=tuple(model_cfg["hidden_dims"]),
                dropout=0.0,  # Disable dropout at inference time
            )
            # Model stays on CPU — no .to(device) needed

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            # Pre-allocate a reusable input tensor to avoid creating a new
            # tensor (and triggering memory allocation) every frame at 360Hz
            self._input_buf = torch.empty(input_dim, dtype=torch.float32)

            self.current_combo = combo_name
            self.norm = ckpt.get("norm", {})
            self.input_dim = ckpt["input_dim"]
            # Derive sequence_history from checkpoint dimensions
            # input_dim = n_state_features + sequence_history * n_history_actions
            n_history_actions = 4  # throttle, brake, steering, steering_delta
            ckpt_cfg = ckpt.get("cfg", {})
            stored_seq = ckpt_cfg.get("training", {}).get("sequence_history")
            if isinstance(stored_seq, int) and stored_seq > 0:
                self.sequence_history = stored_seq
                self.n_state_features = ckpt["input_dim"] - stored_seq * n_history_actions
            else:
                # Fallback: assume 15 frames of history
                self.sequence_history = 15
                self.n_state_features = ckpt["input_dim"] - 15 * n_history_actions
            logger.info(
                f"Model input_dim={ckpt['input_dim']} "
                f"(state={self.n_state_features}, history={self.sequence_history}x{n_history_actions})"
            )
            self._reset_ema()

            logger.info(
                f"Loaded model for '{combo_name}' "
                f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})"
            )
            if self.norm:
                logger.info(
                    f"Norm constants: speed_max={self.norm.get('speed_max_ms')} "
                    f"steer_lock={self.norm.get('steering_lock_radians')} "
                    f"rpm_max={self.norm.get('rpm_max')}"
                )
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint {best_path}: {e}")
            return False

    def predict(
        self,
        state_vector: np.ndarray,
        car_speed_ms: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Run one inference step.

        Args:
            state_vector: numpy array of shape (input_dim,)
            car_speed_ms: current car speed in m/s for safety cutoff

        Returns:
            (throttle, brake, steering) all smoothed and clamped
        """
        # Safety cutoff: if car is nearly stopped, don't send inputs
        if car_speed_ms < self.min_speed_cutoff and car_speed_ms >= 0:
            # Allow gentle throttle to get moving, but no brake/steer
            return 0.3, 0.0, self.ema_steering

        if self.model is None:
            logger.warning("No model loaded - returning safe state")
            return 0.0, 0.0, 0.0

        # Inference — copy into pre-allocated buffer (zero-alloc hot path)
        self._input_buf.copy_(torch.from_numpy(state_vector))
        raw_throttle, raw_brake, raw_steering = self.model.predict(self._input_buf)

        # EMA smoothing
        throttle = self._ema(self.ema_throttle, raw_throttle)
        brake = self._ema(self.ema_brake, raw_brake)
        steering = self._ema(self.ema_steering, raw_steering)

        self.ema_throttle = throttle
        self.ema_brake = brake
        self.ema_steering = steering

        # Clamp to configured limits
        throttle = np.clip(throttle, self.throttle_min, self.throttle_max)
        brake = np.clip(brake, self.brake_min, self.brake_max)
        steering = np.clip(steering, self.steering_min, self.steering_max)

        # Mutual exclusion: if braking hard, reduce throttle
        # (model usually learns this, but enforce it explicitly as safety)
        if brake > 0.3:
            throttle = min(throttle, 1.0 - brake)

        return float(throttle), float(brake), float(steering)

    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average."""
        return self.ema_alpha * new + (1.0 - self.ema_alpha) * prev

    def _reset_ema(self):
        """Reset EMA state on model/session change."""
        self.ema_throttle = 0.0
        self.ema_brake = 0.0
        self.ema_steering = 0.0

    def reset(self):
        """Reset agent state (call on session start/end)."""
        self._reset_ema()

    @property
    def is_ready(self) -> bool:
        return self.model is not None
