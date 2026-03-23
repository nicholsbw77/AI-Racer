"""
agent/inference.py

Loads a trained DrivingPolicyNet checkpoint and runs inference.
Includes EMA output smoothing for jitter-free control at 360Hz.

Enhanced with:
  - Track-aware inference using TrackMap features
  - Safety controller integration
  - Adaptive EMA based on segment type (tighter in corners)
  - Confidence estimation placeholder
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from model import DrivingPolicyNet
from track_map import TrackMap, SegmentType

logger = logging.getLogger(__name__)


class DrivingAgent:
    """
    Wraps a trained DrivingPolicyNet for live inference.

    Features:
      - Loads checkpoint by track/car combo name
      - EMA smoothing on all outputs (adaptive by context)
      - Output clamping with configurable limits
      - Track-aware inference with segment context
      - Fallback to safe state if model not loaded
    """

    def __init__(self, cfg: dict, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.inf_cfg = cfg.get("inference", {})

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model: Optional[DrivingPolicyNet] = None
        self.current_combo: Optional[str] = None

        # EMA state
        alpha = self.inf_cfg.get("ema_alpha", 0.20)
        self.ema_alpha_base = alpha
        self.ema_alpha = alpha
        self.ema_throttle = 0.0
        self.ema_brake = 0.0
        self.ema_steering = 0.0

        # Adaptive EMA settings
        self.ema_alpha_corner = self.inf_cfg.get("ema_alpha_corner", 0.40)
        self.ema_alpha_braking = self.inf_cfg.get("ema_alpha_braking", 0.50)
        self.ema_alpha_straight = self.inf_cfg.get("ema_alpha_straight", 0.20)

        # Output limits
        self.throttle_min = self.inf_cfg.get("throttle_min", 0.0)
        self.throttle_max = self.inf_cfg.get("throttle_max", 1.0)
        self.brake_min = self.inf_cfg.get("brake_min", 0.0)
        self.brake_max = self.inf_cfg.get("brake_max", 1.0)
        self.steering_min = self.inf_cfg.get("steering_min", -1.0)
        self.steering_max = self.inf_cfg.get("steering_max", 1.0)

        self.min_speed_cutoff = self.inf_cfg.get("min_speed_cutoff", 2.0)

        # Track map for context-aware inference
        self.track_map: Optional[TrackMap] = None

        logger.info(f"DrivingAgent initialized on {device} (EMA alpha={alpha})")

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
            ckpt = torch.load(best_path, map_location=self.device)
            model_cfg = ckpt["cfg"]["model"]

            self.model = DrivingPolicyNet(
                input_dim=ckpt["input_dim"],
                hidden_dims=tuple(model_cfg["hidden_dims"]),
                dropout=0.0,  # Disable dropout at inference time
            ).to(self.device)

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

            self.current_combo = combo_name
            self._reset_ema()

            logger.info(
                f"Loaded model for '{combo_name}' "
                f"(epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint {best_path}: {e}")
            return False

    def load_track_map(self, combo_name: str) -> bool:
        """Load track map for context-aware inference."""
        map_path = Path(self.cfg["paths"]["checkpoints"]) / combo_name / "track_map.yaml"
        if not map_path.exists():
            # Also check processed data folder
            map_path = Path(self.cfg["paths"]["processed_data"]) / combo_name / "track_map.yaml"

        if map_path.exists():
            self.track_map = TrackMap()
            if self.track_map.load(str(map_path)):
                logger.info(f"Track map loaded for '{combo_name}'")
                return True
            self.track_map = None

        logger.info(f"No track map found for '{combo_name}' — running without track context")
        return False

    def predict(
        self,
        state_vector: np.ndarray,
        car_speed_ms: float = 0.0,
        lap_dist_pct: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Run one inference step.

        Args:
            state_vector: numpy array of shape (input_dim,)
            car_speed_ms: current car speed in m/s for safety cutoff
            lap_dist_pct: current track position for adaptive EMA

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

        # Adapt EMA alpha based on current segment type
        self._adapt_ema(lap_dist_pct)

        # Inference
        x = torch.from_numpy(state_vector).to(self.device)
        raw_throttle, raw_brake, raw_steering = self.model.predict(x)

        # EMA smoothing (with adaptive alpha)
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
        if brake > 0.3:
            throttle = min(throttle, 1.0 - brake)

        return float(throttle), float(brake), float(steering)

    def _adapt_ema(self, lap_dist_pct: float):
        """Adapt EMA smoothing based on current track segment."""
        if self.track_map is None or not self.track_map.is_built:
            self.ema_alpha = self.ema_alpha_base
            return

        seg = self.track_map.get_segment(lap_dist_pct)
        if seg is None:
            self.ema_alpha = self.ema_alpha_base
            return

        if seg.segment_type == SegmentType.BRAKING_ZONE:
            # Braking zones need fast response — higher alpha
            self.ema_alpha = self.ema_alpha_braking
        elif seg.is_cornering:
            # Corners need responsive steering — higher alpha
            self.ema_alpha = self.ema_alpha_corner
        elif seg.segment_type in (SegmentType.STRAIGHT, SegmentType.ACCELERATION_ZONE):
            # Straights can be smoother — lower alpha
            self.ema_alpha = self.ema_alpha_straight
        else:
            self.ema_alpha = self.ema_alpha_base

    def get_track_features(
        self,
        lap_dist_pct: float,
        current_speed: float = 0.0,
    ) -> Optional[np.ndarray]:
        """Get track-aware features for the current position, if available."""
        if self.track_map is None or not self.track_map.is_built:
            return None

        lookahead = self.cfg.get("track", {}).get("lookahead_segments", 5)
        return self.track_map.get_track_features(
            lap_dist_pct, current_speed, lookahead
        )

    def get_boundary_proximity(
        self, lap_dist_pct: float, track_pos: float = 0.0
    ) -> float:
        """Get proximity to track boundary (0 = center, 1 = edge)."""
        if self.track_map is None:
            return 0.0
        return self.track_map.get_boundary_proximity(lap_dist_pct, track_pos)

    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average."""
        return self.ema_alpha * new + (1.0 - self.ema_alpha) * prev

    def _reset_ema(self):
        """Reset EMA state on model/session change."""
        self.ema_throttle = 0.0
        self.ema_brake = 0.0
        self.ema_steering = 0.0
        self.ema_alpha = self.ema_alpha_base

    def reset(self):
        """Reset agent state (call on session start/end)."""
        self._reset_ema()

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @property
    def has_track_map(self) -> bool:
        return self.track_map is not None and self.track_map.is_built
