"""
agent/safety_controller.py

Safety layer between AI model outputs and the vJoy controller.
Monitors telemetry signals (track position, lateral G, speed) and blends
corrective inputs with the model's predictions to keep the car on track.

Correction modes (applied in priority order):
  1. Off-track recovery   -- hard brake, steer toward center
  2. Edge proximity       -- proportional throttle cut and corrective steering
  3. Excessive lateral G  -- throttle reduction and gentle braking
  4. Speed limiting       -- per-section speed cap

When the model keeps the car centered and within limits, the safety
controller produces zero intervention (blend factor = 0).
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from telemetry import CarState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_SAFETY_CONFIG: Dict = {
    "edge_warning_threshold": 0.75,     # |track_pos| above this triggers warning
    "edge_danger_threshold": 0.90,      # |track_pos| above this triggers braking
    "max_lateral_g": 35.0,              # m/s^2 (~3.5 g)
    "recovery_steer_gain": 0.5,         # how aggressively to steer back
    "recovery_brake": 0.8,              # brake force during off-track recovery
    "max_blend_factor": 0.8,            # never fully override model
    "off_track_frames_for_recovery": 3, # consecutive frames off-track before full recovery
    "speed_limit_ms": None,             # optional global speed cap (m/s), None = disabled
    "speed_profile": None,              # dict mapping lap_dist_pct ranges to max speed
    "enabled": True,                    # master enable switch
}


# ---------------------------------------------------------------------------
# Intervention statistics
# ---------------------------------------------------------------------------

@dataclass
class InterventionStats:
    """Cumulative counters for safety interventions."""

    total_frames: int = 0
    off_track_frames: int = 0
    edge_warning_frames: int = 0
    edge_danger_frames: int = 0
    lat_g_frames: int = 0
    speed_limit_frames: int = 0
    recovery_mode_activations: int = 0
    last_log_time: float = 0.0

    def any_intervention(self) -> bool:
        return (
            self.off_track_frames
            + self.edge_warning_frames
            + self.edge_danger_frames
            + self.lat_g_frames
            + self.speed_limit_frames
        ) > 0


# ---------------------------------------------------------------------------
# SafetyController
# ---------------------------------------------------------------------------

class SafetyController:
    """
    Wraps around model outputs and applies safety corrections before the
    values are sent to the vJoy controller.

    Usage::

        safety = SafetyController(config)
        throttle, brake, steering = safety.apply(
            model_throttle, model_brake, model_steering, car_state
        )
        controller.set_inputs(throttle, brake, steering)

    Args:
        config: Dict of tunable parameters (see DEFAULT_SAFETY_CONFIG).
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.cfg: Dict = {**DEFAULT_SAFETY_CONFIG}
        if config is not None:
            self.cfg.update(config)

        # State tracking -------------------------------------------------
        self._consecutive_off_track: int = 0
        self._consecutive_edge: int = 0
        self._in_recovery: bool = False

        # Statistics ------------------------------------------------------
        self.stats = InterventionStats(last_log_time=time.monotonic())

        # Logging interval (seconds)
        self._log_interval: float = 10.0

        logger.info("SafetyController initialised (enabled=%s)", self.cfg["enabled"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        model_throttle: float,
        model_brake: float,
        model_steering: float,
        car_state: CarState,
    ) -> Tuple[float, float, float]:
        """Apply safety corrections to model outputs.

        Args:
            model_throttle: Model-predicted throttle (0-1).
            model_brake:    Model-predicted brake (0-1).
            model_steering: Model-predicted steering (-1 to +1).
            car_state:      Current ``CarState`` from telemetry.

        Returns:
            Tuple of (throttle, brake, steering) after safety blending.
        """
        self.stats.total_frames += 1

        if not self.cfg["enabled"]:
            return model_throttle, model_brake, model_steering

        # Accumulate corrections from each mode.  Each mode produces a
        # (safety_throttle, safety_brake, safety_steering, blend) tuple.
        # We apply the *highest-priority* mode that fires, since they are
        # checked in priority order and earlier modes dominate.

        blend: float = 0.0
        safety_throttle: float = model_throttle
        safety_brake: float = model_brake
        safety_steering: float = model_steering

        track_pos = car_state.track_pos
        abs_track_pos = abs(track_pos)

        # --- Priority 1: off-track recovery ---
        if not car_state.is_on_track:
            self._consecutive_off_track += 1
            self.stats.off_track_frames += 1

            if self._consecutive_off_track >= self.cfg["off_track_frames_for_recovery"]:
                if not self._in_recovery:
                    self._in_recovery = True
                    self.stats.recovery_mode_activations += 1
                    logger.warning(
                        "RECOVERY MODE activated (off-track for %d frames, "
                        "track_pos=%.3f)",
                        self._consecutive_off_track,
                        track_pos,
                    )

            # Blend increases with consecutive off-track frames,
            # reaching max_blend_factor at off_track_frames_for_recovery.
            threshold = max(1, self.cfg["off_track_frames_for_recovery"])
            blend = min(
                self.cfg["max_blend_factor"],
                self._consecutive_off_track / threshold,
            )
            # Steer toward center: if track_pos > 0 the car is right of
            # center, so steer left (negative).
            center_steer = -_sign(track_pos) * self.cfg["recovery_steer_gain"]
            safety_throttle = 0.0
            safety_brake = self.cfg["recovery_brake"]
            safety_steering = center_steer

        else:
            # Back on track -- wind down recovery state.
            self._consecutive_off_track = 0
            if self._in_recovery:
                self._in_recovery = False
                logger.info("Recovery mode ended -- car back on track")

            # --- Priority 2: edge proximity ---
            if abs_track_pos > self.cfg["edge_warning_threshold"]:
                self.stats.edge_warning_frames += 1

                # Blend proportional to proximity beyond the warning line.
                edge_range = 1.0 - self.cfg["edge_warning_threshold"]
                if edge_range > 0:
                    proximity = (abs_track_pos - self.cfg["edge_warning_threshold"]) / edge_range
                else:
                    proximity = 1.0
                proximity = min(proximity, 1.0)

                blend = proximity * self.cfg["max_blend_factor"]

                # Corrective steering toward center.
                center_steer = -_sign(track_pos) * self.cfg["recovery_steer_gain"] * proximity
                safety_steering = center_steer

                # Throttle reduction proportional to proximity.
                safety_throttle = model_throttle * (1.0 - proximity)

                # Light braking if in the danger zone.
                if abs_track_pos > self.cfg["edge_danger_threshold"]:
                    self.stats.edge_danger_frames += 1
                    danger_frac = (
                        (abs_track_pos - self.cfg["edge_danger_threshold"])
                        / (1.0 - self.cfg["edge_danger_threshold"])
                    )
                    danger_frac = min(danger_frac, 1.0)
                    safety_brake = max(model_brake, 0.3 + 0.5 * danger_frac)
                else:
                    safety_brake = model_brake

            # --- Priority 3: excessive lateral G ---
            if abs(car_state.lat_g) > self.cfg["max_lateral_g"]:
                self.stats.lat_g_frames += 1

                excess = (abs(car_state.lat_g) - self.cfg["max_lateral_g"]) / self.cfg["max_lateral_g"]
                excess = min(excess, 1.0)
                lat_blend = excess * self.cfg["max_blend_factor"]

                # Only raise the overall blend -- don't lower it.
                if lat_blend > blend:
                    blend = lat_blend

                safety_throttle = min(safety_throttle, model_throttle * (1.0 - excess))
                safety_brake = max(safety_brake, 0.15 + 0.25 * excess)
                # Don't fight the steering -- keep model steering.
                safety_steering = model_steering

            # --- Priority 4: speed limiting ---
            speed_limit = self._get_speed_limit(car_state)
            if speed_limit is not None and car_state.speed > speed_limit:
                self.stats.speed_limit_frames += 1

                overspeed = (car_state.speed - speed_limit) / max(speed_limit, 1.0)
                overspeed = min(overspeed, 1.0)
                spd_blend = overspeed * self.cfg["max_blend_factor"]

                if spd_blend > blend:
                    blend = spd_blend

                safety_throttle = min(safety_throttle, model_throttle * (1.0 - overspeed))
                safety_brake = max(safety_brake, 0.2 + 0.6 * overspeed)

        # --- Blend model and safety outputs ---
        out_throttle = _blend(model_throttle, safety_throttle, blend)
        out_brake = _blend(model_brake, safety_brake, blend)
        out_steering = _blend(model_steering, safety_steering, blend)

        # Clamp to valid ranges.
        out_throttle = _clamp(out_throttle, 0.0, 1.0)
        out_brake = _clamp(out_brake, 0.0, 1.0)
        out_steering = _clamp(out_steering, -1.0, 1.0)

        # Edge tracking (for stats only -- the correction already happened).
        if abs_track_pos > self.cfg["edge_warning_threshold"]:
            self._consecutive_edge += 1
        else:
            self._consecutive_edge = 0

        # Periodic logging.
        self._maybe_log_stats()

        return out_throttle, out_brake, out_steering

    def reset(self) -> None:
        """Reset internal state (call on session or lap change)."""
        self._consecutive_off_track = 0
        self._consecutive_edge = 0
        self._in_recovery = False
        self.stats = InterventionStats(last_log_time=time.monotonic())
        logger.info("SafetyController state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_speed_limit(self, car_state: CarState) -> Optional[float]:
        """Return the speed limit (m/s) for the car's current track position.

        Checks the per-section ``speed_profile`` first, then falls back to
        the global ``speed_limit_ms``.
        """
        profile = self.cfg.get("speed_profile")
        if profile is not None:
            lap_pct = car_state.lap_dist_pct
            for (start, end), limit in profile.items():
                if start <= lap_pct < end:
                    return float(limit)

        if self.cfg["speed_limit_ms"] is not None:
            return float(self.cfg["speed_limit_ms"])

        return None

    def _maybe_log_stats(self) -> None:
        """Log intervention statistics every ``_log_interval`` seconds."""
        now = time.monotonic()
        if now - self.stats.last_log_time < self._log_interval:
            return

        s = self.stats
        if s.total_frames == 0:
            return

        pct = lambda n: 100.0 * n / s.total_frames

        logger.info(
            "Safety stats (%d frames): off_track=%.1f%% edge_warn=%.1f%% "
            "edge_danger=%.1f%% lat_g=%.1f%% speed_lim=%.1f%% "
            "recoveries=%d",
            s.total_frames,
            pct(s.off_track_frames),
            pct(s.edge_warning_frames),
            pct(s.edge_danger_frames),
            pct(s.lat_g_frames),
            pct(s.speed_limit_frames),
            s.recovery_mode_activations,
        )
        self.stats.last_log_time = now


# ---------------------------------------------------------------------------
# Pure utility functions
# ---------------------------------------------------------------------------

def _blend(model_val: float, safety_val: float, factor: float) -> float:
    """Linearly blend between model and safety values.

    ``factor`` = 0  =>  pure model output
    ``factor`` = 1  =>  pure safety output
    """
    return (1.0 - factor) * model_val + factor * safety_val


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _sign(x: float) -> float:
    """Return -1, 0, or +1."""
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Simulate several scenarios and print the corrections applied."""

    print("=" * 72)
    print("SafetyController self-test")
    print("=" * 72)

    safety = SafetyController()

    scenarios: List[Tuple[str, float, float, float, CarState]] = [
        # (name, model_thr, model_brk, model_str, car_state)
        (
            "Centered, normal driving",
            0.80, 0.00, 0.05,
            CarState(
                speed=40.0, throttle=0.8, brake=0.0, steering=0.05,
                track_pos=0.1, is_on_track=True, lat_g=5.0,
                lap_dist_pct=0.25, session_active=True,
            ),
        ),
        (
            "Near edge (track_pos=0.80)",
            0.90, 0.00, 0.10,
            CarState(
                speed=55.0, throttle=0.9, brake=0.0, steering=0.10,
                track_pos=0.80, is_on_track=True, lat_g=12.0,
                lap_dist_pct=0.40, session_active=True,
            ),
        ),
        (
            "Danger zone (track_pos=0.95)",
            0.70, 0.00, 0.15,
            CarState(
                speed=50.0, throttle=0.7, brake=0.0, steering=0.15,
                track_pos=0.95, is_on_track=True, lat_g=15.0,
                lap_dist_pct=0.50, session_active=True,
            ),
        ),
        (
            "Off-track (left side)",
            0.60, 0.00, -0.20,
            CarState(
                speed=30.0, throttle=0.6, brake=0.0, steering=-0.20,
                track_pos=-1.1, is_on_track=False, lat_g=8.0,
                lap_dist_pct=0.60, session_active=True,
            ),
        ),
        (
            "Excessive lateral G (45 m/s^2)",
            0.85, 0.00, 0.30,
            CarState(
                speed=60.0, throttle=0.85, brake=0.0, steering=0.30,
                track_pos=0.3, is_on_track=True, lat_g=45.0,
                lap_dist_pct=0.70, session_active=True,
            ),
        ),
    ]

    for name, m_thr, m_brk, m_str, state in scenarios:
        out_thr, out_brk, out_str = safety.apply(m_thr, m_brk, m_str, state)

        thr_delta = out_thr - m_thr
        brk_delta = out_brk - m_brk
        str_delta = out_str - m_str
        intervened = abs(thr_delta) > 1e-6 or abs(brk_delta) > 1e-6 or abs(str_delta) > 1e-6

        print(f"\n--- {name} ---")
        print(f"  Model  -> thr={m_thr:.3f}  brk={m_brk:.3f}  str={m_str:+.3f}")
        print(f"  Output -> thr={out_thr:.3f}  brk={out_brk:.3f}  str={out_str:+.3f}")
        if intervened:
            print(f"  Delta  -> thr={thr_delta:+.3f}  brk={brk_delta:+.3f}  str={str_delta:+.3f}")
        else:
            print("  (no intervention)")

    # Test sustained off-track triggering full recovery mode.
    print(f"\n--- Sustained off-track (5 frames) ---")
    safety.reset()
    off_track_state = CarState(
        speed=25.0, throttle=0.5, brake=0.0, steering=0.0,
        track_pos=1.2, is_on_track=False, lat_g=3.0,
        lap_dist_pct=0.30, session_active=True,
    )
    for frame in range(1, 6):
        out_thr, out_brk, out_str = safety.apply(0.5, 0.0, 0.0, off_track_state)
        print(
            f"  Frame {frame}: thr={out_thr:.3f}  brk={out_brk:.3f}  "
            f"str={out_str:+.3f}  recovery={safety._in_recovery}"
        )

    # Test speed limiter.
    print(f"\n--- Speed limiter (limit=50 m/s, speed=65 m/s) ---")
    safety_speed = SafetyController({"speed_limit_ms": 50.0})
    fast_state = CarState(
        speed=65.0, throttle=1.0, brake=0.0, steering=0.0,
        track_pos=0.0, is_on_track=True, lat_g=2.0,
        lap_dist_pct=0.10, session_active=True,
    )
    out_thr, out_brk, out_str = safety_speed.apply(1.0, 0.0, 0.0, fast_state)
    print(f"  Model  -> thr=1.000  brk=0.000  str=+0.000")
    print(f"  Output -> thr={out_thr:.3f}  brk={out_brk:.3f}  str={out_str:+.3f}")

    # Test disabled.
    print(f"\n--- Disabled safety controller ---")
    safety_off = SafetyController({"enabled": False})
    out_thr, out_brk, out_str = safety_off.apply(1.0, 0.0, 0.5, off_track_state)
    print(f"  Model  -> thr=1.000  brk=0.000  str=+0.500")
    print(f"  Output -> thr={out_thr:.3f}  brk={out_brk:.3f}  str={out_str:+.3f}")
    print("  (should be unchanged)")

    print(f"\n{'=' * 72}")
    print("Self-test complete.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )
    _run_self_test()
