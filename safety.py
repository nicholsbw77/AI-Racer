"""
safety.py

Comprehensive safety controller for the iRacing AI bot.

Multiple independent safety layers that can each independently kill or
attenuate control outputs. Designed to fail-safe: any ambiguity → release inputs.

Safety layers (checked in order, any can override):
  1. Heartbeat watchdog     — kill if no fresh telemetry for N ms
  2. Session state gate     — kill if session inactive / not on track
  3. Speed envelope         — attenuate at low speed, kill at zero
  4. G-force limiter        — reduce inputs if lateral/longitudinal G exceeds limits
  5. Steering rate limiter  — cap steering change per tick to prevent snap oversteer
  6. Brake/throttle mutex   — prevent simultaneous hard brake + throttle
  7. Track boundary monitor — reduce speed / correct steering if near track edge
  8. Incident detector      — detect spins, collisions, stuck states
  9. Lap time watchdog      — kill if lap time exceeds N× personal best
 10. Pit lane governor      — enforce pit speed limit

All safety decisions are logged for post-session analysis.
"""

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)


class SafetyAction(Enum):
    """What the safety system decided to do."""
    PASS = auto()           # No intervention, pass through
    ATTENUATE = auto()      # Reduce outputs (partial intervention)
    OVERRIDE = auto()       # Replace outputs entirely
    KILL = auto()           # Release all inputs (full kill)


@dataclass
class SafetyState:
    """Tracks the internal state of the safety controller across ticks."""
    # Heartbeat
    last_telemetry_time: float = 0.0
    heartbeat_failures: int = 0

    # Speed tracking
    zero_speed_frames: int = 0
    low_speed_frames: int = 0

    # Spin detection
    prev_heading_rate: float = 0.0
    spin_frames: int = 0
    spin_detected: bool = False
    spin_recovery_frames: int = 0

    # Stuck detection
    stuck_frames: int = 0
    stuck_detected: bool = False

    # Lap time watchdog
    lap_start_time: float = 0.0
    current_lap_start_pct: float = 0.0
    last_lap_dist_pct: float = 0.0
    personal_best_s: float = 0.0
    lap_watchdog_triggered: bool = False

    # G-force tracking
    peak_lat_g: float = 0.0
    peak_lon_g: float = 0.0
    g_limit_active: bool = False

    # Steering rate tracking
    prev_steering_output: float = 0.0

    # Track boundary
    consecutive_off_track: int = 0
    track_boundary_active: bool = False

    # Incident counter
    incident_count: int = 0
    last_incident_time: float = 0.0

    # Overall
    total_interventions: int = 0
    total_kills: int = 0
    frames_since_last_kill: int = 0


@dataclass
class SafetyVerdict:
    """Result of a safety check — what to do with the control outputs."""
    action: SafetyAction = SafetyAction.PASS
    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    reason: str = ""
    layer: str = ""  # which safety layer triggered


class SafetyConfig:
    """Safety configuration with sane defaults."""

    def __init__(self, cfg: dict):
        safety_cfg = cfg.get("safety", {})

        # Heartbeat watchdog
        self.heartbeat_timeout_ms: float = safety_cfg.get("heartbeat_timeout_ms", 200.0)
        self.heartbeat_max_failures: int = safety_cfg.get("heartbeat_max_failures", 3)

        # Speed envelope
        self.min_speed_cutoff_ms: float = safety_cfg.get("min_speed_cutoff_ms", 2.0)
        self.crawl_speed_ms: float = safety_cfg.get("crawl_speed_ms", 5.0)
        self.crawl_throttle: float = safety_cfg.get("crawl_throttle", 0.3)
        self.zero_speed_kill_frames: int = safety_cfg.get("zero_speed_kill_frames", 300)

        # G-force limits (m/s^2, ~1g = 9.81 m/s^2)
        self.max_lat_g: float = safety_cfg.get("max_lat_g", 35.0)  # ~3.5g
        self.max_lon_g: float = safety_cfg.get("max_lon_g", 40.0)  # ~4.0g
        self.g_force_throttle_reduction: float = safety_cfg.get("g_force_throttle_reduction", 0.5)

        # Steering rate limiter (max change per tick, normalized -1 to 1)
        self.max_steering_rate: float = safety_cfg.get("max_steering_rate", 0.15)
        self.emergency_steering_rate: float = safety_cfg.get("emergency_steering_rate", 0.25)

        # Brake/throttle mutex
        self.brake_throttle_threshold: float = safety_cfg.get("brake_throttle_threshold", 0.3)

        # Spin detection
        self.spin_yaw_rate_threshold: float = safety_cfg.get("spin_yaw_rate_threshold", 2.0)  # rad/s
        self.spin_detect_frames: int = safety_cfg.get("spin_detect_frames", 10)
        self.spin_recovery_timeout: int = safety_cfg.get("spin_recovery_timeout", 180)  # 3s at 60Hz

        # Stuck detection
        self.stuck_speed_threshold: float = safety_cfg.get("stuck_speed_threshold", 1.0)  # m/s
        self.stuck_detect_frames: int = safety_cfg.get("stuck_detect_frames", 600)  # 10s at 60Hz

        # Lap time watchdog
        self.lap_time_kill_multiplier: float = safety_cfg.get("lap_time_kill_multiplier", 2.5)

        # Track boundary
        self.track_edge_warning: float = safety_cfg.get("track_edge_warning", 0.85)
        self.track_edge_critical: float = safety_cfg.get("track_edge_critical", 0.95)
        self.off_track_kill_frames: int = safety_cfg.get("off_track_kill_frames", 60)
        self.boundary_speed_reduction: float = safety_cfg.get("boundary_speed_reduction", 0.6)

        # Pit lane
        self.pit_speed_limit_ms: float = safety_cfg.get("pit_speed_limit_ms", 18.0)  # ~65 km/h

        # Incident cooldown
        self.incident_cooldown_s: float = safety_cfg.get("incident_cooldown_s", 5.0)
        self.max_incidents_per_session: int = safety_cfg.get("max_incidents_per_session", 10)


class SafetyController:
    """
    Multi-layer safety controller that wraps all model outputs.

    Usage:
        safety = SafetyController(cfg)
        # In control loop:
        verdict = safety.check(state, throttle, brake, steering)
        if verdict.action == SafetyAction.KILL:
            controller.release()
        else:
            controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering)
    """

    def __init__(self, cfg: dict):
        self.config = SafetyConfig(cfg)
        self.state = SafetyState()
        self._log_interval = 60  # frames between periodic safety logs
        self._frame_count = 0

    def check(
        self,
        car_state,  # CarState from telemetry.py
        raw_throttle: float,
        raw_brake: float,
        raw_steering: float,
        track_boundary_pct: float = 0.0,  # 0 = center, 1 = edge
        is_on_track: bool = True,
        on_pit_road: bool = False,
        session_active: bool = True,
    ) -> SafetyVerdict:
        """
        Run all safety checks on proposed control outputs.

        Returns SafetyVerdict with the (possibly modified) outputs
        and the action taken.
        """
        self._frame_count += 1
        self.state.frames_since_last_kill += 1

        # Start with proposed outputs
        throttle = raw_throttle
        brake = raw_brake
        steering = raw_steering

        # ---- Layer 1: Heartbeat watchdog ----
        verdict = self._check_heartbeat(car_state)
        if verdict.action == SafetyAction.KILL:
            return verdict

        # ---- Layer 2: Session state gate ----
        verdict = self._check_session_state(session_active, is_on_track, on_pit_road)
        if verdict.action == SafetyAction.KILL:
            return verdict

        # ---- Layer 3: Incident detector (spin/stuck) ----
        verdict = self._check_incidents(car_state)
        if verdict.action == SafetyAction.KILL:
            return verdict
        if verdict.action == SafetyAction.OVERRIDE:
            return verdict

        # ---- Layer 4: Pit lane governor ----
        if on_pit_road:
            throttle, brake = self._govern_pit_speed(car_state.speed, throttle, brake)

        # ---- Layer 5: Speed envelope ----
        throttle, brake, steering, speed_verdict = self._check_speed_envelope(
            car_state.speed, throttle, brake, steering
        )
        if speed_verdict.action == SafetyAction.KILL:
            return speed_verdict

        # ---- Layer 6: G-force limiter ----
        throttle, steering = self._limit_g_forces(
            car_state.lat_g, car_state.lon_g, car_state.speed, throttle, steering
        )

        # ---- Layer 7: Steering rate limiter ----
        steering = self._limit_steering_rate(steering, car_state.speed)

        # ---- Layer 8: Brake/throttle mutex ----
        throttle, brake = self._brake_throttle_mutex(throttle, brake)

        # ---- Layer 9: Track boundary monitor ----
        throttle, steering = self._check_track_boundary(
            track_boundary_pct, car_state.speed, throttle, steering, is_on_track
        )

        # ---- Layer 10: Lap time watchdog ----
        verdict = self._check_lap_watchdog(car_state)
        if verdict.action == SafetyAction.KILL:
            return verdict

        # Update state tracking
        self.state.prev_steering_output = steering

        # Clamp final outputs
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steering = max(-1.0, min(1.0, steering))

        return SafetyVerdict(
            action=SafetyAction.PASS,
            throttle=throttle,
            brake=brake,
            steering=steering,
        )

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _check_heartbeat(self, car_state) -> SafetyVerdict:
        """Kill if no fresh telemetry data."""
        now = time.perf_counter()
        age_ms = (now - car_state.timestamp) * 1000.0

        if age_ms > self.config.heartbeat_timeout_ms:
            self.state.heartbeat_failures += 1
            if self.state.heartbeat_failures >= self.config.heartbeat_max_failures:
                self._log_kill("heartbeat", f"Telemetry stale for {age_ms:.0f}ms")
                return SafetyVerdict(
                    action=SafetyAction.KILL,
                    reason=f"Telemetry stale ({age_ms:.0f}ms)",
                    layer="heartbeat",
                )
        else:
            self.state.heartbeat_failures = 0

        self.state.last_telemetry_time = now
        return SafetyVerdict(action=SafetyAction.PASS)

    def _check_session_state(
        self, session_active: bool, is_on_track: bool, on_pit_road: bool
    ) -> SafetyVerdict:
        """Kill if session inactive or car is off track (not in pits)."""
        if not session_active:
            return SafetyVerdict(
                action=SafetyAction.KILL,
                reason="Session not active",
                layer="session_gate",
            )

        if not is_on_track and not on_pit_road:
            self.state.consecutive_off_track += 1
            if self.state.consecutive_off_track >= self.config.off_track_kill_frames:
                self._log_kill("session_gate",
                               f"Off track for {self.state.consecutive_off_track} frames")
                return SafetyVerdict(
                    action=SafetyAction.KILL,
                    reason=f"Off track for {self.state.consecutive_off_track} frames",
                    layer="session_gate",
                )
        else:
            self.state.consecutive_off_track = 0

        return SafetyVerdict(action=SafetyAction.PASS)

    def _check_incidents(self, car_state) -> SafetyVerdict:
        """Detect spins and stuck states."""
        # Spin detection via yaw rate (approximated from lateral G and speed)
        if car_state.speed > 3.0:
            # Approximate yaw rate from lat_g / speed (simplified bicycle model)
            yaw_rate = abs(car_state.lat_g / max(car_state.speed, 1.0))

            if yaw_rate > self.config.spin_yaw_rate_threshold:
                self.state.spin_frames += 1
            else:
                self.state.spin_frames = max(0, self.state.spin_frames - 2)

            if self.state.spin_frames >= self.config.spin_detect_frames:
                if not self.state.spin_detected:
                    self.state.spin_detected = True
                    self.state.spin_recovery_frames = 0
                    self.state.incident_count += 1
                    self.state.last_incident_time = time.perf_counter()
                    logger.warning(f"SPIN DETECTED (yaw_rate={yaw_rate:.2f} rad/s, "
                                   f"incident #{self.state.incident_count})")

        # Spin recovery: hold brake, center steering
        if self.state.spin_detected:
            self.state.spin_recovery_frames += 1
            if self.state.spin_recovery_frames > self.config.spin_recovery_timeout:
                self.state.spin_detected = False
                self.state.spin_frames = 0
                logger.info("Spin recovery timeout — resuming normal control")
            elif car_state.speed < 3.0 and self.state.spin_recovery_frames > 30:
                self.state.spin_detected = False
                self.state.spin_frames = 0
                logger.info("Spin recovery complete — car stopped, resuming")
            else:
                return SafetyVerdict(
                    action=SafetyAction.OVERRIDE,
                    throttle=0.0,
                    brake=0.8,
                    steering=0.0,
                    reason="Spin recovery — braking to stop",
                    layer="incident_detector",
                )

        # Stuck detection: car not moving for too long
        if car_state.speed < self.config.stuck_speed_threshold:
            self.state.stuck_frames += 1
        else:
            self.state.stuck_frames = 0
            self.state.stuck_detected = False

        if self.state.stuck_frames >= self.config.stuck_detect_frames:
            if not self.state.stuck_detected:
                self.state.stuck_detected = True
                self.state.incident_count += 1
                logger.warning(f"STUCK DETECTED (speed < {self.config.stuck_speed_threshold} m/s "
                               f"for {self.state.stuck_frames} frames)")
            self._log_kill("incident_detector", "Car stuck")
            return SafetyVerdict(
                action=SafetyAction.KILL,
                reason="Car stuck — manual intervention needed",
                layer="incident_detector",
            )

        # Too many incidents → kill session
        if self.state.incident_count >= self.config.max_incidents_per_session:
            self._log_kill("incident_detector",
                           f"Max incidents reached ({self.state.incident_count})")
            return SafetyVerdict(
                action=SafetyAction.KILL,
                reason=f"Too many incidents ({self.state.incident_count})",
                layer="incident_detector",
            )

        return SafetyVerdict(action=SafetyAction.PASS)

    def _govern_pit_speed(
        self, speed: float, throttle: float, brake: float
    ) -> Tuple[float, float]:
        """Enforce pit lane speed limit."""
        limit = self.config.pit_speed_limit_ms

        if speed > limit * 1.05:
            # Over limit — brake
            throttle = 0.0
            brake = min(0.5, (speed - limit) / limit)
        elif speed > limit * 0.9:
            # Near limit — coast
            throttle = min(throttle, 0.1)
        elif speed < limit * 0.5:
            # Well below — allow gentle throttle
            throttle = min(throttle, 0.4)

        return throttle, brake

    def _check_speed_envelope(
        self, speed: float, throttle: float, brake: float, steering: float
    ) -> Tuple[float, float, float, SafetyVerdict]:
        """Manage behavior at low/zero speed."""
        cfg = self.config

        if speed < 0.1:
            self.state.zero_speed_frames += 1
            if self.state.zero_speed_frames >= cfg.zero_speed_kill_frames:
                return 0, 0, 0, SafetyVerdict(
                    action=SafetyAction.KILL,
                    reason=f"Zero speed for {self.state.zero_speed_frames} frames",
                    layer="speed_envelope",
                )
        else:
            self.state.zero_speed_frames = 0

        if speed < cfg.min_speed_cutoff_ms:
            # Very low speed: gentle throttle only, no steering
            self.state.low_speed_frames += 1
            return cfg.crawl_throttle, 0.0, self.state.prev_steering_output, SafetyVerdict(
                action=SafetyAction.ATTENUATE
            )
        elif speed < cfg.crawl_speed_ms:
            # Low speed: reduce steering authority
            self.state.low_speed_frames += 1
            steer_limit = 0.3 + 0.7 * (speed / cfg.crawl_speed_ms)
            steering = max(-steer_limit, min(steer_limit, steering))
            return throttle, brake, steering, SafetyVerdict(action=SafetyAction.ATTENUATE)
        else:
            self.state.low_speed_frames = 0

        return throttle, brake, steering, SafetyVerdict(action=SafetyAction.PASS)

    def _limit_g_forces(
        self,
        lat_g: float,
        lon_g: float,
        speed: float,
        throttle: float,
        steering: float,
    ) -> Tuple[float, float]:
        """Reduce inputs when approaching G-force limits."""
        cfg = self.config

        # Track peaks for logging
        self.state.peak_lat_g = max(self.state.peak_lat_g, abs(lat_g))
        self.state.peak_lon_g = max(self.state.peak_lon_g, abs(lon_g))

        lat_ratio = abs(lat_g) / cfg.max_lat_g
        lon_ratio = abs(lon_g) / cfg.max_lon_g

        if lat_ratio > 0.9:
            # Near lateral G limit — reduce throttle to prevent oversteer
            reduction = 1.0 - (lat_ratio - 0.9) / 0.1 * cfg.g_force_throttle_reduction
            throttle *= max(0.2, reduction)
            self.state.g_limit_active = True
        else:
            self.state.g_limit_active = False

        if lon_ratio > 0.95:
            # Near longitudinal G limit — reduce throttle
            throttle *= 0.5

        return throttle, steering

    def _limit_steering_rate(self, steering: float, speed: float) -> float:
        """Cap steering rate of change to prevent snap inputs."""
        prev = self.state.prev_steering_output
        delta = steering - prev

        # Use tighter limits at high speed
        if speed > 30.0:  # > 108 km/h
            max_rate = self.config.max_steering_rate * 0.5
        elif speed > 15.0:  # > 54 km/h
            max_rate = self.config.max_steering_rate * 0.75
        else:
            max_rate = self.config.max_steering_rate

        # Allow emergency steering (hard braking zone)
        if abs(delta) > self.config.emergency_steering_rate:
            max_rate = self.config.emergency_steering_rate

        clamped_delta = max(-max_rate, min(max_rate, delta))
        return prev + clamped_delta

    def _brake_throttle_mutex(
        self, throttle: float, brake: float
    ) -> Tuple[float, float]:
        """Prevent simultaneous hard braking and throttle."""
        threshold = self.config.brake_throttle_threshold

        if brake > threshold:
            # Braking hard — scale down throttle proportionally
            throttle_limit = max(0.0, 1.0 - brake * 1.5)
            throttle = min(throttle, throttle_limit)
        elif throttle > 0.9 and brake > 0.1:
            # Full throttle — reduce light braking (left-foot braking ok at low levels)
            brake = min(brake, 0.1)

        return throttle, brake

    def _check_track_boundary(
        self,
        boundary_pct: float,
        speed: float,
        throttle: float,
        steering: float,
        is_on_track: bool,
    ) -> Tuple[float, float]:
        """Reduce speed and correct steering near track edges."""
        cfg = self.config

        if not is_on_track:
            # Off track — reduce throttle significantly
            return throttle * 0.2, steering

        if boundary_pct > cfg.track_edge_critical:
            # Critical — hard throttle reduction
            self.state.track_boundary_active = True
            throttle *= (1.0 - cfg.boundary_speed_reduction)
            if self._frame_count % 120 == 0:
                logger.warning(f"Track boundary CRITICAL: edge_pct={boundary_pct:.2f}")
        elif boundary_pct > cfg.track_edge_warning:
            # Warning — moderate reduction
            self.state.track_boundary_active = True
            reduction = (boundary_pct - cfg.track_edge_warning) / (
                cfg.track_edge_critical - cfg.track_edge_warning
            )
            throttle *= (1.0 - reduction * cfg.boundary_speed_reduction * 0.5)
        else:
            self.state.track_boundary_active = False

        return throttle, steering

    def _check_lap_watchdog(self, car_state) -> SafetyVerdict:
        """Kill if current lap is taking way too long (car probably crashed/stuck)."""
        if self.state.personal_best_s <= 0:
            return SafetyVerdict(action=SafetyAction.PASS)

        # Detect lap start
        pct = car_state.lap_dist_pct
        if self.state.last_lap_dist_pct > 0.95 and pct < 0.05:
            self.state.lap_start_time = time.perf_counter()
            self.state.lap_watchdog_triggered = False

        self.state.last_lap_dist_pct = pct

        if self.state.lap_start_time > 0:
            elapsed = time.perf_counter() - self.state.lap_start_time
            cutoff = self.state.personal_best_s * self.config.lap_time_kill_multiplier

            if elapsed > cutoff and not self.state.lap_watchdog_triggered:
                self.state.lap_watchdog_triggered = True
                self._log_kill("lap_watchdog",
                               f"Lap time {elapsed:.1f}s exceeds "
                               f"{self.config.lap_time_kill_multiplier}x PB "
                               f"({self.state.personal_best_s:.1f}s)")
                return SafetyVerdict(
                    action=SafetyAction.KILL,
                    reason=f"Lap too slow ({elapsed:.1f}s vs {cutoff:.1f}s limit)",
                    layer="lap_watchdog",
                )

        return SafetyVerdict(action=SafetyAction.PASS)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_personal_best(self, pb_seconds: float):
        """Set personal best for lap time watchdog."""
        self.state.personal_best_s = pb_seconds
        logger.info(f"Safety: personal best set to {pb_seconds:.3f}s")

    def reset_lap(self):
        """Reset lap-related state (call on lap completion)."""
        self.state.lap_start_time = time.perf_counter()
        self.state.lap_watchdog_triggered = False

    def reset(self):
        """Full reset of safety state (call on session start)."""
        pb = self.state.personal_best_s  # preserve PB across resets
        self.state = SafetyState()
        self.state.personal_best_s = pb
        self._frame_count = 0
        logger.info("Safety controller reset")

    def _log_kill(self, layer: str, reason: str):
        """Log a kill event."""
        self.state.total_kills += 1
        self.state.frames_since_last_kill = 0
        logger.warning(f"SAFETY KILL [{layer}]: {reason} "
                       f"(kill #{self.state.total_kills})")

    def get_summary(self) -> dict:
        """Return session safety summary for logging."""
        return {
            "total_interventions": self.state.total_interventions,
            "total_kills": self.state.total_kills,
            "incident_count": self.state.incident_count,
            "peak_lat_g": self.state.peak_lat_g,
            "peak_lon_g": self.state.peak_lon_g,
            "spins_detected": self.state.incident_count,
        }

    def log_summary(self):
        """Log the session safety summary."""
        s = self.get_summary()
        logger.info(
            f"Safety summary: {s['total_kills']} kills, "
            f"{s['incident_count']} incidents, "
            f"peak G: lat={s['peak_lat_g']:.1f} lon={s['peak_lon_g']:.1f} m/s^2"
        )
