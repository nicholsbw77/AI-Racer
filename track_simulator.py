"""
track_simulator.py

Virtual 2D track simulator for offline AI driver testing.
No iRacing required — uses simplified bicycle model physics.

Usage:
  python track_simulator.py                    # Run PID demo on oval
  python track_simulator.py --track road       # Run on road course
  python track_simulator.py --laps 3           # Run 3 laps

Components:
  Track          - Centerline + width definition
  CarPhysics     - Simplified bicycle model
  TrackSimulator - Combines track + car + state vector builder
"""

import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track Definition
# ---------------------------------------------------------------------------

@dataclass
class TrackPoint:
    """A single point on the track centerline."""
    x: float
    y: float
    heading: float  # radians, 0=east, pi/2=north
    cumulative_dist: float = 0.0


class Track:
    """
    A 2D track defined by a centerline polyline and a width.

    The centerline is stored as an array of (x, y) waypoints.
    Queries use linear interpolation between waypoints.
    """

    def __init__(self, centerline: np.ndarray, width: float = 12.0, name: str = "track"):
        """
        Args:
            centerline: (N, 2) array of (x, y) waypoints in meters.
            width: Track width in meters (constant).
            name: Human-readable name.
        """
        self.centerline = np.asarray(centerline, dtype=np.float64)
        self.width = width
        self.name = name

        # Compute cumulative distances and headings
        diffs = np.diff(self.centerline, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._cumulative_dist = np.zeros(len(self.centerline))
        self._cumulative_dist[1:] = np.cumsum(seg_lengths)
        self.total_length = self._cumulative_dist[-1]

        # Heading at each point (tangent direction)
        self._headings = np.zeros(len(self.centerline))
        self._headings[:-1] = np.arctan2(diffs[:, 1], diffs[:, 0])
        self._headings[-1] = self._headings[-2]

        # Curvature at each point
        self._curvatures = np.zeros(len(self.centerline))
        if len(self.centerline) > 2:
            dh = np.diff(self._headings)
            # Handle wrapping
            dh = (dh + np.pi) % (2 * np.pi) - np.pi
            ds = seg_lengths
            ds = np.maximum(ds, 0.01)
            # dh has N-1 elements, ds has N-1 elements, curvatures[1:-1] needs N-2
            self._curvatures[1:-1] = dh[:-1] / ds[:-1]

        # Build normalized pct lookup
        self._pct = self._cumulative_dist / self.total_length

    def get_point_at_pct(self, pct: float) -> Tuple[float, float, float]:
        """Get (x, y, heading) at a normalized track position [0, 1]."""
        pct = pct % 1.0
        dist = pct * self.total_length
        idx = np.searchsorted(self._cumulative_dist, dist) - 1
        idx = max(0, min(idx, len(self.centerline) - 2))

        seg_start = self._cumulative_dist[idx]
        seg_len = self._cumulative_dist[idx + 1] - seg_start
        if seg_len < 1e-6:
            t = 0.0
        else:
            t = (dist - seg_start) / seg_len

        p0 = self.centerline[idx]
        p1 = self.centerline[idx + 1]
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])

        h0 = self._headings[idx]
        h1 = self._headings[idx + 1]
        dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi
        heading = h0 + t * dh

        return float(x), float(y), float(heading)

    def get_curvature_at_pct(self, pct: float) -> float:
        """Get track curvature at a normalized position."""
        pct = pct % 1.0
        idx = int(pct * (len(self._curvatures) - 1))
        idx = max(0, min(idx, len(self._curvatures) - 1))
        return float(self._curvatures[idx])

    def get_lateral_offset(self, x: float, y: float, hint_pct: float = 0.0) -> Tuple[float, float]:
        """
        Get lateral offset from centerline and the closest lap_dist_pct.

        Returns:
            (track_pos, closest_pct) where track_pos is in [-1, +1]
            (negative = left of centerline, positive = right)
        """
        # Search near the hint first for efficiency
        n = len(self.centerline)
        hint_idx = int(hint_pct * (n - 1)) % n

        # Search window around hint (wider to handle corners)
        window = max(n // 3, 80)
        best_dist_sq = float("inf")
        best_idx = hint_idx
        best_t = 0.0

        for di in range(-window, window + 1):
            i = (hint_idx + di) % (n - 1)
            p0 = self.centerline[i]
            p1 = self.centerline[(i + 1) % n]

            seg = p1 - p0
            seg_len_sq = seg[0] ** 2 + seg[1] ** 2
            if seg_len_sq < 1e-10:
                continue

            # Project point onto segment
            t = ((x - p0[0]) * seg[0] + (y - p0[1]) * seg[1]) / seg_len_sq
            t = max(0.0, min(1.0, t))

            px = p0[0] + t * seg[0]
            py = p0[1] + t * seg[1]
            d_sq = (x - px) ** 2 + (y - py) ** 2

            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_idx = i
                best_t = t

        # Compute signed lateral offset
        p0 = self.centerline[best_idx]
        p1 = self.centerline[(best_idx + 1) % n]
        heading = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        # Vector from centerline to car
        px = p0[0] + best_t * (p1[0] - p0[0])
        py = p0[1] + best_t * (p1[1] - p0[1])
        dx = x - px
        dy = y - py

        # Cross product gives signed distance (positive = right of heading)
        cross = math.cos(heading) * dy - math.sin(heading) * dx
        lateral_dist = math.copysign(math.sqrt(best_dist_sq), cross)

        # Normalize to [-1, +1] using half-width
        track_pos = np.clip(lateral_dist / (self.width / 2.0), -1.0, 1.0)

        # Compute lap_dist_pct
        dist = self._cumulative_dist[best_idx] + best_t * (
            self._cumulative_dist[min(best_idx + 1, n - 1)] - self._cumulative_dist[best_idx]
        )
        closest_pct = dist / self.total_length

        return float(track_pos), float(closest_pct)

    def is_on_track(self, x: float, y: float, hint_pct: float = 0.0) -> bool:
        """Check if (x, y) is within track boundaries."""
        track_pos, _ = self.get_lateral_offset(x, y, hint_pct)
        return abs(track_pos) <= 1.0

    # ----- Factory methods -----

    @classmethod
    def simple_oval(cls, length: float = 1500.0, corner_radius: float = 150.0,
                    width: float = 15.0, n_points: int = 400) -> "Track":
        """Create a simple oval track (closed loop)."""
        straight = length / 2 - corner_radius * np.pi / 2
        straight = max(straight, 100.0)

        points = []
        pts_per_section = n_points // 4

        # Bottom straight (left to right)
        for i in range(pts_per_section):
            t = i / pts_per_section
            points.append([t * straight, 0.0])

        # Right turn (semicircle, going from bottom to top)
        for i in range(pts_per_section):
            angle = -np.pi / 2 + np.pi * i / pts_per_section
            points.append([
                straight + corner_radius * np.cos(angle),
                corner_radius + corner_radius * np.sin(angle),
            ])

        # Top straight (right to left)
        for i in range(pts_per_section):
            t = 1.0 - i / pts_per_section
            points.append([t * straight, 2 * corner_radius])

        # Left turn (semicircle, going from top to bottom)
        for i in range(pts_per_section):
            angle = np.pi / 2 + np.pi * i / pts_per_section
            points.append([
                corner_radius * np.cos(angle),
                corner_radius + corner_radius * np.sin(angle),
            ])

        # Close the loop by appending the first point
        points.append(points[0])

        return cls(np.array(points), width=width, name="simple_oval")

    @classmethod
    def road_course(cls, width: float = 12.0, n_points: int = 400) -> "Track":
        """Create a road course with varied corners."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Parametric road course with multiple curve types
        r_base = 300.0
        x = (r_base + 80 * np.sin(2 * t) + 40 * np.sin(3 * t)) * np.cos(t)
        y = (r_base + 80 * np.sin(2 * t) + 40 * np.sin(3 * t)) * np.sin(t)

        # Add chicane perturbation
        x += 30 * np.sin(5 * t)
        y += 20 * np.cos(4 * t)

        points = np.column_stack([x, y])
        # Close the loop
        points = np.vstack([points, points[0:1]])
        return cls(points, width=width, name="road_course")

    @classmethod
    def figure_eight(cls, radius: float = 150.0, width: float = 12.0,
                     n_points: int = 300) -> "Track":
        """Create a figure-8 track (lemniscate of Bernoulli variant)."""
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        scale = radius * 1.5
        denom = 1 + np.sin(t) ** 2
        x = scale * np.cos(t) / denom
        y = scale * np.sin(t) * np.cos(t) / denom
        points = np.column_stack([x, y])
        # Close the loop
        points = np.vstack([points, points[0:1]])
        return cls(points, width=width, name="figure_eight")


# ---------------------------------------------------------------------------
# Car Physics (Simplified Bicycle Model)
# ---------------------------------------------------------------------------

@dataclass
class CarConfig:
    """Physical parameters for the simplified car model."""
    max_speed: float = 80.0          # m/s (~180 mph)
    max_accel: float = 12.0          # m/s² (full throttle)
    max_brake_decel: float = 35.0    # m/s² (hard braking)
    max_steer_angle: float = 1.0     # radians (~57 degrees)
    wheelbase: float = 2.7           # meters
    drag_coeff: float = 0.002        # aerodynamic drag
    rolling_resistance: float = 0.3  # m/s² constant resistance
    off_track_drag: float = 5.0      # extra drag when off track
    off_track_grip: float = 0.4      # grip multiplier off track
    mass: float = 1400.0             # kg (for G-force calc)


class CarPhysics:
    """Simplified bicycle model car physics."""

    def __init__(self, config: Optional[CarConfig] = None):
        self.cfg = config or CarConfig()
        self.reset()

    def reset(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0):
        """Reset car to initial state."""
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0
        self.gear = 1
        self.rpm = 1000.0
        self.lat_accel = 0.0
        self.lon_accel = 0.0
        self.yaw_rate = 0.0
        self._prev_speed = 0.0

    def step(self, throttle: float, brake: float, steering: float,
             dt: float = 1.0 / 60, on_track: bool = True) -> None:
        """
        Advance physics by one timestep.

        Args:
            throttle: 0.0 to 1.0
            brake: 0.0 to 1.0
            steering: -1.0 (left) to +1.0 (right)
            dt: timestep in seconds
            on_track: whether car is on track surface
        """
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)

        c = self.cfg
        self._prev_speed = self.speed

        # Grip and drag modifiers for off-track
        grip = 1.0 if on_track else c.off_track_grip
        extra_drag = 0.0 if on_track else c.off_track_drag

        # Longitudinal dynamics
        accel = throttle * c.max_accel * grip
        decel = brake * c.max_brake_decel
        drag = c.drag_coeff * self.speed ** 2 + c.rolling_resistance + extra_drag

        net_accel = accel - decel - drag
        self.speed = max(0.0, self.speed + net_accel * dt)
        self.speed = min(self.speed, c.max_speed)
        self.lon_accel = net_accel

        # Steering dynamics (bicycle model)
        steer_angle = steering * c.max_steer_angle

        if self.speed > 0.5:
            # Turn radius = wheelbase / tan(steer_angle)
            if abs(steer_angle) > 0.001:
                turn_radius = c.wheelbase / math.tan(steer_angle)
                self.yaw_rate = self.speed / turn_radius * grip
            else:
                self.yaw_rate = 0.0

            # Limit yaw rate by available grip
            max_yaw = (grip * 40.0) / max(self.speed, 1.0)  # ~4g lateral limit
            self.yaw_rate = np.clip(self.yaw_rate, -max_yaw, max_yaw)

            # Lateral acceleration
            self.lat_accel = self.speed * self.yaw_rate
        else:
            self.yaw_rate = 0.0
            self.lat_accel = 0.0

        # Update position and heading
        self.heading += self.yaw_rate * dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        self.x += self.speed * math.cos(self.heading) * dt
        self.y += self.speed * math.sin(self.heading) * dt

        # Simple gear/RPM model
        self.gear = max(1, min(6, int(self.speed / (c.max_speed / 6)) + 1))
        gear_ratio = [0, 3.5, 2.5, 1.8, 1.3, 1.0, 0.8]
        self.rpm = min(8000, max(800, self.speed * gear_ratio[self.gear] * 60))

    def get_state_dict(self) -> dict:
        """Return state matching CarState fields."""
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "speed": self.speed,
            "gear": self.gear,
            "rpm": self.rpm,
            "lat_g": self.lat_accel,
            "lon_g": self.lon_accel,
            "yaw_rate": self.yaw_rate,
        }


# ---------------------------------------------------------------------------
# Track Simulator
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Statistics for a simulation episode."""
    laps_completed: int = 0
    total_frames: int = 0
    on_track_frames: int = 0
    off_track_count: int = 0
    total_time: float = 0.0
    average_speed: float = 0.0
    max_speed: float = 0.0
    best_lap_time: float = float("inf")
    lap_times: List[float] = field(default_factory=list)

    @property
    def on_track_pct(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return 100.0 * self.on_track_frames / self.total_frames


class TrackSimulator:
    """
    Combines Track + CarPhysics for offline AI testing.

    The simulator produces state vectors compatible with the trained model,
    allowing direct testing of checkpoints without iRacing.
    """

    def __init__(self, track: Track, car_config: Optional[CarConfig] = None,
                 hz: int = 60):
        self.track = track
        self.car = CarPhysics(car_config)
        self.hz = hz
        self.dt = 1.0 / hz

        # State tracking
        self._lap_dist_pct = 0.0
        self._track_pos = 0.0
        self._on_track = True
        self._last_lap_pct = 0.0
        self._lap_start_time = 0.0
        self._sim_time = 0.0
        self._speed_max = 80.0  # for normalization
        self._steering_lock = 0.6

        # Action history ring buffer
        self._history_len = 20
        self._throttle_hist = np.zeros(self._history_len, dtype=np.float32)
        self._brake_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steering_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steer_delta_hist = np.zeros(self._history_len, dtype=np.float32)
        self._hist_ptr = 0

    def reset(self) -> dict:
        """Place car at start line and return initial observation."""
        x, y, heading = self.track.get_point_at_pct(0.0)
        self.car.reset(x, y, heading)
        self._lap_dist_pct = 0.0
        self._track_pos = 0.0
        self._on_track = True
        self._last_lap_pct = 0.0
        self._lap_start_time = 0.0
        self._sim_time = 0.0

        # Reset history
        self._throttle_hist[:] = 0.0
        self._brake_hist[:] = 0.0
        self._steering_hist[:] = 0.0
        self._steer_delta_hist[:] = 0.0
        self._hist_ptr = 0

        return self._get_observation()

    def step(self, throttle: float, brake: float, steering: float) -> Tuple[dict, bool]:
        """
        Advance one physics step.

        Returns:
            (observation_dict, lap_completed)
        """
        # Update physics
        self.car.step(throttle, brake, steering, self.dt, self._on_track)
        self._sim_time += self.dt

        # Update track position
        self._track_pos, self._lap_dist_pct = self.track.get_lateral_offset(
            self.car.x, self.car.y, self._lap_dist_pct
        )
        self._on_track = abs(self._track_pos) <= 1.0

        # Update history
        ptr = self._hist_ptr % self._history_len
        prev_ptr = (self._hist_ptr - 1) % self._history_len
        self._throttle_hist[ptr] = throttle
        self._brake_hist[ptr] = brake
        steer_norm = np.clip(steering, -1.0, 1.0)
        prev_steer = self._steering_hist[prev_ptr]
        self._steering_hist[ptr] = steer_norm
        self._steer_delta_hist[ptr] = np.clip(steer_norm - prev_steer, -0.3, 0.3)
        self._hist_ptr += 1

        # Detect lap completion (pct wraps from ~1.0 back to ~0.0)
        lap_completed = False
        if self._last_lap_pct > 0.90 and self._lap_dist_pct < 0.10:
            lap_completed = True
            self._lap_start_time = self._sim_time
        # Also detect wrap the other way (car going backwards)
        elif self._last_lap_pct < 0.10 and self._lap_dist_pct > 0.90:
            pass  # Going backwards, ignore

        self._last_lap_pct = self._lap_dist_pct

        return self._get_observation(), lap_completed

    def _get_observation(self) -> dict:
        """Build observation dict matching telemetry CarState fields."""
        return {
            "speed": self.car.speed,
            "throttle": self._throttle_hist[(self._hist_ptr - 1) % self._history_len],
            "brake": self._brake_hist[(self._hist_ptr - 1) % self._history_len],
            "steering": self.car.heading,
            "gear": self.car.gear,
            "rpm": self.car.rpm,
            "lat_g": self.car.lat_accel,
            "lon_g": self.car.lon_accel,
            "yaw_rate": self.car.yaw_rate,
            "lap_dist_pct": self._lap_dist_pct,
            "track_pos": self._track_pos,
            "is_on_track": self._on_track,
            "on_pit_road": False,
            "session_active": True,
            "x": self.car.x,
            "y": self.car.y,
            "heading": self.car.heading,
            "sim_time": self._sim_time,
        }

    def build_state_vector(self, obs: dict, sequence_history: int = 15) -> np.ndarray:
        """
        Build model-compatible state vector from observation.
        Matches telemetry.py build_state_vector format exactly.
        """
        speed_norm = np.clip(obs["speed"] / self._speed_max, 0.0, 1.0)
        speed_delta = 0.0  # simplified
        gear_norm = np.clip(obs["gear"] / 7.0, 0.0, 1.0)
        rpm_norm = np.clip(obs["rpm"] / 8000.0, 0.0, 1.0)
        lat_g_norm = np.clip(obs["lat_g"] / 40.0, -1.0, 1.0)
        lon_g_norm = np.clip(obs["lon_g"] / 40.0, -1.0, 1.0)
        steer_norm = np.clip(obs.get("steering_input", 0.0), -1.0, 1.0)
        steer_abs = abs(steer_norm)
        heavy_braking = float(obs["brake"] > 0.3 and lon_g_norm < -0.05)
        full_throttle = float(obs["throttle"] > 0.95)

        track_pos = obs["track_pos"]
        track_pos_abs = abs(track_pos)
        near_edge = float(track_pos_abs > 0.75)
        on_rumble = float(track_pos_abs > 0.90)
        track_pos_sign = float(np.sign(track_pos))

        current_state = np.array([
            obs["lap_dist_pct"],
            speed_norm,
            speed_delta,
            gear_norm,
            rpm_norm,
            lat_g_norm,
            lon_g_norm,
            track_pos,
            steer_abs,
            heavy_braking,
            full_throttle,
            near_edge,
            on_rumble,
            track_pos_sign,
        ], dtype=np.float32)

        # Action history
        history_frames = []
        for i in range(sequence_history):
            p = (self._hist_ptr - 1 - i) % self._history_len
            frame = np.array([
                self._throttle_hist[p],
                self._brake_hist[p],
                self._steering_hist[p],
                self._steer_delta_hist[p],
            ], dtype=np.float32)
            history_frames.append(frame)

        history_flat = np.concatenate(history_frames)
        return np.concatenate([current_state, history_flat])

    def run_episode(
        self,
        agent_fn: Callable[[dict], Tuple[float, float, float]],
        max_laps: int = 1,
        max_steps: int = 100000,
    ) -> EpisodeStats:
        """
        Run a full episode with an agent function.

        Args:
            agent_fn: Callable that takes observation dict, returns (throttle, brake, steering)
            max_laps: Stop after this many laps
            max_steps: Safety limit on total steps

        Returns:
            EpisodeStats with performance metrics
        """
        stats = EpisodeStats()
        obs = self.reset()
        speed_sum = 0.0
        was_off_track = False
        lap_start_step = 0

        for step_i in range(max_steps):
            throttle, brake, steering = agent_fn(obs)
            obs, lap_completed = self.step(throttle, brake, steering)
            # Store steering input for state vector
            obs["steering_input"] = steering

            stats.total_frames += 1
            speed_sum += obs["speed"]
            stats.max_speed = max(stats.max_speed, obs["speed"])

            if obs["is_on_track"]:
                stats.on_track_frames += 1
                was_off_track = False
            else:
                if not was_off_track:
                    stats.off_track_count += 1
                    was_off_track = True

            if lap_completed:
                stats.laps_completed += 1
                lap_time = (step_i - lap_start_step) * self.dt
                stats.lap_times.append(lap_time)
                stats.best_lap_time = min(stats.best_lap_time, lap_time)
                lap_start_step = step_i
                logger.info(
                    f"Lap {stats.laps_completed}: {lap_time:.2f}s "
                    f"(on_track: {stats.on_track_pct:.1f}%)"
                )
                if stats.laps_completed >= max_laps:
                    break

        stats.total_time = stats.total_frames * self.dt
        stats.average_speed = speed_sum / max(stats.total_frames, 1)

        return stats


# ---------------------------------------------------------------------------
# PID Controller (Demo)
# ---------------------------------------------------------------------------

class PIDTrackFollower:
    """
    PID controller that follows the track centerline using both
    lateral offset error and heading error for stable cornering.
    """

    def __init__(self, track: Track):
        self.track = track
        self._lateral_integral = 0.0
        self._prev_lateral_error = 0.0

        # Steering gains
        self.kp_lateral = 1.5       # proportional on lateral offset
        self.ki_lateral = 0.005     # integral on lateral offset
        self.kd_lateral = 0.3       # derivative on lateral offset
        self.kp_heading = 1.0       # proportional on heading error

        # Speed control
        self.target_speed = 30.0  # m/s base target

    def __call__(self, obs: dict) -> Tuple[float, float, float]:
        """PID control step. Returns (throttle, brake, steering)."""
        track_pos = obs["track_pos"]
        speed = obs["speed"]
        lap_pct = obs["lap_dist_pct"]
        car_heading = obs["heading"]

        # --- Heading error ---
        _, _, track_heading = self.track.get_point_at_pct(lap_pct)
        heading_error = (track_heading - car_heading + math.pi) % (2 * math.pi) - math.pi

        # --- Lateral PID ---
        lat_error = track_pos
        self._lateral_integral = np.clip(self._lateral_integral + lat_error, -3.0, 3.0)
        lat_derivative = lat_error - self._prev_lateral_error
        self._prev_lateral_error = lat_error

        lateral_correction = -(
            self.kp_lateral * lat_error +
            self.ki_lateral * self._lateral_integral +
            self.kd_lateral * lat_derivative
        )

        # --- Combined steering: heading + lateral correction ---
        # Heading error dominates to keep car pointed along track
        # Lateral correction nudges car toward centerline
        steering = self.kp_heading * heading_error + lateral_correction * 0.5
        steering = np.clip(steering, -1.0, 1.0)

        # --- Speed control ---
        curvature = abs(self.track.get_curvature_at_pct(lap_pct))

        # Look ahead for upcoming curvature
        lookahead_dist = max(speed * 1.5, 20.0)  # 1.5 seconds ahead
        lookahead_pct = (lap_pct + lookahead_dist / self.track.total_length) % 1.0
        lookahead_curv = abs(self.track.get_curvature_at_pct(lookahead_pct))
        curvature = max(curvature, lookahead_curv)

        # Target speed decreases with curvature
        # v_max = sqrt(a_lat_max / curvature), a_lat_max ~ 20 m/s²
        if curvature > 0.001:
            corner_speed = min(self.target_speed, math.sqrt(20.0 / curvature))
        else:
            corner_speed = self.target_speed
        target_speed = max(8.0, corner_speed)

        # Slow down when near edge
        if abs(track_pos) > 0.7:
            target_speed *= 0.7

        # Throttle/brake based on speed error
        speed_error = target_speed - speed
        if speed_error > 2.0:
            throttle = min(1.0, 0.3 + speed_error * 0.1)
            brake = 0.0
        elif speed_error < -3.0:
            throttle = 0.0
            brake = min(1.0, abs(speed_error) * 0.15)
        else:
            throttle = 0.3
            brake = 0.0

        return throttle, brake, steering


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Track Simulator Demo")
    parser.add_argument("--track", default="oval",
                        choices=["oval", "road", "figure8"],
                        help="Track type")
    parser.add_argument("--laps", type=int, default=2, help="Number of laps")
    parser.add_argument("--hz", type=int, default=60, help="Simulation Hz")
    parser.add_argument("--speed", type=float, default=30.0,
                        help="PID target speed (m/s)")
    args = parser.parse_args()

    # Create track
    if args.track == "oval":
        track = Track.simple_oval()
    elif args.track == "road":
        track = Track.road_course()
    elif args.track == "figure8":
        track = Track.figure_eight()
    else:
        track = Track.simple_oval()

    logger.info(f"Track: {track.name} ({track.total_length:.0f}m, {track.width:.0f}m wide)")

    # Create simulator
    sim = TrackSimulator(track, hz=args.hz)

    # Create PID controller
    pid = PIDTrackFollower(track)
    pid.target_speed = args.speed

    # Run episode
    logger.info(f"Running {args.laps} laps with PID controller (target {args.speed:.0f} m/s)...")
    stats = sim.run_episode(pid, max_laps=args.laps)

    # Print results
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    print(f"Track:           {track.name}")
    print(f"Laps completed:  {stats.laps_completed}")
    print(f"Total time:      {stats.total_time:.1f}s")
    print(f"On-track:        {stats.on_track_pct:.1f}%")
    print(f"Off-track events:{stats.off_track_count}")
    print(f"Average speed:   {stats.average_speed:.1f} m/s ({stats.average_speed * 3.6:.0f} km/h)")
    print(f"Max speed:       {stats.max_speed:.1f} m/s ({stats.max_speed * 3.6:.0f} km/h)")
    if stats.lap_times:
        print(f"Best lap:        {stats.best_lap_time:.2f}s")
        print(f"Lap times:       {[f'{t:.2f}s' for t in stats.lap_times]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
