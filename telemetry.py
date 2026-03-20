"""
agent/telemetry.py

Reads live telemetry from iRacing via pyirsdk shared memory.
Runs at up to 360Hz synchronized to iRacing's physics tick.

Key iRacing SDK variables used:
  Speed           - m/s
  Throttle        - 0-1
  Brake           - 0-1
  SteeringWheelAngle - radians
  Gear            - integer
  RPM             - float
  LatAccel        - m/s² lateral
  LongAccel       - m/s² longitudinal
  LapDistPct      - 0-1 normalized track position
  PlayerTrackSurface - surface type (track/pit/grass etc)
  SessionState    - session status
  IsOnTrack       - bool
  YawRate         - rad/s yaw (for lateral position estimation)
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import pyirsdk - not available on Linux/training machines
try:
    import irsdk
    IRSDK_AVAILABLE = True
except ImportError:
    IRSDK_AVAILABLE = False
    logger.warning("pyirsdk not available - telemetry reader will use mock mode")


# iRacing PlayerTrackSurface enum values
SURFACE_NOT_IN_WORLD = -1
SURFACE_OFF_TRACK = 0
SURFACE_IN_PIT_STALL = 1
SURFACE_APPROACHING_PITS = 2
SURFACE_ON_TRACK = 3


@dataclass
class CarState:
    """Snapshot of car state at a single time step."""
    speed: float = 0.0           # m/s
    throttle: float = 0.0        # 0-1
    brake: float = 0.0           # 0-1
    steering: float = 0.0        # radians (raw)
    gear: float = 0.0            # integer gear
    rpm: float = 0.0             # RPM
    lat_g: float = 0.0           # m/s² lateral acceleration
    lon_g: float = 0.0           # m/s² longitudinal acceleration
    yaw_rate: float = 0.0        # rad/s yaw rotation
    lap_dist_pct: float = 0.0    # 0-1 position on track
    track_pos: float = 0.0       # lateral offset (-1 to +1), estimated
    surface_type: int = 3        # iRacing surface enum (3=on track)
    is_on_track: bool = False
    on_pit_road: bool = False
    session_active: bool = False
    track_id: str = ""
    car_id: str = ""
    timestamp: float = field(default_factory=time.perf_counter)


class TelemetryReader:
    """
    Reads iRacing telemetry at up to 360Hz.
    Uses wait_for_data() to sync exactly to iRacing's physics tick.
    Thread-safe via a lock on the latest state.
    """

    def __init__(self, target_hz: int = 360):
        self.target_hz = target_hz
        self._ir = None
        self._connected = False
        self._latest_state: Optional[CarState] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Normalization constants (updated on track/car change)
        self._speed_max = 100.0      # m/s, updated from observed max
        self._rpm_max = 8000.0
        self._steering_lock = np.pi  # radians, ~180deg default

        # State history for feature engineering (ring buffer)
        # Must be >= sequence_history (up to 90 frames at 360Hz/250ms)
        self._history_len = 100
        self._throttle_hist = np.zeros(self._history_len, dtype=np.float32)
        self._brake_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steering_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steer_delta_hist = np.zeros(self._history_len, dtype=np.float32)
        self._hist_ptr = 0

        # Track position estimator state
        self._track_pos_estimate = 0.0
        self._track_map = None  # Optional TrackMap for better estimation

        # Forced track position override (for pit exit seeding)
        self._forced_track_pos = None    # float or None
        self._forced_blend = 0.0         # 1.0 = fully forced, decays to 0.0
        self._forced_decay_rate = 0.005  # per frame (~3s to halve at 60Hz)

    def connect(self) -> bool:
        """Attempt to connect to iRacing. Returns True if successful."""
        if not IRSDK_AVAILABLE:
            logger.warning("pyirsdk not available, running in mock mode")
            return False

        self._ir = irsdk.IRSDK()
        result = self._ir.startup()
        if result and self._ir.is_initialized and self._ir.is_connected:
            self._connected = True
            logger.info("Connected to iRacing")
            self._update_car_track_info()
            return True
        else:
            logger.warning("iRacing not running or not connected")
            return False

    def _update_car_track_info(self):
        """Read static session info (track name, car)."""
        if not self._connected:
            return
        try:
            weekend = self._ir["WeekendInfo"]
            if weekend:
                self._track_name = weekend.get("TrackDisplayName", "unknown_track")
                self._track_id = str(weekend.get("TrackID", ""))
                logger.info(f"Track: {self._track_name} (ID: {self._track_id})")

            drivers = self._ir["DriverInfo"]
            if drivers:
                driver_idx = drivers.get("DriverCarIdx", 0)
                driver_list = drivers.get("Drivers", [])
                if driver_list and driver_idx < len(driver_list):
                    self._car_name = driver_list[driver_idx].get("CarScreenName", "unknown_car")
                    self._car_id = str(driver_list[driver_idx].get("CarID", ""))
                    logger.info(f"Car: {self._car_name} (ID: {self._car_id})")
        except Exception as e:
            logger.warning(f"Could not read session info: {e}")

    def get_combo_name(self) -> str:
        """Return a sanitized track_car combo string for checkpoint lookup."""
        track = getattr(self, "_track_name", "unknown_track")
        car = getattr(self, "_car_name", "unknown_car")
        # Sanitize: lowercase, replace spaces/special chars with underscore
        combo = f"{track}_{car}".lower()
        for ch in " -/.()":
            combo = combo.replace(ch, "_")
        # Collapse multiple underscores
        while "__" in combo:
            combo = combo.replace("__", "_")
        return combo.strip("_")

    def start(self):
        """Start background telemetry reading thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name="TelemetryReader",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"Telemetry reader started (target {self.target_hz}Hz)")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Telemetry reader stopped")

    def _read_loop(self):
        """
        Main telemetry loop. Runs in background thread.
        Uses wait_for_data() to sync to iRacing physics tick.
        """
        # Set thread to high priority on Windows
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                2  # THREAD_PRIORITY_HIGHEST
            )
        except Exception:
            pass

        timeout = 1.0 / self.target_hz * 1000  # ms

        while self._running:
            if not self._connected:
                time.sleep(0.5)
                self.connect()
                continue

            try:
                # Sync to iRacing physics tick
                # pyirsdk >=1.2 has wait_for_data, older versions use freeze_var_buffer
                if hasattr(self._ir, 'wait_for_data'):
                    if not self._ir.wait_for_data(timeout_ms=int(timeout)):
                        continue
                else:
                    # Fallback: poll at target Hz
                    self._ir.freeze_var_buffer_latest()
                    time.sleep(1.0 / self.target_hz)

                state = self._read_state()
                with self._lock:
                    self._latest_state = state
                    self._update_history(state)

            except ConnectionError:
                logger.warning("iRacing disconnected")
                self._connected = False
            except Exception as e:
                logger.warning(f"Telemetry read error: {e}")
                time.sleep(1.0 / self.target_hz)

    @staticmethod
    def _to_float(val, default=0.0) -> float:
        """Safely convert an iRacing SDK value to float.
        Some pyirsdk versions return lists for certain vars."""
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return float(val[0]) if val else default
        return float(val)

    def set_track_map(self, track_map):
        """Attach a TrackMap for improved track position estimation."""
        self._track_map = track_map
        logger.info("Track map attached for position estimation")

    def set_forced_track_pos(self, track_pos: float, blend: float = 1.0,
                              decay_rate: float = 0.005):
        """Seed the track position with a known value.

        Use this when you know exactly where the car is (e.g. pit exit).
        The forced value blends with the estimated value and decays over time.

        Args:
            track_pos: Known lateral position (-1 to +1).
            blend: Initial blend factor (1.0 = fully forced).
            decay_rate: Per-frame decay (0.005 ≈ 3s to halve at 60Hz).
        """
        self._forced_track_pos = float(track_pos)
        self._forced_blend = float(blend)
        self._forced_decay_rate = float(decay_rate)
        logger.info(
            "Forced track_pos seeded: %.2f (blend=%.2f, decay=%.4f)",
            track_pos, blend, decay_rate,
        )

    def _estimate_track_pos(
        self, speed: float, steering: float, lat_g: float,
        yaw_rate: float, lap_dist_pct: float, dt: float = 1.0 / 60
    ) -> float:
        """
        Estimate lateral track position from dynamics.

        Uses three complementary signals:
        1. Steering deviation from expected (if track map available)
        2. Yaw rate integration for lateral drift detection
        3. Lateral G vs expected curvature mismatch

        The estimate drifts toward center (0.0) slowly when signals are
        ambiguous, which is safe — the safety controller handles edge cases.
        """
        # Approach: integrate lateral velocity relative to track centerline
        # lateral_vel ≈ speed * sin(slip_angle) ≈ yaw_rate * speed deviation
        # Simplified: use steering angle as proxy for lateral intent

        steer_norm = np.clip(steering / self._steering_lock, -1.0, 1.0)

        if self._track_map is not None:
            # Use track map for better estimation
            estimated_pos = self._track_map.estimate_track_pos(
                lap_dist_pct, speed, steering, lat_g
            )
        else:
            # Fallback: integrate from yaw rate and lateral G
            # Lateral acceleration = centripetal + lateral drift
            # If speed > 0, lateral drift rate ≈ lat_g / speed (simplified)
            if speed > 2.0:
                # Estimated lateral drift in track-widths per second
                # Typical track width ~12m, so normalize
                lateral_drift_rate = (lat_g / max(speed, 5.0)) * dt * 0.1
                self._track_pos_estimate += lateral_drift_rate

                # Decay toward center when going straight (low steering)
                decay = 0.995 if abs(steer_norm) > 0.05 else 0.98
                self._track_pos_estimate *= decay
            else:
                # Car nearly stopped - assume centered
                self._track_pos_estimate *= 0.95

            estimated_pos = np.clip(self._track_pos_estimate, -1.0, 1.0)

        # Blend with forced track position if active
        if self._forced_blend > 0.01 and self._forced_track_pos is not None:
            b = self._forced_blend
            track_pos = b * self._forced_track_pos + (1.0 - b) * estimated_pos
            # Decay the forced override — also move the forced value
            # toward the estimate so it converges smoothly
            self._forced_blend = max(0.0, self._forced_blend - self._forced_decay_rate)
            self._forced_track_pos += (estimated_pos - self._forced_track_pos) * 0.01
        else:
            track_pos = estimated_pos
            self._forced_blend = 0.0

        return float(np.clip(track_pos, -1.0, 1.0))

    def _read_state(self) -> CarState:
        """Read current frame from iRacing shared memory."""
        ir = self._ir
        f = self._to_float

        speed = f(ir["Speed"])
        throttle = f(ir["Throttle"])
        brake = f(ir["Brake"])
        steering = f(ir["SteeringWheelAngle"])
        gear = f(ir["Gear"])
        rpm = f(ir["RPM"])
        lat_g = f(ir["LatAccel"])
        lon_g = f(ir["LongAccel"])
        yaw_rate = f(ir["YawRate"]) if ir["YawRate"] is not None else 0.0
        lap_dist_pct = f(ir["LapDistPct"])
        is_on_track = bool(f(ir["IsOnTrack"]))
        on_pit_road = bool(f(ir["OnPitRoad"] or 0))
        session_state = int(f(ir["SessionState"]))

        # Read track surface type (PlayerTrackSurface)
        surface_raw = ir["PlayerTrackSurface"]
        surface_type = int(f(surface_raw)) if surface_raw is not None else SURFACE_ON_TRACK

        # Estimate lateral track position from dynamics
        track_pos = self._estimate_track_pos(
            speed, steering, lat_g, yaw_rate, lap_dist_pct
        )

        # Override track_pos if surface indicates off-track
        if surface_type == SURFACE_OFF_TRACK:
            # Push estimate toward edge based on steering direction
            steer_sign = np.sign(steering) if abs(steering) > 0.01 else np.sign(self._track_pos_estimate)
            self._track_pos_estimate = np.clip(
                self._track_pos_estimate + steer_sign * 0.05, -1.0, 1.0
            )
            track_pos = self._track_pos_estimate

        # Update dynamic normalization constants
        self._speed_max = max(self._speed_max, speed * 1.05)
        self._rpm_max = max(self._rpm_max, rpm * 1.05)

        return CarState(
            speed=speed,
            throttle=throttle,
            brake=brake,
            steering=steering,
            gear=gear,
            rpm=rpm,
            lat_g=lat_g,
            lon_g=lon_g,
            yaw_rate=yaw_rate,
            lap_dist_pct=lap_dist_pct,
            track_pos=track_pos,
            surface_type=surface_type,
            is_on_track=is_on_track,
            on_pit_road=on_pit_road,
            session_active=(session_state > 0),
            timestamp=time.perf_counter(),
        )

    def _update_history(self, state: CarState):
        """Update ring buffer of recent actions for feature vector."""
        ptr = self._hist_ptr % self._history_len
        prev_ptr = (self._hist_ptr - 1) % self._history_len

        self._throttle_hist[ptr] = state.throttle
        self._brake_hist[ptr] = state.brake

        steer_norm = np.clip(state.steering / self._steering_lock, -1.0, 1.0)
        prev_steer = self._steering_hist[prev_ptr]
        steer_delta = float(steer_norm - prev_steer)

        self._steering_hist[ptr] = steer_norm
        self._steer_delta_hist[ptr] = np.clip(steer_delta, -0.3, 0.3)
        self._hist_ptr += 1

    def inject_bot_actions(self, throttle: float, brake: float, steering: float):
        """Inject the bot's output into the history buffer.

        During live driving the iRacing telemetry echoes vJoy inputs back,
        but there can be a delay.  Injecting the bot's own predictions
        ensures the history matches what the model expects.
        """
        with self._lock:
            ptr = (self._hist_ptr - 1) % self._history_len
            prev_ptr = (self._hist_ptr - 2) % self._history_len

            self._throttle_hist[ptr] = throttle
            self._brake_hist[ptr] = brake

            prev_steer = self._steering_hist[prev_ptr]
            steer_delta = float(steering - prev_steer)

            self._steering_hist[ptr] = steering
            self._steer_delta_hist[ptr] = np.clip(steer_delta, -0.3, 0.3)

    def get_state(self) -> Optional[CarState]:
        """Get latest state snapshot (thread-safe)."""
        with self._lock:
            return self._latest_state

    def build_state_vector(
        self,
        state: CarState,
        sequence_history: int = 15,
        speed_max: Optional[float] = None,
        steering_lock: Optional[float] = None,
        n_state_features: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build the full state vector for model inference.
        Mirrors the feature engineering in loader.py exactly.

        Args:
            n_state_features: If set, truncate state features to this count.
                This handles backwards compatibility with models trained
                before boundary features were added.

        Returns numpy array of shape (input_dim,)
        """
        sm = speed_max or self._speed_max
        sl = steering_lock or self._steering_lock

        # Current state features (must match STATE_FEATURES order in loader.py)
        speed_norm = np.clip(state.speed / sm, 0.0, 1.0)
        speed_delta = 0.0  # derived below from history
        gear_norm = np.clip(state.gear / 7.0, 0.0, 1.0)
        rpm_norm = np.clip(state.rpm / self._rpm_max, 0.0, 1.0)
        lat_g_norm = np.clip(state.lat_g / 40.0, -1.0, 1.0)
        lon_g_norm = np.clip(state.lon_g / 40.0, -1.0, 1.0)
        steer_norm = np.clip(state.steering / sl, -1.0, 1.0)
        steer_abs = abs(steer_norm)

        # Compute speed delta from recent history
        speed_delta = 0.0  # simplified; full implementation reads speed ring buffer

        heavy_braking = float(state.brake > 0.3 and lon_g_norm < -0.05)
        full_throttle = float(state.throttle > 0.95)

        # Track boundary awareness features
        track_pos_abs = abs(state.track_pos)
        near_edge = float(track_pos_abs > 0.75)       # approaching edge
        on_rumble = float(track_pos_abs > 0.90)        # likely on rumble strip
        track_pos_sign = np.sign(state.track_pos)      # which side (-1=left, +1=right)

        # Full 14-feature state vector (current layout in loader.py)
        all_state_features = np.array([
            state.lap_dist_pct,  # 0: lap_dist_pct
            speed_norm,          # 1: speed
            speed_delta,         # 2: speed_delta
            gear_norm,           # 3: gear
            rpm_norm,            # 4: rpm
            lat_g_norm,          # 5: lat_g
            lon_g_norm,          # 6: lon_g
            state.track_pos,     # 7: track_pos
            steer_abs,           # 8: steering_abs
            heavy_braking,       # 9: heavy_braking
            full_throttle,       # 10: full_throttle
            near_edge,           # 11: near_edge (added in track nav update)
            on_rumble,           # 12: on_rumble (added in track nav update)
            track_pos_sign,      # 13: track_pos_sign (added in track nav update)
        ], dtype=np.float32)

        # Truncate to match the model's expected state feature count.
        # Models trained before boundary features were added expect fewer.
        if n_state_features is not None and n_state_features < len(all_state_features):
            current_state = all_state_features[:n_state_features]
        else:
            current_state = all_state_features

        # Action history (newest first), matching HISTORY_ACTIONS order:
        # [throttle, brake, steering, steering_delta] × history_length
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

    @property
    def is_connected(self) -> bool:
        return self._connected
