"""
agent/telemetry.py

Reads live telemetry from iRacing via pyirsdk shared memory.
Runs at up to 360Hz synchronized to iRacing's physics tick.

Enhanced with:
  - Expanded CarState with yaw rate, velocity components, tire data
  - Speed history ring buffer for acceleration computation
  - Track-aware state vector building with segment features
  - Integration hooks for TrackMap and SafetyController

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
  YawRate         - rad/s yaw rotation rate
  VelocityX/Y/Z   - m/s body-frame velocities
  PlayerTrackSurface - surface type (track/pit/grass etc)
  SessionState    - session status
  IsOnTrack       - bool
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


@dataclass
class CarState:
    """Snapshot of car state at a single time step."""
    # Core driving state
    speed: float = 0.0           # m/s
    throttle: float = 0.0        # 0-1
    brake: float = 0.0           # 0-1
    steering: float = 0.0        # radians (raw)
    gear: float = 0.0            # integer gear
    rpm: float = 0.0             # RPM

    # Dynamics
    lat_g: float = 0.0           # m/s² lateral acceleration
    lon_g: float = 0.0           # m/s² longitudinal acceleration
    yaw_rate: float = 0.0        # rad/s yaw rotation rate

    # Velocity components (body frame)
    velocity_x: float = 0.0      # m/s forward
    velocity_y: float = 0.0      # m/s lateral (slip indicator)

    # Track position
    lap_dist_pct: float = 0.0    # 0-1 position on track
    track_pos: float = 0.0       # lateral offset (-1 to +1)
    lap_number: int = 0          # current lap counter

    # Computed dynamics
    speed_delta: float = 0.0     # speed change from previous frame (m/s per tick)
    slip_angle: float = 0.0      # estimated slip angle (radians)

    # GPS position (decimal degrees; 0.0 if unavailable or session not started)
    gps_lat: float = 0.0
    gps_lon: float = 0.0

    # Status flags
    is_on_track: bool = False
    on_pit_road: bool = False
    session_active: bool = False

    # Session info
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
        self._history_len = 30  # increased from 20 for richer context
        self._throttle_hist = np.zeros(self._history_len, dtype=np.float32)
        self._brake_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steering_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steer_delta_hist = np.zeros(self._history_len, dtype=np.float32)
        self._speed_hist = np.zeros(self._history_len, dtype=np.float32)
        self._lat_g_hist = np.zeros(self._history_len, dtype=np.float32)
        self._lon_g_hist = np.zeros(self._history_len, dtype=np.float32)
        self._yaw_rate_hist = np.zeros(self._history_len, dtype=np.float32)
        self._hist_ptr = 0

        # Previous frame for delta computation
        self._prev_speed = 0.0

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
                # Try to get track length
                length_str = weekend.get("TrackLength", "")
                self._track_length_m = self._parse_track_length(length_str)
                logger.info(f"Track: {self._track_name} (ID: {self._track_id}, "
                            f"length: {self._track_length_m:.0f}m)")

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

    @staticmethod
    def _parse_track_length(length_str: str) -> float:
        """Parse track length from iRacing format like '4.01 km' or '2.49 mi'."""
        import re
        match = re.match(r"([\d.]+)\s*(km|mi)", str(length_str))
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit == "km":
                return value * 1000.0
            elif unit == "mi":
                return value * 1609.344
        return 0.0

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

    def get_track_length_m(self) -> float:
        """Return track length in meters (0 if unknown)."""
        return getattr(self, "_track_length_m", 0.0)

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
                if hasattr(self._ir, 'wait_for_data'):
                    if not self._ir.wait_for_data(timeout_ms=int(timeout)):
                        continue
                else:
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
        lap_dist_pct = f(ir["LapDistPct"])
        is_on_track = bool(f(ir["IsOnTrack"]))
        on_pit_road = bool(f(ir["OnPitRoad"] or 0))
        session_state = int(f(ir["SessionState"]))

        # Enhanced telemetry channels
        yaw_rate = f(ir["YawRate"]) if ir["YawRate"] is not None else 0.0
        velocity_x = f(ir["VelocityX"]) if ir["VelocityX"] is not None else speed
        velocity_y = f(ir["VelocityY"]) if ir["VelocityY"] is not None else 0.0
        lap_number = int(f(ir["Lap"])) if ir["Lap"] is not None else 0

        # Track position: try CarIdxLapDistPct for lateral position
        track_pos = 0.0
        try:
            track_surface = ir["PlayerTrackSurface"]
            if track_surface is not None:
                # PlayerTrackSurface gives us surface type but not lateral offset
                # We approximate track_pos from lateral velocity and steering
                if speed > 3.0:
                    # Slip angle approximation: atan2(vy, vx)
                    track_pos = np.clip(velocity_y / max(speed, 1.0) * 5.0, -1.0, 1.0)
        except Exception:
            pass

        # Compute speed delta
        speed_delta = speed - self._prev_speed
        self._prev_speed = speed

        # Compute slip angle
        slip_angle = 0.0
        if speed > 3.0:
            slip_angle = np.arctan2(velocity_y, max(abs(velocity_x), 0.1))

        # GPS position — (0.0, 0.0) during session init before car spawns
        gps_lat = f(ir["Lat"]) if ir["Lat"] is not None else 0.0
        gps_lon = f(ir["Lon"]) if ir["Lon"] is not None else 0.0

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
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            lap_dist_pct=lap_dist_pct,
            track_pos=track_pos,
            lap_number=lap_number,
            speed_delta=speed_delta,
            slip_angle=slip_angle,
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            is_on_track=is_on_track,
            on_pit_road=on_pit_road,
            session_active=(session_state > 0),
            timestamp=time.perf_counter(),
        )

    def _update_history(self, state: CarState):
        """Update ring buffer of recent state for feature vector."""
        ptr = self._hist_ptr % self._history_len
        prev_ptr = (self._hist_ptr - 1) % self._history_len

        self._throttle_hist[ptr] = state.throttle
        self._brake_hist[ptr] = state.brake

        steer_norm = np.clip(state.steering / self._steering_lock, -1.0, 1.0)
        prev_steer = self._steering_hist[prev_ptr]
        steer_delta = float(steer_norm - prev_steer)

        self._steering_hist[ptr] = steer_norm
        self._steer_delta_hist[ptr] = np.clip(steer_delta, -0.3, 0.3)

        # Enhanced history channels
        self._speed_hist[ptr] = state.speed
        self._lat_g_hist[ptr] = state.lat_g
        self._lon_g_hist[ptr] = state.lon_g
        self._yaw_rate_hist[ptr] = state.yaw_rate

        self._hist_ptr += 1

    def prewarm_history(self, throttle: float, brake: float, steering_normalized: float):
        """Fill the entire history ring buffer with the given values.

        Call this once when transitioning from pit exit to model control so the
        model sees a plausible recent-history (not a buffer full of pit-exit
        zeros) when it makes its first prediction.
        """
        with self._lock:
            self._throttle_hist[:] = throttle
            self._brake_hist[:] = brake
            self._steering_hist[:] = steering_normalized
            self._steer_delta_hist[:] = 0.0

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
        track_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build the full state vector for model inference.
        Mirrors the feature engineering in loader.py exactly.

        Args:
            state: Current car state
            sequence_history: Number of history frames to include
            speed_max: Override for speed normalization max
            steering_lock: Override for steering normalization
            track_features: Optional track-aware features from TrackMap

        Returns numpy array of shape (input_dim,)
        """
        sm = speed_max or self._speed_max
        sl = steering_lock or self._steering_lock

        # Current state features (must match STATE_FEATURES order in loader.py)
        speed_norm = np.clip(state.speed / sm, 0.0, 1.0)
        gear_norm = np.clip(state.gear / 7.0, 0.0, 1.0)
        rpm_norm = np.clip(state.rpm / self._rpm_max, 0.0, 1.0)
        lat_g_norm = np.clip(state.lat_g / 40.0, -1.0, 1.0)
        lon_g_norm = np.clip(state.lon_g / 40.0, -1.0, 1.0)
        steer_norm = np.clip(state.steering / sl, -1.0, 1.0)
        steer_abs = abs(steer_norm)

        # Compute speed delta from history
        speed_delta = 0.0
        if self._hist_ptr >= 2:
            curr_ptr = (self._hist_ptr - 1) % self._history_len
            prev_ptr = (self._hist_ptr - 2) % self._history_len
            if sm > 0:
                speed_delta = (self._speed_hist[curr_ptr] - self._speed_hist[prev_ptr]) / sm
            speed_delta = np.clip(speed_delta, -0.5, 0.5)

        heavy_braking = float(state.brake > 0.3 and lon_g_norm < -0.05)
        full_throttle = float(state.throttle > 0.95)

        # Enhanced features
        yaw_rate_norm = np.clip(state.yaw_rate / 3.0, -1.0, 1.0)  # normalize ±3 rad/s
        slip_angle_norm = np.clip(state.slip_angle / 0.5, -1.0, 1.0)  # normalize ±0.5 rad

        current_state = np.array([
            state.lap_dist_pct,  # lap_dist_pct
            speed_norm,          # speed
            speed_delta,         # speed_delta
            gear_norm,           # gear
            rpm_norm,            # rpm
            lat_g_norm,          # lat_g
            lon_g_norm,          # lon_g
            state.track_pos,     # track_pos
            steer_abs,           # steering_abs
            heavy_braking,       # heavy_braking
            full_throttle,       # full_throttle
            yaw_rate_norm,       # yaw_rate (NEW)
            slip_angle_norm,     # slip_angle (NEW)
        ], dtype=np.float32)

        # Action history (newest first), matching HISTORY_ACTIONS order:
        # [throttle, brake, steering, steering_delta] x history_length
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

        # Combine: current state + history + optional track features
        parts = [current_state, history_flat]
        if track_features is not None:
            parts.append(track_features)

        return np.concatenate(parts)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def history_length(self) -> int:
        return self._history_len
