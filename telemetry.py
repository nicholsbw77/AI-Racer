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
    speed: float = 0.0           # m/s
    throttle: float = 0.0        # 0-1
    brake: float = 0.0           # 0-1
    steering: float = 0.0        # radians (raw)
    gear: float = 0.0            # integer gear
    rpm: float = 0.0             # RPM
    lat_g: float = 0.0           # m/s² lateral acceleration
    lon_g: float = 0.0           # m/s² longitudinal acceleration
    lap_dist_pct: float = 0.0    # 0-1 position on track
    track_pos: float = 0.0       # lateral offset (-1 to +1)
    is_on_track: bool = False
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
        self._history_len = 20
        self._throttle_hist = np.zeros(self._history_len, dtype=np.float32)
        self._brake_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steering_hist = np.zeros(self._history_len, dtype=np.float32)
        self._steer_delta_hist = np.zeros(self._history_len, dtype=np.float32)
        self._hist_ptr = 0

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
        session_state = int(f(ir["SessionState"]))

        # Track position: iRacing provides this as fraction, center=0
        # Some SDK versions: use CarIdxTrackSurface or similar
        track_pos = 0.0  # TODO: derive from iRacing track width data

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
            lap_dist_pct=lap_dist_pct,
            track_pos=track_pos,
            is_on_track=is_on_track,
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
    ) -> np.ndarray:
        """
        Build the full state vector for model inference.
        Mirrors the feature engineering in loader.py exactly.

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
        ptr = (self._hist_ptr - 1) % self._history_len
        prev_ptr = (self._hist_ptr - 2) % self._history_len
        # Use throttle hist slot to estimate - or just set 0 if no history yet
        speed_delta = 0.0  # simplified; full implementation reads speed ring buffer

        heavy_braking = float(state.brake > 0.3 and lon_g_norm < -0.05)
        full_throttle = float(state.throttle > 0.95)

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
        ], dtype=np.float32)

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
