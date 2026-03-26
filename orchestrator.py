"""
orchestrator.py

Main entry point for the live iRacing AI bot.

Ties together:
  - TelemetryReader (pyirsdk @ 360Hz)
  - DrivingAgent (model inference + track-aware features)
  - SafetyController (multi-layer safety system)
  - TrackMap / LiveTracker (GPS-free track positioning)
  - VJoyController (virtual controller output)

Usage:
  python orchestrator.py --track sebring_2023 --car mx5_cup
  python orchestrator.py --auto           # auto-detect track/car from iRacing session
  python orchestrator.py --mock           # dry run without iRacing/vJoy (for testing)

Safety features:
  - Ctrl+C to stop at any time -> releases all inputs instantly
  - 10-layer safety controller (see safety.py)
  - Automatic input release if iRacing disconnects
  - Track boundary monitoring via TrackMap
  - Spin/collision detection and recovery
  - Lap time watchdog: kills bot if lap too slow
"""

import sys
import os
import json
import time
import signal
import logging
import argparse
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from telemetry import TelemetryReader, CarState
from inference import DrivingAgent
from controller import VJoyController, MockController
from safety import SafetyController, SafetyAction, SafetyVerdict
from track_map import TrackMap, LiveTracker, SegmentType


def setup_logging() -> Path:
    """Configure logging to both console and a timestamped log file."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"session_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(file_handler)

    return log_file


log_file = setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _resolve_sequence_history(cfg)
    return cfg


def _resolve_sequence_history(cfg: dict):
    train_cfg = cfg["training"]
    if train_cfg.get("sequence_history") == "auto":
        hz = int(train_cfg.get("data_hz", 60))
        ms = float(train_cfg.get("context_window_ms", 250))
        train_cfg["sequence_history"] = max(1, round(ms / 1000.0 * hz))
        logger.info(
            f"sequence_history=auto resolved to {train_cfg['sequence_history']} "
            f"frames ({ms:.0f}ms @ {hz}Hz)"
        )


PIT_EXIT_CONFIG_DIR = Path(__file__).parent / "pit_exit_configs"

PIT_EXIT_DEFAULTS = {
    "straight_duration": 5.0,
    "turn_angle": -5.0,
    "turn_duration": 1.2,
    "turn_throttle": 0.2,
    "straight_throttle": 0.2,
    "ramp_duration": 0.5,
    "cruise_until_lap_pct": 0.04,
    "cruise_throttle": 0.2,
    "pit_exit_track_pos": 0.0,
    "stall_pullout_left_dur": 0.0,
    "stall_pullout_right_dur": 0.0,
    "stall_pullout_steer": 0.35,
    "stall_pullout_throttle": 0.2,
}


class PitExitPhase(Enum):
    IDLE = auto()
    STALL_PULLOUT_LEFT = auto()
    STALL_PULLOUT_RIGHT = auto()
    STRAIGHT = auto()
    TURN = auto()
    RAMP = auto()
    CRUISE = auto()
    COMPLETE = auto()


class PitExitExecutor:
    """
    Phase-based state machine for driving out of the pits.

    Reads a per-track JSON config and steps through:
      STALL_PULLOUT_LEFT -> STALL_PULLOUT_RIGHT -> STRAIGHT -> TURN -> RAMP -> CRUISE -> COMPLETE
    Phases with zero duration are skipped.
    Returns (throttle, brake, steering) each tick, or None when complete (handoff to model).
    """

    # Phase order for advancing
    PHASE_ORDER = [
        PitExitPhase.STALL_PULLOUT_LEFT,
        PitExitPhase.STALL_PULLOUT_RIGHT,
        PitExitPhase.STRAIGHT,
        PitExitPhase.TURN,
        PitExitPhase.RAMP,
        PitExitPhase.CRUISE,
        PitExitPhase.COMPLETE,
    ]

    def __init__(self, config_dir: Path = PIT_EXIT_CONFIG_DIR):
        self.config_dir = config_dir
        self.config: dict = dict(PIT_EXIT_DEFAULTS)
        self.phase = PitExitPhase.IDLE
        self.phase_start_time = 0.0
        self.activate_time = 0.0
        self.turn_steer_at_ramp_start = 0.0
        self.start_lap_pct = 0.0  # lap_pct when pit exit started
        self._crossed_sf = False   # have we crossed start/finish during exit?
        self._active = False
        self._config_loaded = False

    def load_config(self, combo_name: str) -> bool:
        """Load pit exit config for a given track/car combo. Returns True if found."""
        self.config_dir.mkdir(exist_ok=True)

        # Try exact match first, then fuzzy keyword match
        path = self.config_dir / f"{combo_name}.json"
        if not path.exists():
            path = self._fuzzy_match(combo_name)

        if path and path.exists():
            with open(path) as f:
                loaded = json.load(f)
            self.config = {**PIT_EXIT_DEFAULTS, **loaded}
            self._config_loaded = True
            logger.info(f"Pit exit config loaded: {path.name}")
            return True

        logger.warning(
            f"No pit exit config for '{combo_name}'. "
            f"Using defaults. Run pit_exit_gui.py to configure."
        )
        self.config = dict(PIT_EXIT_DEFAULTS)
        self._config_loaded = False
        return False

    def _fuzzy_match(self, combo_name: str) -> Optional[Path]:
        """Try to match combo name to existing configs by keyword overlap."""
        if not self.config_dir.exists():
            return None
        keywords = {k for k in combo_name.split("_") if len(k) > 2}
        best_path = None
        best_score = 0
        for p in self.config_dir.glob("*.json"):
            score = sum(1 for kw in keywords if kw in p.stem)
            if score > best_score:
                best_score = score
                best_path = p
        if best_score > 0:
            logger.info(f"Pit exit config fuzzy match: {best_path.name} (score {best_score})")
            return best_path
        return None

    def activate(self, lap_dist_pct: float = 0.0):
        """Called when car enters pit road. Resets state machine to first phase."""
        now = time.perf_counter()
        self._active = True
        self.activate_time = now
        self.start_lap_pct = lap_dist_pct
        self._crossed_sf = False
        # Start at first non-zero phase
        self.phase = PitExitPhase.IDLE
        self._advance_phase(now)
        logger.info(f"Pit exit executor ACTIVATED at lap_pct={lap_dist_pct:.3f} - phase: {self.phase.name}")

    def deactivate(self):
        """Force deactivate (e.g., session ended)."""
        self._active = False
        self.phase = PitExitPhase.IDLE

    @property
    def is_active(self) -> bool:
        return self._active

    def tick(self, state: CarState) -> Optional[Tuple[float, float, float]]:
        """
        Called each frame while pit exit is active.
        Returns (throttle, brake, steering) or None if complete.
        """
        if not self._active:
            return None

        if self.phase == PitExitPhase.COMPLETE:
            logger.info("Pit exit complete -- handing off to model")
            self._active = False
            self.phase = PitExitPhase.IDLE
            return None

        now = time.perf_counter()
        elapsed = now - self.phase_start_time
        cfg = self.config

        throttle = 0.0
        brake = 0.0
        steering = 0.0

        # Track S/F crossing at every tick (car may cross during pre-cruise phases)
        if state.lap_dist_pct < 0.5 and self.start_lap_pct > 0.5:
            self._crossed_sf = True

        if self.phase == PitExitPhase.STALL_PULLOUT_LEFT:
            duration = cfg["stall_pullout_left_dur"]
            if elapsed >= duration:
                self._advance_phase(now)
                return self.tick(state)  # recurse into next phase
            throttle = cfg["stall_pullout_throttle"]
            # Positive value → after _steer_sign(-1) → negative vJoy → LEFT turn
            steering = abs(cfg["stall_pullout_steer"])

        elif self.phase == PitExitPhase.STALL_PULLOUT_RIGHT:
            duration = cfg["stall_pullout_right_dur"]
            if elapsed >= duration:
                self._advance_phase(now)
                return self.tick(state)
            throttle = cfg["stall_pullout_throttle"]
            # Negative value → after _steer_sign(-1) → positive vJoy → RIGHT turn
            steering = -abs(cfg["stall_pullout_steer"])

        elif self.phase == PitExitPhase.STRAIGHT:
            duration = cfg["straight_duration"]
            if elapsed >= duration:
                self._advance_phase(now)
                return self.tick(state)
            throttle = cfg["straight_throttle"]
            steering = 0.0

        elif self.phase == PitExitPhase.TURN:
            duration = cfg["turn_duration"]
            if elapsed >= duration:
                # Save turn steering for ramp interpolation
                self.turn_steer_at_ramp_start = cfg["turn_angle"] / 30.0
                self._advance_phase(now)
                return self.tick(state)
            throttle = cfg["turn_throttle"]
            # Normalize degrees to -1..+1 range (assume ~30 deg max lock)
            steering = cfg["turn_angle"] / 30.0
            steering = max(-1.0, min(1.0, steering))

        elif self.phase == PitExitPhase.RAMP:
            duration = cfg["ramp_duration"]
            if duration <= 0 or elapsed >= duration:
                self._advance_phase(now)
                return self.tick(state)
            # Linearly interpolate steering from turn angle back to 0
            progress = elapsed / duration
            steering = self.turn_steer_at_ramp_start * (1.0 - progress)
            throttle = cfg["cruise_throttle"]

        elif self.phase == PitExitPhase.CRUISE:
            # Cruise until we reach the target lap_dist_pct
            target_pct = cfg["cruise_until_lap_pct"]
            min_cruise = 1.0

            # Check exit condition — handle wrap-around
            # True wrap-around only when pit lane is near END of track (start_lap_pct > 0.5)
            # and target is near START (target_pct < 0.5). e.g. pit at 0.948, target 0.04.
            # If pit exits near the start (start_lap_pct < 0.5) and target < start,
            # the car already passed the target — hand off after min_cruise.
            reached_target = False
            if target_pct <= 0:
                # cruise_until_lap_pct=0 means: hand off to model as soon as
                # the car is off pit road (with a 1s minimum to settle).
                reached_target = not state.on_pit_road and elapsed >= min_cruise
            elif elapsed >= min_cruise:
                if self.start_lap_pct > 0.5 and target_pct < 0.5:
                    # True wrap-around: pit near end of track, target near start
                    # Must cross S/F first, then reach target
                    reached_target = self._crossed_sf and state.lap_dist_pct >= target_pct
                elif state.lap_dist_pct >= target_pct:
                    # Normal case: target is ahead of current position
                    reached_target = True
                elif state.lap_dist_pct < self.start_lap_pct and self.start_lap_pct <= target_pct:
                    # Car went backwards past start somehow — safety fallback
                    reached_target = True
                else:
                    # Car already past target when cruise began (pit exit past target point)
                    # e.g. pit exits at 0.089, target was 0.002 — already passed it
                    reached_target = elapsed >= min_cruise * 2

            if reached_target:
                self._advance_phase(now)
                return self.tick(state)
            # Safety timeout
            if elapsed >= 30.0:
                logger.warning("Pit exit cruise phase timed out (30s)")
                self._advance_phase(now)
                return self.tick(state)
            throttle = cfg["cruise_throttle"]
            # cruise_steering: small constant steer to follow curved pit roads
            # (e.g. oval tracks where pit road curves through turns 1-2).
            # Positive = left turn after steer_invert; negative = right turn.
            # Defaults to 0.0 (straight pit road / road course).
            steering = float(cfg.get("cruise_steering", 0.0))

        # Gear management: 1st below 5 m/s, 2nd above
        # (gear shifts handled by orchestrator, we just return controls)

        return throttle, brake, steering

    def _advance_phase(self, now: float):
        """Move to the next phase, skipping zero-duration phases."""
        if self.phase == PitExitPhase.IDLE:
            idx = 0
        else:
            try:
                idx = self.PHASE_ORDER.index(self.phase) + 1
            except ValueError:
                idx = len(self.PHASE_ORDER) - 1

        while idx < len(self.PHASE_ORDER):
            candidate = self.PHASE_ORDER[idx]

            # Check if this phase should be skipped (zero duration)
            skip = False
            if candidate == PitExitPhase.STALL_PULLOUT_LEFT:
                skip = self.config.get("stall_pullout_left_dur", 0.0) <= 0
            elif candidate == PitExitPhase.STALL_PULLOUT_RIGHT:
                skip = self.config.get("stall_pullout_right_dur", 0.0) <= 0
            elif candidate == PitExitPhase.RAMP:
                skip = self.config.get("ramp_duration", 0.0) <= 0

            if skip:
                idx += 1
                continue

            self.phase = candidate
            self.phase_start_time = now
            logger.info(f"Pit exit phase: {self.phase.name}")
            return

        # Ran out of phases
        self.phase = PitExitPhase.COMPLETE
        self.phase_start_time = now

    def get_status(self) -> str:
        """Return human-readable status for logging."""
        if not self._active:
            return "IDLE"
        elapsed = time.perf_counter() - self.phase_start_time
        return f"{self.phase.name} ({elapsed:.1f}s)"


class BotOrchestrator:
    """
    Main control loop: read telemetry -> predict -> safety check -> send inputs.
    Runs at target 360Hz synchronized to iRacing physics tick.
    """

    def __init__(self, cfg: dict, mock: bool = False):
        self.cfg = cfg
        self.mock = mock
        self._running = False

        # Components
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Inference device: {device}")

        self.telemetry = TelemetryReader(
            target_hz=cfg["inference"]["loop_hz"]
        )
        self.agent = DrivingAgent(cfg, device=device)
        self.safety = SafetyController(cfg)

        if mock:
            self.controller = MockController()
        else:
            self.controller = VJoyController(device_id=1)

        # Track positioning
        self.tracker: Optional[LiveTracker] = None

        # Metrics
        self._lap_count = 0
        self._frame_count = 0
        self._last_lap_dist = 0.0
        self._session_start = 0.0
        self._loop_times = []  # for Hz monitoring

        # Pit exit executor (phase-based, config-driven)
        self._pit_exit = PitExitExecutor()

        # Steering axis sign: iRacing records SteeringWheelAngle as negative for
        # RIGHT turns.  If the vJoy axis maps negative→LEFT (flipped), set
        # steer_invert: true in config.yaml to correct it.
        self._steer_sign = -1.0 if cfg.get("inference", {}).get("steer_invert", False) else 1.0
        if self._steer_sign < 0:
            logger.info("Steering axis: INVERTED (steer_invert=true in config)")

        # Map-guided launch: after pit exit, follow track-map reference values
        # until car reaches minimum race speed before handing to the model.
        # The model was trained at race speed; handing off at 5 m/s puts it
        # out-of-distribution and causes random outputs.
        self._map_launch_active = False
        self._map_launch_min_speed_ms = float(
            cfg.get("inference", {}).get("launch_min_speed_ms", 35.0)
        )

        # Safety stats
        self._safety_kills = 0
        self._safety_interventions = 0

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _match_checkpoint(self, raw_combo: str) -> str:
        """Match auto-detected combo to an existing checkpoint folder."""
        ckpt_dir = Path("checkpoints")
        if not ckpt_dir.exists():
            return raw_combo

        candidates = [d.name for d in ckpt_dir.iterdir()
                       if d.is_dir() and (d / "best.pt").exists()]
        if not candidates:
            return raw_combo

        keywords = set(raw_combo.split("_"))
        keywords = {k for k in keywords if len(k) > 2}

        best_match = raw_combo
        best_score = 0

        for cand in candidates:
            score = sum(1 for kw in keywords if kw in cand)
            if score > best_score:
                best_score = score
                best_match = cand

        if best_score > 0:
            logger.info(f"Checkpoint match: '{best_match}' (score {best_score}/{len(keywords)} keywords)")
        else:
            logger.warning(f"No checkpoint matched for '{raw_combo}'. Available: {candidates}")

        return best_match

    def start(self, combo_name: Optional[str]):
        """Start the bot for a given track/car combo (None = auto-detect)."""

        # Connect to iRacing first (needed for auto-detect)
        if not self.mock:
            if not self.telemetry.connect():
                logger.warning("iRacing not detected. Waiting...")
                while not self.telemetry.connect():
                    logger.info("Retrying iRacing connection in 5s...")
                    time.sleep(5)

        # Auto-detect combo from session
        if combo_name is None:
            raw_combo = self.telemetry.get_combo_name()
            logger.info(f"Auto-detected session: {raw_combo}")
            combo_name = self._match_checkpoint(raw_combo)
            logger.info(f"Matched checkpoint: {combo_name}")

        logger.info(f"Starting bot for: {combo_name}")

        # Connect controller
        if not self.controller.connect():
            logger.error("Failed to connect controller. Exiting.")
            return

        # Load model
        if not self.agent.load_checkpoint(combo_name):
            if not self.mock:
                logger.error(f"No checkpoint for '{combo_name}'. Train first.")
                logger.error(f"Run: python train.py --combo {combo_name}")
                return
            else:
                logger.warning("Mock mode: running without model (random outputs)")

        # Load pit exit config for this combo
        self._pit_exit.load_config(combo_name)

        # Apply checkpoint norm constants to the telemetry reader so that
        # build_state_vector uses the exact same scales seen during training.
        if self.agent.norm:
            n = self.agent.norm
            if n.get("speed_max_ms"):
                self.telemetry._speed_max = float(n["speed_max_ms"])
            if n.get("steering_lock_radians"):
                self.telemetry._steering_lock = float(n["steering_lock_radians"])
            if n.get("rpm_max"):
                self.telemetry._rpm_max = float(n["rpm_max"])
            logger.info(
                f"Inference norms applied: speed_max={self.telemetry._speed_max:.2f}m/s  "
                f"steer_lock={self.telemetry._steering_lock:.4f}rad  "
                f"rpm_max={self.telemetry._rpm_max:.0f}"
            )

        # Load track map if available
        self.agent.load_track_map(combo_name)
        if self.agent.has_track_map:
            self.tracker = LiveTracker(
                self.agent.track_map,
                lookahead=self.cfg.get("track", {}).get("lookahead_segments", 5),
            )
            # Set personal best for lap watchdog
            if self.agent.track_map.personal_best_s > 0:
                self.safety.set_personal_best(self.agent.track_map.personal_best_s)

        # Start telemetry background thread
        self.telemetry.start()

        logger.info("Bot active. Press Ctrl+C to stop.")
        logger.info(f"Target loop rate: {self.cfg['inference']['loop_hz']}Hz")
        logger.info(f"Safety controller: ACTIVE (10 layers)")
        logger.info(f"Track map: {'LOADED' if self.agent.has_track_map else 'NOT AVAILABLE'}")

        self._running = True
        self._session_start = time.perf_counter()
        self.safety.reset()
        self._run_loop(combo_name)

    def _run_loop(self, combo_name: str):
        """Main control loop."""
        target_period = 1.0 / self.cfg["inference"]["loop_hz"]
        sequence_history = self.cfg["training"]["sequence_history"]

        last_track = ""
        last_car = ""

        while self._running:
            loop_start = time.perf_counter()

            # Get latest telemetry
            state = self.telemetry.get_state()

            if state is None:
                time.sleep(0.001)
                continue

            # Check if session changed (track/car switch)
            if state.track_id and state.track_id != last_track:
                logger.info(f"Track changed: {state.track_id}")
                last_track = state.track_id
                new_combo = f"{state.track_id}_{state.car_id}"
                if new_combo != combo_name:
                    logger.info(f"Auto-switching model to {new_combo}")
                    self.agent.load_checkpoint(new_combo)
                    self._pit_exit.load_config(new_combo)
                    self.agent.load_track_map(new_combo)
                    if self.agent.has_track_map:
                        self.tracker = LiveTracker(
                            self.agent.track_map,
                            lookahead=self.cfg.get("track", {}).get("lookahead_segments", 5),
                        )
                    self.safety.reset()
                    combo_name = new_combo

            # Debug: log state flags periodically
            if self._frame_count % 300 == 0:
                logger.info(
                    f"State: on_track={state.is_on_track} "
                    f"pit_road={state.on_pit_road} "
                    f"session_active={state.session_active} "
                    f"speed={state.speed:.1f}m/s "
                    f"gear={state.gear:.0f} "
                    f"lap_pct={state.lap_dist_pct:.3f} "
                    f"yaw={state.yaw_rate:.2f}rad/s"
                )

            # --- Pit exit executor (phase-based) ---
            if (state.on_pit_road or self._pit_exit.is_active) and state.session_active:
                if not self._pit_exit.is_active and state.on_pit_road:
                    self._pit_exit.activate(state.lap_dist_pct)

                pit_result = self._pit_exit.tick(state)
                if pit_result is None:
                    # Pit exit just completed — decide whether to activate the
                    # map-guided launch or hand straight to the model.
                    pit_cfg = self._pit_exit.config
                    if "launch_min_speed_ms" in pit_cfg:
                        self._map_launch_min_speed_ms = float(pit_cfg["launch_min_speed_ms"])

                    skip_launch = self._map_launch_min_speed_ms <= 0 or not self.agent.has_track_map

                    if skip_launch:
                        # launch_min_speed_ms=0 in pit config means: skip map
                        # launch entirely and hand off directly to the model.
                        # Use this when the pit exit speed is already close to
                        # the training distribution (e.g. slow road-course corners
                        # right after pit exit).
                        steer_lock = self.telemetry._steering_lock
                        steer_norm = float(np.clip(state.steering / steer_lock, -1.0, 1.0))
                        self.telemetry.prewarm_history(float(state.throttle), 0.0, steer_norm)
                        logger.info(
                            f"Pit exit complete — direct model handoff "
                            f"pct={state.lap_dist_pct:.3f} speed={state.speed:.1f}m/s"
                        )
                    else:
                        self._map_launch_active = True
                        logger.info(
                            f"Map-guided launch activated at pct={state.lap_dist_pct:.3f} "
                            f"speed={state.speed:.1f}m/s — target >{self._map_launch_min_speed_ms:.0f}m/s"
                        )
                if pit_result is not None:
                    throttle, brake, steering = pit_result

                    # Gear management during pit exit
                    # Stay in 2nd once rolling — downshifting mid-exit stalls the car
                    if state.speed < 2.0 and state.gear < 1:
                        target_gear = 1
                    elif state.speed > 4.0:
                        target_gear = 2
                    else:
                        target_gear = max(1, int(state.gear))
                    if target_gear != self.controller._current_gear and target_gear > 0:
                        self.controller.shift_to(target_gear)

                    # Send pit exit controls directly to controller
                    # Skip the full safety stack during pit exit — it causes
                    # false spin detection from noisy G-force telemetry and
                    # kills throttle. Only enforce pit speed governor.
                    pit_limit = self.safety.config.pit_speed_limit_ms
                    if state.on_pit_road and state.speed > pit_limit * 1.05:
                        throttle = 0.0
                        brake = min(0.5, (state.speed - pit_limit) / pit_limit)
                    self.controller.set_inputs(throttle, brake, steering * self._steer_sign)

                    if self._frame_count % 60 == 0:
                        logger.info(
                            f"PIT EXIT [{self._pit_exit.get_status()}]: "
                            f"thr={throttle:.2f} steer={steering:.3f} "
                            f"speed={state.speed:.1f}m/s"
                        )

                    self._frame_count += 1
                    self._sleep_remaining(loop_start, target_period)
                    continue

            # --- Map-guided launch phase ---
            # After pit exit, follow track-map reference values until the car
            # reaches minimum race speed.  This prevents the model from seeing
            # an out-of-distribution slow-speed start that causes random outputs.
            if self._map_launch_active and state.session_active and state.is_on_track:
                # Prefer GPS-based segment lookup: more accurate when lap_dist_pct
                # is unreliable (e.g. pit road exit, early-lap positioning).
                seg = self.agent.track_map.find_segment_by_gps(state.gps_lat, state.gps_lon)
                if seg is None:
                    seg = self.agent.track_map.get_segment(state.lap_dist_pct)

                # Exit condition: at race speed on a non-braking segment
                at_speed = state.speed >= self._map_launch_min_speed_ms
                safe_seg = seg is None or seg.segment_type not in (
                    SegmentType.BRAKING_ZONE, SegmentType.CORNER_ENTRY, SegmentType.CORNER_APEX
                )
                if at_speed and safe_seg:
                    self._map_launch_active = False
                    steer_lock = self.telemetry._steering_lock
                    steer_norm = float(np.clip(state.steering / steer_lock, -1.0, 1.0))
                    self.telemetry.prewarm_history(float(state.throttle), 0.0, steer_norm)
                    logger.info(
                        f"Map launch complete — model handoff at "
                        f"pct={state.lap_dist_pct:.3f} speed={state.speed:.1f}m/s"
                    )
                    # Fall through to model inference this frame
                else:
                    # Drive using track-map reference values.
                    # Use ref_steering for the racing line.
                    # Boost throttle to build speed quickly; respect corner braking.
                    seg_type = seg.segment_type if seg else None
                    is_straight_seg = seg_type in (
                        SegmentType.STRAIGHT,
                        SegmentType.ACCELERATION_ZONE,
                        SegmentType.UNKNOWN,
                        None,
                    )
                    ref_steer = seg.ref_steering if seg else 0.0
                    ref_brake = seg.ref_brake if seg else 0.0
                    ref_thr   = seg.ref_throttle if seg else 0.5
                    ref_spd   = seg.ref_speed if seg else self._map_launch_min_speed_ms

                    # On straight/unknown segments ref_steer can be bad data
                    # (e.g. pit-exit merge area). Cap it so the car doesn't spin.
                    if is_straight_seg:
                        steer_cap = float(
                            self.cfg.get("inference", {}).get("launch_straight_steer_cap", 0.08)
                        )
                        ref_steer = max(-steer_cap, min(steer_cap, ref_steer))

                    # Scale steering by (speed / ref_speed) ^ power.
                    # power=1.0 → linear (too little steer at low speed)
                    # power=0.5 → square-root (more steer at low speed)
                    # power=0.0 → always full ref_steer
                    # Tunable via inference.launch_steer_power in config.yaml
                    if ref_spd > 0.5:
                        _power = float(
                            self.cfg.get("inference", {}).get("launch_steer_power", 0.5)
                        )
                        speed_ratio = min(1.0, state.speed / ref_spd)
                        ref_steer = ref_steer * (speed_ratio ** _power)

                    # Spin detection: if yaw rate is very high the car is already
                    # rotating — counter-steer and cut throttle to recover.
                    yaw_spin_thresh = float(
                        self.cfg.get("inference", {}).get("launch_spin_yaw_thresh", 1.5)
                    )
                    is_spinning = abs(state.yaw_rate) > yaw_spin_thresh
                    if is_spinning:
                        # Counter-steer: oppose the spin direction
                        counter = -float(np.sign(state.yaw_rate)) * 0.25
                        ref_steer = counter
                        throttle = 0.1
                        brake = 0.0
                    else:
                        # In a braking zone / corner: respect reference values so the
                        # car doesn't overpower at low speed.  Only boost on straights.
                        steer_mag = abs(ref_steer)
                        needs_brake = (
                            seg is not None
                            and seg.segment_type in (
                                SegmentType.BRAKING_ZONE,
                                SegmentType.CORNER_ENTRY,
                                SegmentType.CORNER_APEX,
                            )
                            and state.speed > ref_spd * 0.9
                        )
                        if needs_brake:
                            # Active braking zone — follow reference exactly
                            throttle = ref_thr
                            brake = ref_brake
                        elif steer_mag > 0.30:
                            # Still in a corner (exit or apex) — use reference throttle,
                            # boosting beyond it risks spinning at below-reference speed
                            throttle = ref_thr
                            brake = 0.0
                        elif steer_mag > 0.10:
                            # Gentle curve / unwinding — modest boost
                            throttle = max(ref_thr, 0.5)
                            brake = 0.0
                        else:
                            # Essentially straight — full throttle to build speed
                            throttle = max(ref_thr, 0.8)
                            brake = 0.0

                    self.controller.set_inputs(throttle, brake, ref_steer * self._steer_sign)

                    if self._frame_count % 60 == 0:
                        seg_name = seg_type.name if seg_type else "?"
                        spin_flag = " [SPIN]" if is_spinning else ""
                        logger.info(
                            f"LAUNCH [{seg_name}{spin_flag} pct={state.lap_dist_pct:.3f}]: "
                            f"thr={throttle:.2f} brk={brake:.2f} "
                            f"steer={ref_steer:.3f} speed={state.speed:.1f}m/s "
                            f"(ref={ref_spd:.1f}m/s yaw={state.yaw_rate:.2f}rad/s)"
                        )

                    self._frame_count += 1
                    self._sleep_remaining(loop_start, target_period)
                    continue

            # --- Safety pre-check: session/track gates ---
            if not state.session_active or (not state.is_on_track and not state.on_pit_road):
                verdict = self.safety.check(
                    state, 0.0, 0.0, 0.0,
                    is_on_track=state.is_on_track,
                    on_pit_road=state.on_pit_road,
                    session_active=state.session_active,
                )
                self._apply_verdict(verdict, state)
                self._frame_count += 1
                time.sleep(0.01)
                continue

            # --- Track positioning update ---
            track_features = None
            boundary_pct = 0.0
            if self.tracker:
                track_ctx = self.tracker.update(
                    state.lap_dist_pct,
                    state.speed,
                    state.steering,
                    state.track_pos,
                )
                track_features = track_ctx["track_features"]
                boundary_pct = track_ctx["boundary_pct"]

                if track_ctx["lap_crossed"]:
                    self.safety.reset_lap()
                    self._lap_count += 1
                    elapsed = time.perf_counter() - self._session_start
                    logger.info(
                        f"Lap {self._lap_count} complete  "
                        f"(session time: {elapsed/60:.1f}min, "
                        f"total frames: {self._frame_count:,})"
                    )
            else:
                # Fallback lap detection without tracker
                self._detect_lap_crossing(state)
                # Try to get track features from agent
                track_features = self.agent.get_track_features(
                    state.lap_dist_pct, state.speed
                )
                boundary_pct = self.agent.get_boundary_proximity(
                    state.lap_dist_pct, state.track_pos
                )

            # --- Model-driven control ---
            state_vec = self.telemetry.build_state_vector(
                state,
                sequence_history=sequence_history,
                track_features=track_features,
            )

            # Run inference
            throttle, brake, steering = self.agent.predict(
                state_vec,
                car_speed_ms=state.speed,
                lap_dist_pct=state.lap_dist_pct,
            )

            # --- Safety controller ---
            verdict = self.safety.check(
                state, throttle, brake, steering,
                track_boundary_pct=boundary_pct,
                is_on_track=state.is_on_track,
                on_pit_road=state.on_pit_road,
                session_active=state.session_active,
            )

            self._apply_verdict(verdict, state)

            # Debug: log model outputs periodically
            if self._frame_count % 60 == 0:
                safety_tag = ""
                if verdict.action != SafetyAction.PASS:
                    safety_tag = f" [SAFETY:{verdict.layer}]"
                logger.info(
                    f"Output: thr={verdict.throttle:.3f} brk={verdict.brake:.3f} "
                    f"steer={verdict.steering:.3f} speed={state.speed:.1f}"
                    f"{safety_tag}"
                )

            # Inject bot outputs into history buffer for next prediction
            self.telemetry.inject_bot_actions(
                verdict.throttle, verdict.brake, verdict.steering
            )

            # Gear shifting: disabled during normal driving (car uses automatic gearbox).
            # Shifting is only active during the pit exit phase (see pit exit block above).

            # Metrics
            self._frame_count += 1

            loop_elapsed = time.perf_counter() - loop_start
            self._loop_times.append(loop_elapsed)

            # Log performance every 360 frames
            if self._frame_count % 360 == 0:
                self._log_stats()

            # Yield remaining time in period
            self._sleep_remaining(loop_start, target_period)

    def _apply_verdict(self, verdict: SafetyVerdict, state: CarState):
        """Apply a safety verdict to the controller."""
        if verdict.action == SafetyAction.KILL:
            self.controller.release()
            self._safety_kills += 1
            if self._safety_kills % 10 == 1:
                logger.warning(f"Safety KILL: {verdict.reason}")
        elif verdict.action == SafetyAction.OVERRIDE:
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering * self._steer_sign)
            self._safety_interventions += 1
        elif verdict.action == SafetyAction.ATTENUATE:
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering * self._steer_sign)
            self._safety_interventions += 1
        else:
            # PASS — send through
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering * self._steer_sign)

    def _sleep_remaining(self, loop_start: float, target_period: float):
        """Sleep for remaining time in the loop period."""
        elapsed = time.perf_counter() - loop_start
        remaining = target_period - elapsed
        if remaining > 0.0001:
            time.sleep(remaining)

    def _detect_lap_crossing(self, state: CarState):
        """Detect lap completion by watching lap_dist_pct wrap around."""
        if self._last_lap_dist > 0.95 and state.lap_dist_pct < 0.05:
            self._lap_count += 1
            self.safety.reset_lap()
            elapsed = time.perf_counter() - self._session_start
            logger.info(
                f"Lap {self._lap_count} complete  "
                f"(session time: {elapsed/60:.1f}min, "
                f"total frames: {self._frame_count:,})"
            )
        self._last_lap_dist = state.lap_dist_pct

    def _log_stats(self):
        """Log loop timing statistics."""
        if not self._loop_times:
            return
        recent = self._loop_times[-360:]
        avg_ms = np.mean(recent) * 1000
        max_ms = np.max(recent) * 1000
        actual_hz = 1.0 / np.mean(recent) if np.mean(recent) > 0 else 0
        self._loop_times = self._loop_times[-360:]

        safety_summary = self.safety.get_summary()
        logger.info(
            f"Loop: {actual_hz:.0f}Hz  avg={avg_ms:.2f}ms  "
            f"max={max_ms:.2f}ms  laps={self._lap_count}  "
            f"safety_kills={safety_summary['total_kills']}  "
            f"incidents={safety_summary['incident_count']}"
        )

    def _shutdown_handler(self, signum, frame):
        """Handle Ctrl+C / SIGTERM gracefully."""
        logger.info("\nShutdown signal received...")
        self._running = False
        self.shutdown()

    def shutdown(self):
        """Clean shutdown: release inputs, stop threads, log summary."""
        logger.info("Releasing controller inputs...")
        self.controller.release()
        self.controller.disconnect()

        logger.info("Stopping telemetry reader...")
        self.telemetry.stop()

        # Log safety summary
        self.safety.log_summary()

        elapsed = time.perf_counter() - self._session_start
        logger.info(
            f"Session summary: {self._lap_count} laps, "
            f"{self._frame_count:,} frames in {elapsed:.1f}s "
            f"(avg {self._frame_count/max(elapsed,0.001):.0f}Hz)"
        )


def main():
    parser = argparse.ArgumentParser(description="iRacing AI Bot")
    parser.add_argument("--track", default=None,
                        help="Track ID (e.g. 'sebring_2023')")
    parser.add_argument("--car", default=None,
                        help="Car ID (e.g. 'mx5_cup_2022')")
    parser.add_argument("--combo", default=None,
                        help="Combined track_car name (overrides --track/--car)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect track/car from running iRacing session")
    parser.add_argument("--mock", action="store_true",
                        help="Dry run: no iRacing or vJoy needed (for testing)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Determine combo name
    if args.combo:
        combo_name = args.combo
    elif args.track and args.car:
        combo_name = f"{args.track}_{args.car}"
    elif args.auto or args.mock:
        combo_name = None
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python orchestrator.py --track sebring_2023 --car mx5_cup")
        print("  python orchestrator.py --auto")
        print("  python orchestrator.py --mock --combo test_track_test_car")
        sys.exit(1)

    bot = BotOrchestrator(cfg, mock=args.mock)
    try:
        bot.start(combo_name)
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        bot.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
