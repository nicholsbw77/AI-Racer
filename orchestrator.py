"""
orchestrator.py

Main entry point for the live iRacing AI bot.

Ties together:
  - TelemetryReader (pyirsdk @ 360Hz)
  - DrivingAgent (model inference)
  - VJoyController (virtual controller output)

Usage:
  python orchestrator.py --track sebring_2023 --car mx5_cup
  python orchestrator.py --auto           # auto-detect track/car from iRacing session
  python orchestrator.py --mock           # dry run without iRacing/vJoy (for testing)

Safety features:
  - Ctrl+C to stop at any time → releases all inputs instantly
  - Automatic input release if iRacing disconnects
  - Speed cutoff: no inputs sent below min_speed_cutoff
  - Watchdog: kills bot if lap time exceeds 2× personal best (car stuck/crashed)
"""

import sys
import os
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import json
import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from telemetry import TelemetryReader, CarState
from inference import DrivingAgent
from controller import VJoyController, MockController
from safety_controller import SafetyController
from track_map import TrackMap
from manual_override import ManualOverride


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


class BotOrchestrator:
    """
    Main control loop: read telemetry → predict → send inputs.
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
        self.safety = SafetyController(cfg.get("safety", {}))

        if mock:
            self.controller = MockController()
        else:
            self.controller = VJoyController(device_id=1)

        # Manual keyboard override (F1-F5)
        self.manual = ManualOverride()

        # Metrics
        self._lap_count = 0
        self._frame_count = 0
        self._last_lap_dist = 0.0
        self._session_start = 0.0
        self._loop_times = []  # for Hz monitoring

        # Pit exit autopilot state
        self._pit_exit_active = False
        self._pit_exit_logged = False
        self._pit_exit_turn_start = None  # timestamp for post-pit-exit turn
        self._pit_stall_pullout_start = None  # timestamp for stall pullout maneuver
        self._pit_stall_pullout_done = False   # True once pullout swerve is finished
        self._pit_exit_cfg = self._load_pit_exit_config()

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    @staticmethod
    def _load_pit_exit_config() -> dict:
        """Load pit exit config from GUI-generated JSON, with defaults."""
        defaults = {
            "straight_duration": 8.0,
            "turn_angle": -60.0,
            "turn_duration": 1.5,
            "turn_throttle": 0.35,
            "straight_throttle": 0.40,
            "ramp_duration": 3.0,
            "cruise_until_lap_pct": 0.20,
            "cruise_throttle": 0.5,
            "pit_exit_track_pos": 0.6,  # known offset from racing line at pit exit
            "stall_pullout_left_dur": 1.2,    # seconds turning left out of stall
            "stall_pullout_right_dur": 0.8,   # seconds turning right to straighten
            "stall_pullout_steer": 0.35,      # steering magnitude for pullout
            "stall_pullout_throttle": 0.35,   # gentle throttle during pullout
        }
        cfg_path = Path(__file__).parent / "pit_exit_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    saved = json.load(f)
                merged = {**defaults, **saved}
                logger.info(f"Loaded pit exit config: {merged}")
                return merged
            except (json.JSONDecodeError, IOError):
                pass
        logger.info(f"Using default pit exit config: {defaults}")
        return defaults

    def _pit_exit_autopilot(self, state) -> tuple:
        """
        Simple autopilot to drive out of pit lane onto the track.

        Strategy:
          - 1st gear, gentle throttle (respect pit speed limit ~60 km/h)
          - Straight steering (pit lanes are mostly straight)
          - Slight steering correction based on lateral G to stay centered
          - Disengage once OnPitRoad goes False (car has exited pit lane)

        Returns:
            (throttle, brake, steering) or None if pit exit is complete
        """
        if not state.on_pit_road:
            # We've exited the pits — start the post-pit left turn at Sebring
            if self._pit_exit_active:
                self._pit_exit_active = False
                self._pit_exit_cfg = self._load_pit_exit_config()
                self._pit_exit_turn_start = time.perf_counter()
                # Seed track position with known offset from racing line
                pit_track_pos = self._pit_exit_cfg.get("pit_exit_track_pos", 0.6)
                self.telemetry.set_forced_track_pos(pit_track_pos)
                logger.info("Pit exit: off pit road, starting merge sequence "
                            "(seeded track_pos=%.2f)", pit_track_pos)

            # Post-pit-exit: drive straight, then turn (config from pit_exit_config.json)
            if self._pit_exit_turn_start is not None:
                pcfg = self._pit_exit_cfg
                straight_dur = pcfg["straight_duration"]
                turn_dur = pcfg["turn_duration"]
                # Convert angle in degrees to steering ratio (-1 to 1)
                # Assume ~180 deg max lock, so divide by 180
                turn_steering = max(-1.0, min(1.0, pcfg["turn_angle"] / 180.0))

                elapsed = time.perf_counter() - self._pit_exit_turn_start
                if elapsed < straight_dur:
                    steer_correction = -state.lat_g * 0.005
                    steering = max(-0.15, min(0.15, steer_correction))
                    throttle = pcfg["straight_throttle"]
                    brake = 0.0
                    if int(elapsed * 10) % 50 == 0:
                        logger.info(
                            f"PIT EXIT STRAIGHT: steer={steering:.3f} thr={throttle:.2f} "
                            f"elapsed={elapsed:.1f}/{straight_dur:.1f}s speed={state.speed:.1f}m/s"
                        )
                    return throttle, brake, steering
                elif elapsed < straight_dur + turn_dur:
                    steering = turn_steering
                    throttle = pcfg["turn_throttle"]
                    brake = 0.0
                    if int(elapsed * 10) % 5 == 0:
                        logger.info(
                            f"PIT EXIT TURN: steer={steering:.2f} thr={throttle:.2f} "
                            f"elapsed={elapsed - straight_dur:.1f}/{turn_dur:.1f}s speed={state.speed:.1f}m/s"
                        )
                    return throttle, brake, steering
                elif elapsed < straight_dur + turn_dur + pcfg.get("ramp_duration", 3.0):
                    # Ramp-up phase: moderate straight driving to build speed
                    # and let telemetry history stabilize before model handoff
                    ramp_dur = pcfg.get("ramp_duration", 3.0)
                    ramp_elapsed = elapsed - straight_dur - turn_dur
                    t = ramp_elapsed / ramp_dur
                    throttle = pcfg["turn_throttle"] + t * (0.5 - pcfg["turn_throttle"])
                    # Ease steering back to center
                    steering_correction = -state.lat_g * 0.005
                    steering = turn_steering * (1.0 - t) + steering_correction * t
                    steering = max(-0.5, min(0.5, steering))
                    brake = 0.0
                    if int(elapsed * 10) % 10 == 0:
                        logger.info(
                            f"PIT EXIT RAMP: steer={steering:.3f} thr={throttle:.2f} "
                            f"blend={t:.1%} elapsed={ramp_elapsed:.1f}/{ramp_dur:.1f}s speed={state.speed:.1f}m/s"
                        )
                    return throttle, brake, steering
                else:
                    # Cruise phase: follow the racing line using track map data
                    # until we reach a part of the track where the model can take over
                    cruise_target = pcfg.get("cruise_until_lap_pct", 0.15)
                    cruise_thr = pcfg.get("cruise_throttle", 0.5)
                    if state.lap_dist_pct < cruise_target:
                        # Use track map typical steering as base if available
                        track_map = getattr(self.telemetry, '_track_map', None)
                        if track_map is not None:
                            bin_data = track_map._get_bin(state.lap_dist_pct)
                            base_steer = bin_data.typical_steering
                            lat_g_correction = -state.lat_g * 0.003

                            # Correct for lateral offset from racing line.
                            # track_pos > 0 means right of line → steer left
                            # (subtract proportional correction).
                            # Gain of 0.4 means track_pos=0.6 → -0.24 correction
                            track_pos_correction = -state.track_pos * 0.4
                            steering = base_steer + lat_g_correction + track_pos_correction
                            steering = max(-0.5, min(0.5, steering))

                            # Also match typical speed — use track map speed
                            typical_speed = bin_data.typical_speed
                            if state.speed < typical_speed * 0.8:
                                throttle = min(cruise_thr + 0.1, 0.7)
                            elif state.speed > typical_speed * 1.1:
                                throttle = 0.1
                            else:
                                throttle = cruise_thr
                        else:
                            # Fallback: lat_g + track_pos correction
                            track_pos_correction = -state.track_pos * 0.4
                            steer_correction = -state.lat_g * 0.005 + track_pos_correction
                            steering = max(-0.3, min(0.3, steer_correction))
                            throttle = cruise_thr
                        brake = 0.0
                        if self._frame_count % 60 == 0:
                            logger.info(
                                f"PIT EXIT CRUISE: steer={steering:.3f} thr={throttle:.2f} "
                                f"speed={state.speed:.1f}m/s lap_pct={state.lap_dist_pct:.3f} "
                                f"track_pos={state.track_pos:.2f} target={cruise_target:.3f}"
                            )
                        return throttle, brake, steering

                    logger.info(
                        f"Pit exit complete — handing off to model at "
                        f"speed={state.speed:.1f}m/s lap_pct={state.lap_dist_pct:.3f} "
                        f"track_pos={state.track_pos:.2f}"
                    )
                    self._pit_exit_turn_start = None
                    # Reset safety controller so it doesn't carry edge/recovery
                    # state from the pit exit phase
                    self.safety.reset()
                    # Note: track_pos is handled by the forced blend that was
                    # seeded at pit exit — it will naturally converge to the
                    # estimator's value over the next few seconds
            return None

        if not self._pit_exit_active:
            self._pit_exit_active = True
            self._pit_stall_pullout_start = None
            self._pit_stall_pullout_done = False
            logger.info("Pit exit autopilot engaged")

        # Shift to 1st if needed, then 2nd once rolling
        if state.speed < 5.0:
            target_gear = 1
        else:
            target_gear = 2
        if target_gear != self.controller._current_gear:
            self.controller.shift_to(target_gear)

        # Pit speed limit is typically 60-80 km/h (16-22 m/s)
        PIT_SPEED_LIMIT = 18.0  # m/s (~65 km/h), safe for most tracks

        # --- Phase 0: Stall pullout (left swerve then right to straighten) ---
        if not self._pit_stall_pullout_done:
            pcfg = self._pit_exit_cfg
            left_dur = pcfg.get("stall_pullout_left_dur", 1.2)
            right_dur = pcfg.get("stall_pullout_right_dur", 0.8)
            pullout_steer = pcfg.get("stall_pullout_steer", 0.35)
            pullout_thr = pcfg.get("stall_pullout_throttle", 0.35)

            # Start the pullout timer once the car begins rolling
            if state.speed > 0.5 and self._pit_stall_pullout_start is None:
                self._pit_stall_pullout_start = time.perf_counter()
                logger.info("Pit stall pullout: car rolling, starting left swerve")

            if self._pit_stall_pullout_start is not None:
                elapsed = time.perf_counter() - self._pit_stall_pullout_start

                if elapsed < left_dur:
                    # Turn left to pull out of stall (one car width)
                    steering = -pullout_steer
                    throttle = pullout_thr
                    brake = 0.0
                    if self._frame_count % 30 == 0:
                        logger.info(
                            f"PIT STALL LEFT: steer={steering:.3f} thr={throttle:.2f} "
                            f"elapsed={elapsed:.1f}/{left_dur:.1f}s speed={state.speed:.1f}m/s"
                        )
                    return throttle, brake, steering

                elif elapsed < left_dur + right_dur:
                    # Turn right to straighten on pit road
                    steering = pullout_steer
                    throttle = pullout_thr
                    brake = 0.0
                    if self._frame_count % 30 == 0:
                        logger.info(
                            f"PIT STALL RIGHT: steer={steering:.3f} thr={throttle:.2f} "
                            f"elapsed={elapsed - left_dur:.1f}/{right_dur:.1f}s speed={state.speed:.1f}m/s"
                        )
                    return throttle, brake, steering

                else:
                    # Pullout done — continue with normal pit road driving
                    self._pit_stall_pullout_done = True
                    logger.info("Pit stall pullout complete, driving down pit road")
            else:
                # Not rolling yet — apply throttle to get moving
                return 0.5, 0.0, 0.0

        # --- Phase 1: Normal pit road driving (straight with speed limit) ---

        # Gentle throttle to get moving, back off near speed limit
        if state.speed < 2.0:
            throttle = 0.5
        elif state.speed < PIT_SPEED_LIMIT * 0.8:
            throttle = 0.4
        elif state.speed < PIT_SPEED_LIMIT:
            throttle = 0.15  # coast near limit
        else:
            throttle = 0.0   # over limit, lift

        brake = 0.0

        # Minimal steering correction using lateral G
        steer_correction = -state.lat_g * 0.005
        steering = max(-0.15, min(0.15, steer_correction))

        if self._frame_count % 60 == 0:
            logger.info(
                f"PIT EXIT: thr={throttle:.2f} steer={steering:.3f} "
                f"speed={state.speed:.1f}m/s gear={target_gear}"
            )

        return throttle, brake, steering

    def _match_checkpoint(self, raw_combo: str) -> str:
        """Match auto-detected combo to an existing checkpoint folder.

        iRacing returns full display names like
        'sebring_international_raceway_cadillac_cts_v_racecar' but checkpoints
        use abbreviated .ibt-derived names like 'cadillacctsvr_sebring_international'.
        We score each checkpoint folder by how many of the detected keywords it contains.
        """
        ckpt_dir = Path("checkpoints")
        if not ckpt_dir.exists():
            return raw_combo

        candidates = [d.name for d in ckpt_dir.iterdir()
                       if d.is_dir() and (d / "best.pt").exists()]
        if not candidates:
            return raw_combo

        # Tokenize the detected combo into keywords
        keywords = set(raw_combo.split("_"))
        # Remove very short / common noise words
        keywords = {k for k in keywords if len(k) > 2}

        best_match = raw_combo
        best_score = 0

        for cand in candidates:
            # Score: how many keywords appear as substrings in the candidate
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
                # Retry loop — wait for iRacing to start
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

        # Propagate normalization constants from checkpoint to telemetry
        norm = self.agent.norm
        if norm.get("speed_max_ms"):
            self.telemetry._speed_max = norm["speed_max_ms"]
            logger.info(f"Set telemetry speed_max={norm['speed_max_ms']:.1f} m/s")
        if norm.get("steering_lock_radians"):
            self.telemetry._steering_lock = norm["steering_lock_radians"]
            logger.info(f"Set telemetry steering_lock={norm['steering_lock_radians']:.3f} rad")
        if norm.get("rpm_max"):
            self.telemetry._rpm_max = norm["rpm_max"]

        # Load track map if available
        checkpoint_dir = Path(self.cfg["paths"]["checkpoints"]) / combo_name
        track_map_path = checkpoint_dir / "track_map.json"
        if track_map_path.exists():
            try:
                track_map = TrackMap.load(str(track_map_path))
                self.telemetry.set_track_map(track_map)
                logger.info(f"Track map loaded: {track_map.summary().splitlines()[0]}")
            except Exception as e:
                logger.warning(f"Could not load track map: {e}")

        # Start telemetry background thread
        self.telemetry.start()

        # Start manual override key listener
        self.manual.start()

        logger.info("Bot active. Press Ctrl+C to stop.")
        logger.info(f"Target loop rate: {self.cfg['inference']['loop_hz']}Hz")
        logger.info("Manual override: F1=stop F2=left F3=right F4=gas F5=hand-back")

        self._running = True
        self._session_start = time.perf_counter()
        self._run_loop(combo_name)

    def _run_loop(self, combo_name: str):
        """Main control loop."""
        target_period = 1.0 / self.cfg["inference"]["loop_hz"]
        # Use sequence_history from the loaded checkpoint (matches training)
        if self.agent.is_ready and hasattr(self.agent, 'sequence_history'):
            sequence_history = self.agent.sequence_history
            n_state_features = self.agent.n_state_features
        else:
            sequence_history = self.cfg["training"]["sequence_history"]
            n_state_features = None  # use default (all 14)

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
                    combo_name = new_combo

            # Debug: log state flags periodically
            if self._frame_count % 300 == 0:
                logger.info(
                    f"State: on_track={state.is_on_track} "
                    f"pit_road={state.on_pit_road} "
                    f"session_active={state.session_active} "
                    f"speed={state.speed:.1f}m/s "
                    f"gear={state.gear:.0f} "
                    f"lap_pct={state.lap_dist_pct:.3f}"
                )

            # Skip if session not active
            if not state.session_active:
                self.controller.release()
                time.sleep(0.01)
                self._frame_count += 1
                continue

            # --- Pit exit autopilot ---
            # If on pit road, use simple autopilot to drive out
            pit_result = self._pit_exit_autopilot(state)
            if pit_result is not None:
                throttle, brake, steering = pit_result
                self.controller.set_inputs(throttle, brake, steering)
                self.telemetry.inject_bot_actions(throttle, brake, steering)
                self._frame_count += 1

                loop_elapsed = time.perf_counter() - loop_start
                remaining = target_period - loop_elapsed
                if remaining > 0.0001:
                    time.sleep(remaining)
                continue

            # Skip if not on track (off-track excursion, not pits)
            if not state.is_on_track:
                self.controller.release()
                time.sleep(0.01)
                self._frame_count += 1
                continue

            # --- Model-driven control ---
            # Build state vector for inference
            state_vec = self.telemetry.build_state_vector(
                state,
                sequence_history=sequence_history,
                n_state_features=n_state_features,
            )

            # Run inference
            throttle, brake, steering = self.agent.predict(
                state_vec,
                car_speed_ms=state.speed,
            )

            # Apply safety controller — corrects outputs near track edges
            throttle, brake, steering = self.safety.apply(
                throttle, brake, steering, state
            )

            # Debug: log model outputs periodically
            if self._frame_count % 60 == 0:
                logger.info(
                    f"Output: thr={throttle:.3f} brk={brake:.3f} "
                    f"steer={steering:.3f} speed={state.speed:.1f} "
                    f"track_pos={state.track_pos:.2f}"
                )

            # Manual override — bypass model if F-keys active
            override = self.manual.get_controls()
            if override is not None:
                throttle, brake, steering = override
                if self._frame_count % 60 == 0:
                    logger.info(
                        f"MANUAL: thr={throttle:.2f} brk={brake:.2f} "
                        f"steer={steering:.3f}"
                    )

            # Send to controller
            self.controller.set_inputs(throttle, brake, steering)

            # Inject bot outputs into history buffer for next prediction
            self.telemetry.inject_bot_actions(throttle, brake, steering)

            # Handle gear shifts — sync controller to iRacing's reported gear.
            # iRacing manages the auto-clutch; we just need to send shift
            # commands when the reported gear differs from what we last sent.
            reported_gear = int(round(state.gear))
            if reported_gear > 0:
                # Sync controller's internal tracking to iRacing ground truth
                # to prevent drift from missed shifts
                self.controller._current_gear = reported_gear

            # RPM-based shift logic: upshift near redline, downshift when lugging
            if state.rpm > 0 and reported_gear > 0:
                if state.rpm > 6800 and reported_gear < 6:
                    self.controller.shift_up()
                elif state.rpm < 2500 and reported_gear > 1 and state.speed > 5.0:
                    self.controller.shift_down()

            # Metrics
            self._frame_count += 1
            self._detect_lap_crossing(state)

            loop_elapsed = time.perf_counter() - loop_start
            self._loop_times.append(loop_elapsed)

            # Log performance every 360 frames (approx ~6 seconds at 60Hz)
            if self._frame_count % 360 == 0:
                self._log_stats()

            # Yield remaining time in period (prevents CPU hammering)
            remaining = target_period - loop_elapsed
            if remaining > 0.0001:
                time.sleep(remaining)

    def _detect_lap_crossing(self, state: CarState):
        """Detect lap completion by watching lap_dist_pct wrap around."""
        if self._last_lap_dist > 0.95 and state.lap_dist_pct < 0.05:
            self._lap_count += 1
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
        self._loop_times = self._loop_times[-360:]  # keep last 1s

        logger.info(
            f"Loop: {actual_hz:.0f}Hz  avg={avg_ms:.2f}ms  "
            f"max={max_ms:.2f}ms  laps={self._lap_count}"
        )

    def _shutdown_handler(self, signum, frame):
        """Handle Ctrl+C / SIGTERM gracefully."""
        logger.info("\nShutdown signal received...")
        self._running = False
        self.shutdown()

    def shutdown(self):
        """Clean shutdown: release inputs, stop threads."""
        logger.info("Releasing controller inputs...")
        self.controller.release()
        self.controller.disconnect()

        logger.info("Stopping manual override listener...")
        self.manual.stop()

        logger.info("Stopping telemetry reader...")
        self.telemetry.stop()

        elapsed = time.perf_counter() - self._session_start
        logger.info(
            f"Session summary: {self._lap_count} laps, "
            f"{self._frame_count:,} frames in {elapsed:.1f}s "
            f"(avg {self._frame_count/elapsed:.0f}Hz)"
        )

        # Log safety controller statistics
        s = self.safety.stats
        if s.total_frames > 0:
            logger.info(
                f"Safety stats: {s.off_track_frames} off-track frames, "
                f"{s.edge_warning_frames} edge warnings, "
                f"{s.edge_danger_frames} edge danger frames, "
                f"{s.recovery_mode_activations} recovery activations "
                f"(over {s.total_frames} total frames)"
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
        # Will auto-detect from session info after connecting
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
