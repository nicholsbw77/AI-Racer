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
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from telemetry import TelemetryReader, CarState
from inference import DrivingAgent
from controller import VJoyController, MockController
from safety import SafetyController, SafetyAction, SafetyVerdict
from track_map import TrackMap, LiveTracker


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

        # Pit exit autopilot state
        self._pit_exit_active = False
        self._pit_exit_logged = False

        # Safety stats
        self._safety_kills = 0
        self._safety_interventions = 0

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

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
            # We've exited the pits
            if self._pit_exit_active:
                logger.info("Pit exit complete -- handing off to model")
                self._pit_exit_active = False
            return None

        if not self._pit_exit_active:
            self._pit_exit_active = True
            logger.info("Pit exit autopilot engaged")

        # Use safety controller's pit speed limit
        pit_limit = self.safety.config.pit_speed_limit_ms

        # Gentle throttle to get moving, back off near speed limit
        if state.speed < 2.0:
            throttle = 0.5
        elif state.speed < pit_limit * 0.8:
            throttle = 0.4
        elif state.speed < pit_limit:
            throttle = 0.15  # coast near limit
        else:
            throttle = 0.0   # over limit, lift

        brake = 0.0

        # Minimal steering correction using lateral G
        steer_correction = -state.lat_g * 0.005
        steering = max(-0.15, min(0.15, steer_correction))

        # Shift to 1st if needed, then 2nd once rolling
        if state.speed < 5.0:
            target_gear = 1
        else:
            target_gear = 2

        if target_gear != self.controller._current_gear:
            self.controller.shift_to(target_gear)

        if self._frame_count % 60 == 0:
            logger.info(
                f"PIT EXIT: thr={throttle:.2f} steer={steering:.3f} "
                f"speed={state.speed:.1f}m/s gear={target_gear}"
            )

        return throttle, brake, steering

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

            # --- Pit exit autopilot ---
            if state.on_pit_road and state.session_active:
                pit_result = self._pit_exit_autopilot(state)
                if pit_result is not None:
                    throttle, brake, steering = pit_result

                    # Still run safety on pit autopilot outputs
                    verdict = self.safety.check(
                        state, throttle, brake, steering,
                        is_on_track=state.is_on_track,
                        on_pit_road=True,
                        session_active=state.session_active,
                    )
                    self._apply_verdict(verdict, state)
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

            # Handle gear shifts
            target_gear = int(round(state.gear))
            if target_gear != self.controller._current_gear and target_gear > 0:
                self.controller.shift_to(target_gear)

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
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering)
            self._safety_interventions += 1
        elif verdict.action == SafetyAction.ATTENUATE:
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering)
            self._safety_interventions += 1
        else:
            # PASS — send through
            self.controller.set_inputs(verdict.throttle, verdict.brake, verdict.steering)

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
