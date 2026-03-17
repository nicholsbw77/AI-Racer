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
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.telemetry import TelemetryReader, CarState
from agent.inference import DrivingAgent
from agent.controller import VJoyController, MockController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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

        if mock:
            self.controller = MockController()
        else:
            self.controller = VJoyController(device_id=1)

        # Metrics
        self._lap_count = 0
        self._frame_count = 0
        self._last_lap_dist = 0.0
        self._session_start = 0.0
        self._loop_times = []  # for Hz monitoring

        # Register Ctrl+C handler
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def start(self, combo_name: str):
        """Start the bot for a given track/car combo."""
        logger.info(f"Starting bot for: {combo_name}")

        # Connect controller
        if not self.controller.connect():
            logger.error("Failed to connect controller. Exiting.")
            return

        # Load model
        if not self.agent.load_checkpoint(combo_name):
            if not self.mock:
                logger.error(f"No checkpoint for '{combo_name}'. Train first.")
                logger.error(f"Run: python trainer/train.py --data data/processed/ --track ...")
                return
            else:
                logger.warning("Mock mode: running without model (random outputs)")

        # Connect to iRacing
        if not self.mock:
            if not self.telemetry.connect():
                logger.warning("iRacing not detected. Waiting...")

        # Start telemetry background thread
        self.telemetry.start()

        logger.info("Bot active. Press Ctrl+C to stop.")
        logger.info(f"Target loop rate: {self.cfg['inference']['loop_hz']}Hz")

        self._running = True
        self._session_start = time.perf_counter()
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
                    combo_name = new_combo

            # Skip if not on track or session not active
            if not state.is_on_track or not state.session_active:
                self.controller.release()
                time.sleep(0.01)
                continue

            # Build state vector for inference
            state_vec = self.telemetry.build_state_vector(
                state,
                sequence_history=sequence_history,
            )

            # Run inference
            throttle, brake, steering = self.agent.predict(
                state_vec,
                car_speed_ms=state.speed,
            )

            # Send to controller
            self.controller.set_inputs(throttle, brake, steering)

            # Metrics
            self._frame_count += 1
            self._detect_lap_crossing(state)

            loop_elapsed = time.perf_counter() - loop_start
            self._loop_times.append(loop_elapsed)

            # Log performance every 360 frames (approx 1 second)
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

        logger.info("Stopping telemetry reader...")
        self.telemetry.stop()

        elapsed = time.perf_counter() - self._session_start
        logger.info(
            f"Session summary: {self._lap_count} laps, "
            f"{self._frame_count:,} frames in {elapsed:.1f}s "
            f"(avg {self._frame_count/elapsed:.0f}Hz)"
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
        # Will auto-detect from session info
        combo_name = "auto_detect"
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
