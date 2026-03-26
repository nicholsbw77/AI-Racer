"""
Manual keyboard override for emergency car recovery.

F-key bindings (while iRacing window or terminal is active):
    F1  — Emergency stop (full brake, hold)
    F2  — Steer left (hold)
    F3  — Steer right (hold)
    F4  — Gas / throttle (hold)
    F5  — Hand back to model (press once)

Any of F1–F4 activates manual override mode (model output is ignored).
F5 deactivates it and returns control to the model.

Uses pynput for global key listening (works even when iRacing has focus).
"""

import logging
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed — manual override keys disabled. "
                   "Install with: pip install pynput")


# Default override control values
OVERRIDE_THROTTLE = 0.5   # moderate gas when F4 held
OVERRIDE_STEER = 0.3      # moderate steer angle when F2/F3 held
OVERRIDE_BRAKE = 1.0      # full brake on F1


class ManualOverride:
    """Keyboard-driven manual override for the driving model."""

    def __init__(self, throttle_val: float = OVERRIDE_THROTTLE,
                 steer_val: float = OVERRIDE_STEER,
                 brake_val: float = OVERRIDE_BRAKE):
        self._active = False          # True = manual mode, model bypassed
        self._lock = threading.Lock()

        # Currently held keys
        self._stop = False    # F1
        self._left = False    # F2
        self._right = False   # F3
        self._gas = False     # F4

        # Tunable override magnitudes
        self._throttle_val = throttle_val
        self._steer_val = steer_val
        self._brake_val = brake_val

        self._listener: Optional[keyboard.Listener] = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self):
        """Start the global key listener in a background thread."""
        if not PYNPUT_AVAILABLE:
            logger.warning("Manual override: pynput not available, skipping")
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        logger.info(
            "Manual override keys active: "
            "F1=stop  F2=left  F3=right  F4=gas  F5=hand-back"
        )

    def stop(self):
        """Stop the key listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_controls(self) -> Optional[Tuple[float, float, float]]:
        """Return (throttle, brake, steering) if manual override is active.

        Returns None if model should drive (override not active).
        """
        if not self._active:
            return None

        with self._lock:
            # Build controls from held keys
            throttle = self._throttle_val if self._gas else 0.0
            brake = self._brake_val if self._stop else 0.0
            steering = 0.0
            if self._left:
                steering = -self._steer_val
            if self._right:
                steering = self._steer_val
            # If both left and right held, cancel out (steering stays 0)
            if self._left and self._right:
                steering = 0.0

            # If braking, cut throttle
            if self._stop:
                throttle = 0.0

        return (throttle, brake, steering)

    def _on_press(self, key):
        """Handle key press events."""
        if not hasattr(key, 'name'):
            return

        with self._lock:
            if key == keyboard.Key.f1:
                if not self._active:
                    logger.info("MANUAL OVERRIDE ACTIVATED (F1 - stop)")
                self._active = True
                self._stop = True
            elif key == keyboard.Key.f2:
                if not self._active:
                    logger.info("MANUAL OVERRIDE ACTIVATED (F2 - left)")
                self._active = True
                self._left = True
            elif key == keyboard.Key.f3:
                if not self._active:
                    logger.info("MANUAL OVERRIDE ACTIVATED (F3 - right)")
                self._active = True
                self._right = True
            elif key == keyboard.Key.f4:
                if not self._active:
                    logger.info("MANUAL OVERRIDE ACTIVATED (F4 - gas)")
                self._active = True
                self._gas = True
            elif key == keyboard.Key.f5:
                if self._active:
                    logger.info("MANUAL OVERRIDE RELEASED — handing back to model")
                    self._active = False
                    self._stop = False
                    self._left = False
                    self._right = False
                    self._gas = False

    def _on_release(self, key):
        """Handle key release events."""
        if not hasattr(key, 'name'):
            return

        with self._lock:
            if key == keyboard.Key.f1:
                self._stop = False
            elif key == keyboard.Key.f2:
                self._left = False
            elif key == keyboard.Key.f3:
                self._right = False
            elif key == keyboard.Key.f4:
                self._gas = False
