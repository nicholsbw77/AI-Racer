"""
Manual override for emergency car recovery via keyboard or gamepad.

Keyboard (F-keys, via pynput — works when iRacing has focus):
    F1  — Emergency stop (full brake, hold)
    F2  — Steer left (hold)
    F3  — Steer right (hold)
    F4  — Gas / throttle (hold)
    F5  — Hand back to model (press once)

Gamepad (via pygame — not captured by iRacing):
    A / Cross      — Emergency stop (full brake)
    X / Square     — Gas / throttle
    D-pad left     — Steer left
    D-pad right    — Steer right
    B / Circle     — Hand back to model
    Left stick X   — Analog steering override (if held past deadzone)

Any override input activates manual mode (model output is ignored).
B / F5 deactivates it and returns control to the model.
"""

import logging
import threading
import time
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Keyboard support
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# Gamepad support
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# Default override control values
OVERRIDE_THROTTLE = 0.5   # moderate gas when F4/X held
OVERRIDE_STEER = 0.3      # moderate steer angle when F2/F3/D-pad held
OVERRIDE_BRAKE = 1.0      # full brake on F1/A

# Gamepad button indices (Xbox layout — most common)
# These can vary by controller; Xbox is the default.
GP_BTN_A = 0          # Emergency stop (brake)
GP_BTN_B = 1          # Hand back to model
GP_BTN_X = 2          # Gas / throttle
GP_STICK_DEADZONE = 0.25  # Left stick deadzone for analog steering


class ManualOverride:
    """Keyboard + gamepad manual override for the driving model."""

    def __init__(self, throttle_val: float = OVERRIDE_THROTTLE,
                 steer_val: float = OVERRIDE_STEER,
                 brake_val: float = OVERRIDE_BRAKE):
        self._active = False          # True = manual mode, model bypassed
        self._lock = threading.Lock()

        # Currently held inputs (from any source)
        self._stop = False    # F1 / A button
        self._left = False    # F2 / D-pad left
        self._right = False   # F3 / D-pad right
        self._gas = False     # F4 / X button
        self._analog_steer = 0.0  # Left stick X axis (-1 to +1)

        # Tunable override magnitudes
        self._throttle_val = throttle_val
        self._steer_val = steer_val
        self._brake_val = brake_val

        # Input backends
        self._keyboard_listener = None
        self._gamepad_thread = None
        self._gamepad_running = False
        self._joystick = None

    @property
    def active(self) -> bool:
        return self._active

    def start(self):
        """Start all available input listeners."""
        self._start_keyboard()
        self._start_gamepad()

    def stop(self):
        """Stop all input listeners."""
        self._stop_keyboard()
        self._stop_gamepad()

    # ------------------------------------------------------------------
    # Unified control output
    # ------------------------------------------------------------------

    def get_controls(self) -> Optional[Tuple[float, float, float]]:
        """Return (throttle, brake, steering) if manual override is active.

        Returns None if model should drive (override not active).
        """
        if not self._active:
            return None

        with self._lock:
            throttle = self._throttle_val if self._gas else 0.0
            brake = self._brake_val if self._stop else 0.0

            # Analog stick takes priority over digital D-pad
            if abs(self._analog_steer) > GP_STICK_DEADZONE:
                steering = self._analog_steer  # full range -1 to +1
            else:
                steering = 0.0
                if self._left:
                    steering = -self._steer_val
                if self._right:
                    steering = self._steer_val
                if self._left and self._right:
                    steering = 0.0

            # If braking, cut throttle
            if self._stop:
                throttle = 0.0

        return (throttle, brake, steering)

    # ------------------------------------------------------------------
    # Keyboard backend (pynput)
    # ------------------------------------------------------------------

    def _start_keyboard(self):
        if not PYNPUT_AVAILABLE:
            logger.warning("Manual override: pynput not available, skipping keyboard")
            return

        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._keyboard_listener.daemon = True
        self._keyboard_listener.start()
        logger.info(
            "Manual override: F1=stop F2=left F3=right F4=gas F5=hand-back"
        )

    def _stop_keyboard(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _on_key_press(self, key):
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
                self._release_override("F5")

    def _on_key_release(self, key):
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

    # ------------------------------------------------------------------
    # Gamepad backend (pygame)
    # ------------------------------------------------------------------

    def _start_gamepad(self):
        if not PYGAME_AVAILABLE:
            logger.info("Manual override: pygame not available, skipping gamepad. "
                        "Install with: pip install pygame")
            return

        self._gamepad_running = True
        self._gamepad_thread = threading.Thread(
            target=self._gamepad_poll_loop, daemon=True
        )
        self._gamepad_thread.start()

    def _stop_gamepad(self):
        self._gamepad_running = False
        if self._gamepad_thread is not None:
            self._gamepad_thread.join(timeout=2.0)
            self._gamepad_thread = None
        if PYGAME_AVAILABLE and pygame.get_init():
            pygame.joystick.quit()

    def _gamepad_poll_loop(self):
        """Background thread: poll gamepad at ~30Hz."""
        try:
            pygame.joystick.init()
        except Exception as e:
            logger.warning(f"Failed to init pygame joystick: {e}")
            return

        # Wait for a joystick to appear
        joystick = None
        while self._gamepad_running and joystick is None:
            pygame.joystick.quit()
            pygame.joystick.init()
            count = pygame.joystick.get_count()
            if count > 0:
                # Use the last joystick (most likely a secondary controller,
                # not the racing wheel which is usually device 0)
                for i in range(count - 1, -1, -1):
                    js = pygame.joystick.Joystick(i)
                    js.init()
                    name = js.get_name().lower()
                    # Skip devices that look like racing wheels
                    if any(w in name for w in ["wheel", "fanatec", "logitech g",
                                                "thrustmaster", "simucube"]):
                        logger.debug(f"Skipping racing wheel: {js.get_name()}")
                        continue
                    joystick = js
                    break
                if joystick is None:
                    # No non-wheel controller found, use the last device anyway
                    joystick = pygame.joystick.Joystick(count - 1)
                    joystick.init()
            if joystick is None:
                time.sleep(2.0)  # retry every 2s

        if joystick is None:
            return

        self._joystick = joystick
        logger.info(
            f"Gamepad override connected: {joystick.get_name()} "
            f"(buttons={joystick.get_numbuttons()}, "
            f"axes={joystick.get_numaxes()}, "
            f"hats={joystick.get_numhats()})"
        )
        logger.info(
            "Gamepad override: A=stop  X=gas  D-pad=steer  "
            "B=hand-back  LStick=analog-steer"
        )

        poll_interval = 1.0 / 30.0  # 30Hz

        while self._gamepad_running:
            try:
                # Pump pygame events (required for joystick updates)
                pygame.event.pump()
                self._read_gamepad(joystick)
            except pygame.error:
                logger.warning("Gamepad disconnected")
                break
            except Exception as e:
                logger.debug(f"Gamepad poll error: {e}")
            time.sleep(poll_interval)

    def _read_gamepad(self, js):
        """Read current gamepad state and update override flags."""
        with self._lock:
            # Buttons
            a_pressed = js.get_button(GP_BTN_A) if js.get_numbuttons() > GP_BTN_A else False
            b_pressed = js.get_button(GP_BTN_B) if js.get_numbuttons() > GP_BTN_B else False
            x_pressed = js.get_button(GP_BTN_X) if js.get_numbuttons() > GP_BTN_X else False

            # D-pad (hat 0)
            hat_x = 0
            if js.get_numhats() > 0:
                hat_x, _ = js.get_hat(0)

            # Left stick X axis (axis 0 on most controllers)
            stick_x = 0.0
            if js.get_numaxes() > 0:
                stick_x = js.get_axis(0)

            # B button = release override
            if b_pressed:
                if self._active:
                    self._release_override("Gamepad B")
                return

            # Any action button activates override
            any_input = (a_pressed or x_pressed or hat_x != 0 or
                         abs(stick_x) > GP_STICK_DEADZONE)

            if any_input and not self._active:
                source = ("A-stop" if a_pressed else
                          "X-gas" if x_pressed else
                          "D-pad" if hat_x != 0 else "LStick")
                logger.info(f"MANUAL OVERRIDE ACTIVATED (Gamepad {source})")
                self._active = True

            if self._active:
                self._stop = a_pressed
                self._gas = x_pressed
                self._left = hat_x < 0
                self._right = hat_x > 0
                self._analog_steer = stick_x

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _release_override(self, source: str):
        """Deactivate manual override and clear all inputs."""
        if self._active:
            logger.info(f"MANUAL OVERRIDE RELEASED ({source}) — handing back to model")
        self._active = False
        self._stop = False
        self._left = False
        self._right = False
        self._gas = False
        self._analog_steer = 0.0
