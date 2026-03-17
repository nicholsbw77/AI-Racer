"""
agent/controller.py

Sends driving inputs to iRacing via vJoy virtual controller.

vJoy axis range: 0x1 to 0x8000 (1 to 32768)
  Center = 16384
  Min    = 1
  Max    = 32768

iRacing must be configured to use the vJoy device as steering wheel + pedals.
In iRacing options → Controls → select vJoy Device 1.

Install vJoy driver: https://sourceforge.net/projects/vjoystick/
Install pyvjoy:      pip install pyvjoy
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# vJoy axis constants
VJOY_MIN = 0x1
VJOY_MAX = 0x8000
VJOY_CENTER = (VJOY_MAX + VJOY_MIN) // 2  # 16384

# vJoy axis IDs (standard mapping)
AXIS_STEERING = 0x30  # HID_USAGE_X  (axis 1)
AXIS_THROTTLE = 0x31  # HID_USAGE_Y  (axis 2)
AXIS_BRAKE    = 0x32  # HID_USAGE_Z  (axis 3)

try:
    import pyvjoy
    VJOY_AVAILABLE = True
except ImportError:
    VJOY_AVAILABLE = False
    logger.warning("pyvjoy not available - controller will run in mock/log mode")


def _to_vjoy_axis(value: float, centered: bool = False) -> int:
    """
    Convert a normalized float to vJoy axis integer.

    Args:
        value: For pedals: 0.0 (released) to 1.0 (fully pressed)
               For steering: -1.0 (full left) to +1.0 (full right)
        centered: True for steering (maps -1..+1 to MIN..MAX centered at 16384)
                  False for pedals (maps 0..1 to MIN..MAX)
    """
    if centered:
        # Steering: -1 → MIN, 0 → CENTER, +1 → MAX
        scaled = (value + 1.0) / 2.0  # normalize to 0-1
    else:
        scaled = value

    raw = int(VJOY_MIN + scaled * (VJOY_MAX - VJOY_MIN))
    return max(VJOY_MIN, min(VJOY_MAX, raw))


class VJoyController:
    """
    Sends throttle, brake, and steering to iRacing via vJoy virtual device.

    Usage:
        ctrl = VJoyController(device_id=1)
        ctrl.connect()
        ctrl.set_inputs(throttle=0.8, brake=0.0, steering=0.1)
        ctrl.release()
    """

    def __init__(self, device_id: int = 1):
        self.device_id = device_id
        self._device = None
        self._connected = False
        self._mock_mode = not VJOY_AVAILABLE

        # Track last sent values to detect changes
        self._last_throttle = -1.0
        self._last_brake = -1.0
        self._last_steering = -99.0

    def connect(self) -> bool:
        """Acquire vJoy device. Returns True if successful."""
        if self._mock_mode:
            logger.info("vJoy mock mode active (pyvjoy not installed)")
            self._connected = True
            return True

        try:
            self._device = pyvjoy.VJoyDevice(self.device_id)
            self._connected = True
            logger.info(f"vJoy Device {self.device_id} acquired")

            # Center all axes on connect
            self.set_inputs(0.0, 0.0, 0.0)
            return True

        except Exception as e:
            logger.error(f"Failed to acquire vJoy device {self.device_id}: {e}")
            logger.error("Make sure vJoy driver is installed and device is enabled")
            return False

    def set_inputs(
        self,
        throttle: float,
        brake: float,
        steering: float,
    ) -> bool:
        """
        Send control inputs to vJoy.

        Args:
            throttle: 0.0 (off) to 1.0 (full)
            brake:    0.0 (off) to 1.0 (full)
            steering: -1.0 (full left) to +1.0 (full right)

        Returns True if sent successfully.
        """
        if not self._connected:
            return False

        # Clamp inputs to valid range
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steering = max(-1.0, min(1.0, steering))

        if self._mock_mode:
            # Log at DEBUG level to avoid flooding console
            logger.debug(
                f"[MOCK] thr={throttle:.3f} brk={brake:.3f} str={steering:.3f}"
            )
            return True

        try:
            self._device.set_axis(AXIS_THROTTLE, _to_vjoy_axis(throttle, centered=False))
            self._device.set_axis(AXIS_BRAKE,    _to_vjoy_axis(brake, centered=False))
            self._device.set_axis(AXIS_STEERING, _to_vjoy_axis(steering, centered=True))

            self._last_throttle = throttle
            self._last_brake = brake
            self._last_steering = steering
            return True

        except Exception as e:
            logger.error(f"vJoy write error: {e}")
            return False

    def release(self):
        """Release all inputs to safe state (no throttle, no brake, centered steering)."""
        if self._connected:
            self.set_inputs(0.0, 0.0, 0.0)
            logger.info("vJoy inputs released to safe state")

    def disconnect(self):
        """Release inputs and disconnect."""
        self.release()
        if self._device and not self._mock_mode:
            try:
                self._device = None
            except Exception:
                pass
        self._connected = False
        logger.info("vJoy device disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected


class MockController:
    """
    Drop-in replacement for VJoyController that just logs outputs.
    Used for testing the agent pipeline without vJoy installed.
    """

    def __init__(self):
        self._connected = False

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockController connected (no hardware output)")
        return True

    def set_inputs(self, throttle: float, brake: float, steering: float) -> bool:
        logger.info(f"[OUTPUT] thr={throttle:.4f}  brk={brake:.4f}  str={steering:.4f}")
        return True

    def release(self):
        logger.info("[OUTPUT] Released")

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
