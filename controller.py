"""
agent/controller.py

Sends driving inputs to iRacing via vJoy virtual controller.
Uses ctypes to talk to vJoyInterface.dll directly (more reliable than pyvjoy).

vJoy axis range: 0x1 to 0x8000 (1 to 32768)
  Center = 16384
  Min    = 1
  Max    = 32768

Axes:
  X  (0x30) = Steering   (-1.0 full left … +1.0 full right)
  Y  (0x31) = Throttle   (0.0 off … 1.0 full)
  Z  (0x32) = Brake      (0.0 off … 1.0 full)

Buttons:
  1 = Shift Up   (paddle right)
  2 = Shift Down (paddle left)

Install vJoy driver: https://sourceforge.net/projects/vjoystick/
"""

import time
import ctypes
import logging

logger = logging.getLogger(__name__)

# vJoy axis constants
VJOY_MIN = 1
VJOY_MAX = 0x8000  # 32768
VJOY_CENTER = (VJOY_MAX + VJOY_MIN) // 2  # 16384

# vJoy axis IDs (HID usage codes)
AXIS_STEERING = 0x30  # X  (axis 1)
AXIS_THROTTLE = 0x31  # Y  (axis 2)
AXIS_BRAKE    = 0x32  # Z  (axis 3)

# vJoy button IDs for paddle shifters (vJoy Device 1 has buttons 1-8)
BTN_SHIFT_UP   = 1
BTN_SHIFT_DOWN = 2

# How long to hold the shift button (seconds) — iRacing needs ~50-100ms
SHIFT_PULSE_SEC = 0.08

# vJoy DLL path
VJOY_DLL_PATH = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"

# VJD status codes
VJD_STAT_OWN  = 0  # Device is owned by this process
VJD_STAT_FREE = 1  # Device is free
VJD_STAT_BUSY = 2  # Device is owned by another process
VJD_STAT_MISS = 3  # Device is not installed or disabled


def _load_vjoy_dll():
    """Load the vJoy DLL. Returns the DLL handle or None."""
    try:
        dll = ctypes.WinDLL(VJOY_DLL_PATH)
        if dll.vJoyEnabled():
            return dll
        else:
            logger.warning("vJoy driver is installed but not enabled")
            return None
    except OSError:
        logger.warning(f"vJoy DLL not found at {VJOY_DLL_PATH}")
        return None


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
        scaled = (value + 1.0) / 2.0
    else:
        scaled = value

    raw = int(VJOY_MIN + scaled * (VJOY_MAX - VJOY_MIN))
    return max(VJOY_MIN, min(VJOY_MAX, raw))


class VJoyController:
    """
    Sends throttle, brake, steering, and gear shifts to iRacing via vJoy.
    Uses ctypes DLL directly for reliable device communication.

    Usage:
        ctrl = VJoyController(device_id=1)
        ctrl.connect()
        ctrl.set_inputs(throttle=0.8, brake=0.0, steering=0.1)
        ctrl.shift_up()
        ctrl.release()
    """

    def __init__(self, device_id: int = 1):
        self.device_id = device_id
        self._dll = None
        self._connected = False

        # Track last sent values
        self._last_throttle = -1.0
        self._last_brake = -1.0
        self._last_steering = -99.0
        self._current_gear = 0

    def connect(self) -> bool:
        """Acquire vJoy device. Returns True if successful."""
        self._dll = _load_vjoy_dll()
        if self._dll is None:
            logger.error("Cannot connect: vJoy DLL not available")
            return False

        # Check device status
        status = self._dll.GetVJDStatus(self.device_id)
        status_names = {0: "OWN", 1: "FREE", 2: "BUSY", 3: "MISSING"}
        logger.info(f"vJoy Device {self.device_id} status: {status_names.get(status, 'UNKNOWN')}")

        # If we already own it, relinquish first for a clean start
        if status == VJD_STAT_OWN:
            self._dll.RelinquishVJD(self.device_id)
            status = self._dll.GetVJDStatus(self.device_id)

        # If busy, try to relinquish (works if same process held it)
        if status == VJD_STAT_BUSY:
            logger.warning("Device is BUSY — attempting to relinquish...")
            self._dll.RelinquishVJD(self.device_id)
            status = self._dll.GetVJDStatus(self.device_id)
            if status == VJD_STAT_BUSY:
                logger.error("Device still BUSY — another process has it locked.")
                logger.error("Kill stale Python processes or reboot to free it.")
                return False

        if status == VJD_STAT_MISS:
            logger.error(f"vJoy Device {self.device_id} not configured. "
                         "Open 'Configure vJoy' and enable Device 1.")
            return False

        # Acquire
        result = self._dll.AcquireVJD(self.device_id)
        if not result:
            logger.error(f"Failed to acquire vJoy Device {self.device_id}")
            return False

        self._connected = True

        # Log axis/button info
        axes = []
        for axis_id, name in [(0x30, "X"), (0x31, "Y"), (0x32, "Z")]:
            if self._dll.GetVJDAxisExist(self.device_id, axis_id):
                axes.append(name)
        n_buttons = self._dll.GetVJDButtonNumber(self.device_id)
        logger.info(f"vJoy Device {self.device_id} acquired — axes: {axes}, buttons: {n_buttons}")

        # Zero everything on connect
        self.set_inputs(0.0, 0.0, 0.0)
        self._dll.SetBtn(0, self.device_id, BTN_SHIFT_UP)
        self._dll.SetBtn(0, self.device_id, BTN_SHIFT_DOWN)

        return True

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

        # Clamp
        throttle = max(0.0, min(1.0, throttle))
        brake = max(0.0, min(1.0, brake))
        steering = max(-1.0, min(1.0, steering))

        try:
            dev = self.device_id
            self._dll.SetAxis(_to_vjoy_axis(steering, centered=True), dev, AXIS_STEERING)
            self._dll.SetAxis(_to_vjoy_axis(throttle, centered=False), dev, AXIS_THROTTLE)
            self._dll.SetAxis(_to_vjoy_axis(brake, centered=False), dev, AXIS_BRAKE)

            self._last_throttle = throttle
            self._last_brake = brake
            self._last_steering = steering
            return True

        except Exception as e:
            logger.error(f"vJoy write error: {e}")
            return False

    def shift_up(self) -> bool:
        """Pulse the shift-up button (paddle right)."""
        if not self._connected:
            return False

        self._current_gear += 1
        try:
            self._dll.SetBtn(1, self.device_id, BTN_SHIFT_UP)
            time.sleep(SHIFT_PULSE_SEC)
            self._dll.SetBtn(0, self.device_id, BTN_SHIFT_UP)
            logger.info(f"Shift UP → gear {self._current_gear}")
            return True
        except Exception as e:
            logger.error(f"Shift up error: {e}")
            return False

    def shift_down(self) -> bool:
        """Pulse the shift-down button (paddle left)."""
        if not self._connected:
            return False

        self._current_gear = max(0, self._current_gear - 1)
        try:
            self._dll.SetBtn(1, self.device_id, BTN_SHIFT_DOWN)
            time.sleep(SHIFT_PULSE_SEC)
            self._dll.SetBtn(0, self.device_id, BTN_SHIFT_DOWN)
            logger.info(f"Shift DOWN → gear {self._current_gear}")
            return True
        except Exception as e:
            logger.error(f"Shift down error: {e}")
            return False

    def shift_to(self, target_gear: int) -> bool:
        """
        Shift to a specific gear by pulsing up/down as needed.

        Args:
            target_gear: Target gear number (1-6 typical, 0 = neutral)
        """
        if not self._connected:
            return False

        target_gear = max(0, target_gear)
        shifts_needed = target_gear - self._current_gear

        if shifts_needed == 0:
            return True

        logger.info(f"Shifting {self._current_gear} → {target_gear} ({shifts_needed:+d})")

        if shifts_needed > 0:
            for _ in range(shifts_needed):
                self.shift_up()
                time.sleep(0.02)
        else:
            for _ in range(abs(shifts_needed)):
                self.shift_down()
                time.sleep(0.02)

        return True

    def release(self):
        """Release all inputs to safe state."""
        if self._connected:
            self.set_inputs(0.0, 0.0, 0.0)
            self._dll.SetBtn(0, self.device_id, BTN_SHIFT_UP)
            self._dll.SetBtn(0, self.device_id, BTN_SHIFT_DOWN)
            logger.info("vJoy inputs released to safe state")

    def disconnect(self):
        """Release inputs and free vJoy device."""
        if self._connected:
            self.release()
            self._dll.RelinquishVJD(self.device_id)
            logger.info(f"vJoy Device {self.device_id} relinquished")
        self._connected = False
        self._dll = None

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
        self._current_gear = 0

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockController connected (no hardware output)")
        return True

    def set_inputs(self, throttle: float, brake: float, steering: float) -> bool:
        logger.info(f"[OUTPUT] thr={throttle:.4f}  brk={brake:.4f}  str={steering:.4f}")
        return True

    def shift_up(self) -> bool:
        self._current_gear += 1
        logger.info(f"[OUTPUT] SHIFT UP → gear {self._current_gear}")
        return True

    def shift_down(self) -> bool:
        self._current_gear = max(0, self._current_gear - 1)
        logger.info(f"[OUTPUT] SHIFT DOWN → gear {self._current_gear}")
        return True

    def shift_to(self, target_gear: int) -> bool:
        shifts = target_gear - self._current_gear
        logger.info(f"[OUTPUT] SHIFT {self._current_gear} → {target_gear} ({shifts:+d})")
        self._current_gear = max(0, target_gear)
        return True

    def release(self):
        logger.info("[OUTPUT] Released")

    def disconnect(self):
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected
