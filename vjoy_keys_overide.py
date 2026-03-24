"""
vjoy_override.py
================
Temporarily hijacks vJoy device axes/buttons with manual control,
then releases back to virtual (programmatic) control.

Requirements:
    pip install pyvjoy keyboard pywin32

Usage:
    python vjoy_override.py

Controls (when in MANUAL mode):
    W/S         - Throttle up/down
    A/D         - Steering left/right
    Q/E         - Brake up/down
    SPACE       - Toggle Button 1 (e.g. handbrake)
    TAB         - Toggle MANUAL/VIRTUAL mode
    ESC         - Exit
"""

import time
import threading
import sys
import ctypes

try:
    import pyvjoy
except ImportError:
    print("[ERROR] pyvjoy not installed. Run: pip install pyvjoy")
    sys.exit(1)

try:
    import keyboard
except ImportError:
    print("[ERROR] keyboard not installed. Run: pip install keyboard")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VJOY_DEVICE_ID = 1          # vJoy device number (1-based)
AXIS_MIN       = 0x0001     # vJoy axis minimum
AXIS_MAX       = 0x7FFF     # vJoy axis maximum (32767)
AXIS_CENTER    = 0x3FFF     # Center / neutral (~16383)

STEP_LARGE     = 0x0800     # ~6% per keypress
STEP_SMALL     = 0x0200     # ~1.5% per keypress

UPDATE_HZ      = 60         # How often the output thread sends state to vJoy


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ControlState:
    """Shared mutable state between keyboard thread and output thread."""

    def __init__(self):
        self.lock = threading.Lock()

        # Axes (vJoy range 1–32767)
        self.steering  = AXIS_CENTER
        self.throttle  = AXIS_MIN
        self.brake     = AXIS_MIN
        self.clutch    = AXIS_CENTER

        # Buttons (bit flags, up to 32)
        self.buttons   = 0

        # Mode
        self.manual_active = False   # True = override active
        self.running       = True

    # --- helpers ---

    def clamp_axis(self, v):
        return max(AXIS_MIN, min(AXIS_MAX, v))

    def nudge(self, attr, delta):
        with self.lock:
            setattr(self, attr, self.clamp_axis(getattr(self, attr) + delta))

    def toggle_button(self, bit):
        """bit is 0-indexed (0 = Button 1)."""
        with self.lock:
            self.buttons ^= (1 << bit)

    def snapshot(self):
        with self.lock:
            return {
                "steering":  self.steering,
                "throttle":  self.throttle,
                "brake":     self.brake,
                "clutch":    self.clutch,
                "buttons":   self.buttons,
                "manual":    self.manual_active,
            }


# ---------------------------------------------------------------------------
# vJoy interface
# ---------------------------------------------------------------------------

class VJoyController:
    """Thin wrapper around pyvjoy with acquire/relinquish semantics."""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.j = pyvjoy.VJoyDevice(device_id)
        self._acquired = False

    def acquire(self):
        """Claim exclusive control of the vJoy device."""
        if not self._acquired:
            # pyvjoy acquires on construction, but we track our own flag
            # so we know when we've "officially" taken over
            self.j.reset()
            self._acquired = True
            print(f"[vJoy] Device {self.device_id} ACQUIRED (manual override active)")

    def relinquish(self):
        """Return device to normal virtual (programmatic) use."""
        if self._acquired:
            self.j.reset()
            self._acquired = False
            print(f"[vJoy] Device {self.device_id} RELINQUISHED (back to virtual)")

    def send(self, state_snapshot: dict):
        """Push axis and button values to vJoy."""
        s = state_snapshot
        self.j.set_axis(pyvjoy.HID_USAGE_SL0,  s["steering"])   # Steering
        self.j.set_axis(pyvjoy.HID_USAGE_X,    s["throttle"])   # Throttle
        self.j.set_axis(pyvjoy.HID_USAGE_Y,    s["brake"])      # Brake
        self.j.set_axis(pyvjoy.HID_USAGE_Z,    s["clutch"])     # Clutch
        self.j.set_button(1, bool(s["buttons"] & 0x01))         # Button 1
        self.j.set_button(2, bool(s["buttons"] & 0x02))         # Button 2

    def reset_to_neutral(self):
        """Center all axes, release all buttons."""
        self.j.reset()


# ---------------------------------------------------------------------------
# Keyboard input thread
# ---------------------------------------------------------------------------

def keyboard_thread(state: ControlState, vjoy: VJoyController):
    """Runs in background, registers hotkeys and mutates ControlState."""

    def toggle_mode():
        with state.lock:
            state.manual_active = not state.manual_active
            if state.manual_active:
                vjoy.acquire()
            else:
                vjoy.relinquish()

    def quit_app():
        print("\n[INFO] Exiting...")
        with state.lock:
            state.running = False
        if vjoy._acquired:
            vjoy.relinquish()

    # Register global hotkeys
    keyboard.add_hotkey("tab",   toggle_mode,   suppress=True)
    keyboard.add_hotkey("esc",   quit_app,      suppress=True)

    # --- Manual axis controls (only act when manual mode is active) ---

    def act_if_manual(fn):
        def wrapper():
            if state.manual_active:
                fn()
        return wrapper

    keyboard.add_hotkey("w", act_if_manual(lambda: state.nudge("throttle",  STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("s", act_if_manual(lambda: state.nudge("throttle", -STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("a", act_if_manual(lambda: state.nudge("steering", -STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("d", act_if_manual(lambda: state.nudge("steering",  STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("q", act_if_manual(lambda: state.nudge("brake",     STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("e", act_if_manual(lambda: state.nudge("brake",    -STEP_LARGE)), suppress=False)
    keyboard.add_hotkey("space", act_if_manual(lambda: state.toggle_button(0)),          suppress=False)

    keyboard.wait()  # Blocks until program ends


# ---------------------------------------------------------------------------
# Output thread: pushes state to vJoy at fixed rate
# ---------------------------------------------------------------------------

def output_thread(state: ControlState, vjoy: VJoyController):
    interval = 1.0 / UPDATE_HZ
    while True:
        with state.lock:
            running = state.running
            manual  = state.manual_active

        if not running:
            break

        if manual:
            snap = state.snapshot()
            try:
                vjoy.send(snap)
            except Exception as e:
                print(f"[ERROR] vJoy send failed: {e}")

        time.sleep(interval)


# ---------------------------------------------------------------------------
# Display thread: console HUD
# ---------------------------------------------------------------------------

def display_thread(state: ControlState):
    bar_width = 20

    def bar(val, lo=AXIS_MIN, hi=AXIS_MAX):
        pct = (val - lo) / (hi - lo)
        filled = int(pct * bar_width)
        return "[" + "█" * filled + "·" * (bar_width - filled) + f"] {pct*100:5.1f}%"

    while True:
        with state.lock:
            if not state.running:
                break
            snap = {
                "steering":  state.steering,
                "throttle":  state.throttle,
                "brake":     state.brake,
                "clutch":    state.clutch,
                "buttons":   state.buttons,
                "manual":    state.manual_active,
            }

        mode_str = " ✋ MANUAL OVERRIDE " if snap["manual"] else " 🤖 VIRTUAL (bot)  "

        lines = [
            "",
            f"  ┌─────────────────────────────────────────────┐",
            f"  │  vJoy Override Tool          Mode:{mode_str}│",
            f"  ├─────────────────────────────────────────────┤",
            f"  │  Steering  {bar(snap['steering'])}    │",
            f"  │  Throttle  {bar(snap['throttle'])}    │",
            f"  │  Brake     {bar(snap['brake'])}    │",
            f"  │  Clutch    {bar(snap['clutch'])}    │",
            f"  │  Buttons   {snap['buttons']:032b}     │",
            f"  ├─────────────────────────────────────────────┤",
            f"  │  TAB=toggle mode  W/S=throttle  A/D=steer   │",
            f"  │  Q/E=brake        SPACE=Btn1    ESC=quit     │",
            f"  └─────────────────────────────────────────────┘",
        ]

        # Move cursor up and redraw
        sys.stdout.write("\033[F" * (len(lines) + 1))
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()

        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Programmatic API  (use this when integrating with your AI bot)
# ---------------------------------------------------------------------------

class VJoyOverrideController:
    """
    High-level API for programmatic use (e.g. from your iracing-ai bot).

    Example
    -------
    ctrl = VJoyOverrideController(device_id=1)
    ctrl.start()

    # Let bot run normally...

    # Take over:
    ctrl.take_over()
    ctrl.set_throttle(0.5)
    ctrl.set_steering(-0.25)
    time.sleep(2.0)

    # Give back:
    ctrl.release()

    ctrl.stop()
    """

    def __init__(self, device_id: int = 1):
        self.vjoy  = VJoyController(device_id)
        self.state = ControlState()
        self._threads = []

    def start(self):
        t = threading.Thread(target=output_thread, args=(self.state, self.vjoy), daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        with self.state.lock:
            self.state.running = False
        if self.vjoy._acquired:
            self.vjoy.relinquish()

    def take_over(self):
        with self.state.lock:
            self.state.manual_active = True
        self.vjoy.acquire()

    def release(self):
        self.vjoy.relinquish()
        with self.state.lock:
            self.state.manual_active = False

    def set_steering(self, value: float):
        """value: -1.0 (full left) to +1.0 (full right)"""
        raw = int(AXIS_CENTER + value * (AXIS_MAX - AXIS_CENTER))
        with self.state.lock:
            self.state.steering = self.state.clamp_axis(raw)

    def set_throttle(self, value: float):
        """value: 0.0 to 1.0"""
        raw = int(AXIS_MIN + value * (AXIS_MAX - AXIS_MIN))
        with self.state.lock:
            self.state.throttle = self.state.clamp_axis(raw)

    def set_brake(self, value: float):
        """value: 0.0 to 1.0"""
        raw = int(AXIS_MIN + value * (AXIS_MAX - AXIS_MIN))
        with self.state.lock:
            self.state.brake = self.state.clamp_axis(raw)

    def set_button(self, index: int, pressed: bool):
        """index: 0-based button index"""
        with self.state.lock:
            if pressed:
                self.state.buttons |=  (1 << index)
            else:
                self.state.buttons &= ~(1 << index)

    @property
    def is_manual(self):
        with self.state.lock:
            return self.state.manual_active


# ---------------------------------------------------------------------------
# Entry point (interactive keyboard mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("vJoy Override Tool")
    print("==================")
    print("TAB  = toggle between MANUAL and VIRTUAL mode")
    print("ESC  = quit\n")

    # Print blank lines so the HUD has room to draw above
    print("\n" * 14)

    state = ControlState()

    try:
        vjoy = VJoyController(VJOY_DEVICE_ID)
    except Exception as e:
        print(f"[ERROR] Could not open vJoy device {VJOY_DEVICE_ID}: {e}")
        print("Make sure vJoy is installed and device is configured.")
        sys.exit(1)

    # Start threads
    t_out = threading.Thread(target=output_thread,  args=(state, vjoy), daemon=True)
    t_hud = threading.Thread(target=display_thread, args=(state,),      daemon=True)
    t_out.start()
    t_hud.start()

    # Keyboard thread blocks until ESC
    try:
        keyboard_thread(state, vjoy)
    except KeyboardInterrupt:
        pass

    with state.lock:
        state.running = False

    time.sleep(0.2)
    print("\n[INFO] Goodbye.")