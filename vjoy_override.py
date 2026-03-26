"""
vjoy_override.py
================
Temporarily hijacks a vJoy virtual device with a physical gamepad,
then releases it back to virtual/programmatic control on demand.

Works great alongside your iRacing AI bot — press the toggle button
on your gamepad to take over, drive manually, then hand back to the bot.

Requirements:
    pip install pyvjoy pygame

Usage:
    python vjoy_override.py
    python vjoy_override.py --list-axes    # discover axis/button indices

Gamepad default mapping (Xbox / PS layout — edit GAMEPAD_MAP below):
    Left stick X        → Steering
    Right trigger (R2)  → Throttle
    Left trigger (L2)   → Brake
    Left stick Y (inv.) → Clutch  (set clutch_axis=None to disable)
    All buttons         → Passed through to vJoy buttons 1–16
    START / OPTIONS     → Toggle MANUAL ↔ VIRTUAL mode
    SELECT / SHARE      → Hold 1 s to quit
"""

import sys
import time
import threading

try:
    import pygame
except ImportError:
    print("[ERROR] pygame not installed.  Run: pip install pygame")
    sys.exit(1)

try:
    import pyvjoy
except ImportError:
    print("[ERROR] pyvjoy not installed.  Run: pip install pyvjoy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# ── Configuration ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

VJOY_DEVICE_ID  = 1     # vJoy device to control (1-based)
GAMEPAD_INDEX   = 0     # pygame joystick index (0 = first gamepad)
UPDATE_HZ       = 60    # vJoy output rate

# Deadzone applied to stick axes (0.0–1.0)
DEADZONE = 0.05

# ── Gamepad axis / button mapping ───────────────────────────────────────────
# Run with --list-axes to discover your controller's indices.
#
# Xbox One / Series (XInput via pygame):
#   Axis 0 = Left stick X      Axis 1 = Left stick Y
#   Axis 2 = Right stick X     Axis 3 = Right stick Y
#   Axis 4 = Left trigger      Axis 5 = Right trigger  (-1 released → +1 full)
#   Button 6 = Back/Select     Button 7 = Start
#
# PS4 DualShock via DirectInput may differ — use --list-axes to check.

GAMEPAD_MAP = {
    "steer_axis":       0,      # Left stick X  → Steering   (center = 0)
    "throttle_axis":    5,      # Right trigger → Throttle
    "brake_axis":       4,      # Left trigger  → Brake
    "clutch_axis":      None,   # Set to an axis index to enable, or None

    # Triggers ship as -1.0 (released) … +1.0 (pressed).
    # True  = remap so 0.0 = not pressed, 1.0 = full press  ← use for Xbox
    # False = pass through as-is  ← use if your triggers start at 0
    "trigger_remap":    True,

    # Button that TOGGLES manual/virtual (rising-edge = single press)
    "toggle_button":    7,      # Xbox=Start(7)   PS4=Options(9)

    # Button that QUITS — hold for QUIT_HOLD_SEC seconds
    "quit_button":      6,      # Xbox=Back(6)    PS4=Share(8)
}

QUIT_HOLD_SEC = 1.0     # how long to hold quit button


# ---------------------------------------------------------------------------
# ── vJoy constants ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

AXIS_MIN    = 0x0001    # 1
AXIS_MAX    = 0x7FFF    # 32767
AXIS_CENTER = 0x3FFF    # ~16383


# ---------------------------------------------------------------------------
# ── Math helpers ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def clamp(v, lo=AXIS_MIN, hi=AXIS_MAX):
    return max(lo, min(hi, v))


def apply_deadzone(v: float, dz: float) -> float:
    if abs(v) < dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


def stick_to_vjoy(raw: float) -> int:
    """Centered axis  -1.0 … +1.0  →  vJoy 1 … 32767."""
    return clamp(int(AXIS_CENTER + raw * (AXIS_MAX - AXIS_CENTER)))


def trigger_to_vjoy(raw: float, remap: bool) -> int:
    """
    Trigger axis  -1.0 (released) … +1.0 (full)  →  vJoy 1 … 32767.
    With remap=True we normalize to 0.0–1.0 first.
    """
    if remap:
        raw = (raw + 1.0) / 2.0
    return clamp(int(AXIS_MIN + raw * (AXIS_MAX - AXIS_MIN)))


# ---------------------------------------------------------------------------
# ── Shared state ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class ControlState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.steering       = AXIS_CENTER
        self.throttle       = AXIS_MIN
        self.brake          = AXIS_MIN
        self.clutch         = AXIS_CENTER
        self.buttons        = 0           # bitmask, up to 32 buttons
        self.manual_active  = False
        self.running        = True

    def snapshot(self):
        with self.lock:
            return dict(
                steering  = self.steering,
                throttle  = self.throttle,
                brake     = self.brake,
                clutch    = self.clutch,
                buttons   = self.buttons,
                manual    = self.manual_active,
            )


# ---------------------------------------------------------------------------
# ── vJoy wrapper ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class VJoyController:
    def __init__(self, device_id: int):
        self.device_id  = device_id
        self.j          = pyvjoy.VJoyDevice(device_id)
        self._acquired  = False

    def acquire(self):
        if not self._acquired:
            self.j.reset()
            self._acquired = True
            print(f"\n[vJoy] ✋ Device {self.device_id} ACQUIRED — MANUAL override active")

    def relinquish(self):
        if self._acquired:
            self.j.reset()
            self._acquired = False
            print(f"\n[vJoy] 🤖 Device {self.device_id} RELINQUISHED — back to VIRTUAL")

    def send(self, s: dict):
        self.j.set_axis(pyvjoy.HID_USAGE_SL0, s["steering"])
        self.j.set_axis(pyvjoy.HID_USAGE_X,   s["throttle"])
        self.j.set_axis(pyvjoy.HID_USAGE_Y,   s["brake"])
        self.j.set_axis(pyvjoy.HID_USAGE_Z,   s["clutch"])
        for bit in range(16):
            self.j.set_button(bit + 1, bool(s["buttons"] & (1 << bit)))

    def reset_neutral(self):
        self.j.reset()


# ---------------------------------------------------------------------------
# ── Gamepad polling thread ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def gamepad_thread(state: ControlState, vjoy: VJoyController):
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[ERROR] No gamepad detected. Plug one in and restart.")
        with state.lock:
            state.running = False
        return

    js = pygame.joystick.Joystick(GAMEPAD_INDEX)
    js.init()
    print(f"[Gamepad] Connected: {js.get_name()}  "
          f"({js.get_numaxes()} axes / {js.get_numbuttons()} buttons)\n")

    gm              = GAMEPAD_MAP
    toggle_btn      = gm["toggle_button"]
    quit_btn        = gm["quit_button"]
    trigger_remap   = gm["trigger_remap"]

    prev_toggle     = False
    quit_held_since = None
    clock           = pygame.time.Clock()

    def safe_axis(idx, default=0.0):
        if idx is None:
            return default
        try:
            return js.get_axis(idx)
        except Exception:
            return default

    def safe_button(idx):
        if idx is None or idx >= js.get_numbuttons():
            return False
        return bool(js.get_button(idx))

    while True:
        with state.lock:
            if not state.running:
                break

        pygame.event.pump()

        # ── Read axes ──────────────────────────────────────────────────────
        steer_raw    = apply_deadzone(safe_axis(gm["steer_axis"]), DEADZONE)
        throttle_raw = safe_axis(gm["throttle_axis"], default=-1.0)
        brake_raw    = safe_axis(gm["brake_axis"],    default=-1.0)
        clutch_raw   = apply_deadzone(safe_axis(gm["clutch_axis"]), DEADZONE)

        steer_v    = stick_to_vjoy(steer_raw)
        throttle_v = trigger_to_vjoy(throttle_raw, trigger_remap)
        brake_v    = trigger_to_vjoy(brake_raw,    trigger_remap)
        clutch_v   = stick_to_vjoy(clutch_raw)

        # ── Read buttons (pass-through bitmask) ────────────────────────────
        btn_mask = 0
        for b in range(min(js.get_numbuttons(), 32)):
            if js.get_button(b):
                btn_mask |= (1 << b)

        # ── Toggle mode on rising edge ─────────────────────────────────────
        toggle_now = safe_button(toggle_btn)
        if toggle_now and not prev_toggle:
            with state.lock:
                state.manual_active = not state.manual_active
                is_manual = state.manual_active
            if is_manual:
                vjoy.acquire()
            else:
                vjoy.relinquish()
        prev_toggle = toggle_now

        # ── Quit on hold ───────────────────────────────────────────────────
        if safe_button(quit_btn):
            if quit_held_since is None:
                quit_held_since = time.monotonic()
            elif time.monotonic() - quit_held_since >= QUIT_HOLD_SEC:
                print("\n[INFO] Quit button held — exiting.")
                with state.lock:
                    state.running = False
                break
        else:
            quit_held_since = None

        # ── Write to shared state ─────────────────────────────────────────
        with state.lock:
            state.steering  = steer_v
            state.throttle  = throttle_v
            state.brake     = brake_v
            state.clutch    = clutch_v
            state.buttons   = btn_mask

        clock.tick(UPDATE_HZ)

    pygame.quit()


# ---------------------------------------------------------------------------
# ── vJoy output thread ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def output_thread(state: ControlState, vjoy: VJoyController):
    interval = 1.0 / UPDATE_HZ
    while True:
        with state.lock:
            if not state.running:
                break
            manual = state.manual_active

        if manual:
            snap = state.snapshot()
            try:
                vjoy.send(snap)
            except Exception as e:
                print(f"[ERROR] vJoy send: {e}")

        time.sleep(interval)


# ---------------------------------------------------------------------------
# ── Console HUD ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def display_thread(state: ControlState):
    BAR = 22

    def bar(val, lo=AXIS_MIN, hi=AXIS_MAX):
        pct    = (val - lo) / (hi - lo)
        filled = int(pct * BAR)
        return "[" + "█" * filled + "·" * (BAR - filled) + f"] {pct*100:5.1f}%"

    def cbar(val):
        """Centered bar for steering/clutch."""
        pct  = (val - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)
        dev  = (pct - 0.5) * 2          # -1 … +1
        half = BAR // 2
        pos  = int(dev * half)
        buf  = [" "] * BAR
        buf[half] = "│"
        lo = min(half, half + pos)
        hi = max(half, half + pos)
        for i in range(lo, hi + 1):
            if 0 <= i < BAR:
                buf[i] = "█"
        buf[half] = "┼"
        return "[" + "".join(buf) + f"] {dev*100:+6.1f}%"

    HUD_HEIGHT = 16
    print("\n" * HUD_HEIGHT)

    while True:
        with state.lock:
            if not state.running:
                break
            snap = state.snapshot()

        mode = " ✋ MANUAL  " if snap["manual"] else " 🤖 VIRTUAL "
        btns = f"{snap['buttons']:016b}"

        rows = [
            "",
            "  ╔════════════════════════════════════════════════════════╗",
            f"  ║  vJoy Gamepad Override              Mode:{mode}        ║",
            "  ╠════════════════════════════════════════════════════════╣",
            f"  ║  Steer    {cbar(snap['steering'])}    ║",
            f"  ║  Throttle {bar(snap['throttle'])}          ║",
            f"  ║  Brake    {bar(snap['brake'])}          ║",
            f"  ║  Clutch   {cbar(snap['clutch'])}    ║",
            "  ╠════════════════════════════════════════════════════════╣",
            f"  ║  Buttons  {btns[:8]} {btns[8:]}                    ║",
            "  ╠════════════════════════════════════════════════════════╣",
            "  ║  START/Options = toggle MANUAL ↔ VIRTUAL              ║",
            "  ║  SELECT/Share  = hold 1 s to quit                     ║",
            "  ╚════════════════════════════════════════════════════════╝",
        ]

        sys.stdout.write(f"\033[{HUD_HEIGHT}F")
        sys.stdout.write("\n".join(rows) + "\n")
        sys.stdout.flush()

        time.sleep(0.08)


# ---------------------------------------------------------------------------
# ── --list-axes helper ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def list_axes():
    """Print all axes and buttons for every connected gamepad, then watch live."""
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("No gamepad detected.")
        pygame.quit()
        return

    pads = []
    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        pads.append(js)
        print(f"\nGamepad [{i}]: {js.get_name()}")
        print(f"  {js.get_numaxes()} axes    {js.get_numbuttons()} buttons")

    print("\nMove sticks / press buttons to see live values. Ctrl+C to stop.\n")
    clock = pygame.time.Clock()
    try:
        while True:
            pygame.event.pump()
            for js in pads:
                axes    = [f"{js.get_axis(a):+.3f}" for a in range(js.get_numaxes())]
                buttons = [str(js.get_button(b)) for b in range(js.get_numbuttons())]
                line    = f"  [{js.get_id()}] axes=[{', '.join(axes)}]  btns=[{', '.join(buttons)}]"
                sys.stdout.write("\r" + line.ljust(120))
                sys.stdout.flush()
            clock.tick(30)
    except KeyboardInterrupt:
        pass

    print()
    pygame.quit()


# ---------------------------------------------------------------------------
# ── Entry point ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--list-axes" in sys.argv:
        list_axes()
        sys.exit(0)

    print("vJoy Gamepad Override")
    print("=====================")
    print(f"  Gamepad index  : {GAMEPAD_INDEX}  (edit GAMEPAD_INDEX to change)")
    print(f"  vJoy device    : {VJOY_DEVICE_ID}  (edit VJOY_DEVICE_ID to change)")
    print(f"  Toggle button  : {GAMEPAD_MAP['toggle_button']}  "
          f"(START/Options by default)")
    print(f"  Quit button    : {GAMEPAD_MAP['quit_button']}  "
          f"(hold {QUIT_HOLD_SEC:.0f} s — SELECT/Share by default)")
    print()
    print("  Tip: run with --list-axes to discover your controller's indices.")
    print()

    state = ControlState()

    try:
        vjoy = VJoyController(VJOY_DEVICE_ID)
    except Exception as e:
        print(f"[ERROR] Could not open vJoy device {VJOY_DEVICE_ID}: {e}")
        print("Make sure vJoy is installed and the device is configured.")
        sys.exit(1)

    t_out = threading.Thread(target=output_thread,  args=(state, vjoy), daemon=True)
    t_hud = threading.Thread(target=display_thread, args=(state,),      daemon=True)
    t_gp  = threading.Thread(target=gamepad_thread, args=(state, vjoy), daemon=True)

    t_out.start()
    t_hud.start()
    t_gp.start()

    try:
        while True:
            with state.lock:
                if not state.running:
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    with state.lock:
        state.running = False

    if vjoy._acquired:
        vjoy.relinquish()

    time.sleep(0.3)
    print("\n[INFO] Goodbye.")
