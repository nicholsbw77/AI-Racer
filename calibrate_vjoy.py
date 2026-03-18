"""
calibrate_vjoy.py

Interactive calibration helper for iRacing's Calibration Wizard.
Uses ctypes DLL directly (no pyvjoy).

Usage:
  1. Open iRacing → Options → Controls → Calibrate
  2. Run: python calibrate_vjoy.py
  3. Follow the prompts — it will sweep each axis when you're ready.

Press Ctrl+C at any time to stop safely.
"""

import ctypes
import time
import sys

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"
VJOY_MIN = 1
VJOY_MAX = 32768
VJOY_CENTER = 16384

AXIS_X = 0x30  # Steering
AXIS_Y = 0x31  # Throttle
AXIS_Z = 0x32  # Brake


def main():
    print("=" * 55)
    print("  vJoy Calibration Helper for iRacing")
    print("  Using direct DLL — no pyvjoy needed")
    print("=" * 55)

    # Load DLL
    try:
        dll = ctypes.WinDLL(VJOY_DLL)
    except OSError:
        print(f"\nERROR: Cannot load {VJOY_DLL}")
        sys.exit(1)

    if not dll.vJoyEnabled():
        print("\nERROR: vJoy not enabled")
        sys.exit(1)

    # Acquire device
    dll.RelinquishVJD(1)  # clean slate
    if not dll.AcquireVJD(1):
        print("\nERROR: Cannot acquire vJoy Device 1")
        sys.exit(1)

    print("\n✓ vJoy Device 1 acquired\n")

    # Set initial state: steering centered, pedals at zero
    dll.SetAxis(VJOY_CENTER, 1, AXIS_X)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Y)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Z)
    print("  Initial state: Steering=CENTER, Throttle=0, Brake=0\n")

    try:
        # ── STEERING ──
        print("=" * 55)
        print("  STEERING (X Axis)")
        print("  iRacing expects: full left → full right → center")
        print("=" * 55)
        input("  Press ENTER when iRacing wizard is on Steering step...")

        print("  Holding CENTER for 1s...")
        dll.SetAxis(VJOY_CENTER, 1, AXIS_X)
        time.sleep(1)

        print("  → Full LEFT...")
        for i in range(50):
            val = int(VJOY_CENTER - (VJOY_CENTER - VJOY_MIN) * (i / 49))
            dll.SetAxis(val, 1, AXIS_X)
            time.sleep(0.03)
        time.sleep(0.5)

        print("  → Full RIGHT...")
        for i in range(100):
            val = int(VJOY_MIN + (VJOY_MAX - VJOY_MIN) * (i / 99))
            dll.SetAxis(val, 1, AXIS_X)
            time.sleep(0.03)
        time.sleep(0.5)

        print("  → Back to CENTER...")
        for i in range(50):
            val = int(VJOY_MAX - (VJOY_MAX - VJOY_CENTER) * (i / 49))
            dll.SetAxis(val, 1, AXIS_X)
            time.sleep(0.03)

        print("  ✓ Steering done — holding at center")
        print("  Click 'Next' in iRacing\n")

        # ── THROTTLE ──
        print("=" * 55)
        print("  THROTTLE (Y Axis)")
        print("  iRacing expects: released → fully pressed → released")
        print("=" * 55)
        input("  Press ENTER when iRacing wizard is on Throttle step...")

        print("  Holding at 0% for 1s...")
        dll.SetAxis(VJOY_MIN, 1, AXIS_Y)
        time.sleep(1)

        print("  → Pressing to 100%...")
        for i in range(100):
            val = int(VJOY_MIN + (VJOY_MAX - VJOY_MIN) * (i / 99))
            dll.SetAxis(val, 1, AXIS_Y)
            time.sleep(0.02)
        time.sleep(0.5)

        print("  → Releasing to 0%...")
        for i in range(100):
            val = int(VJOY_MAX - (VJOY_MAX - VJOY_MIN) * (i / 99))
            dll.SetAxis(val, 1, AXIS_Y)
            time.sleep(0.02)

        print("  ✓ Throttle done — holding at 0%")
        print("  Click 'Next' in iRacing\n")

        # ── BRAKE ──
        print("=" * 55)
        print("  BRAKE (Z Axis)")
        print("  iRacing expects: released → fully pressed → released")
        print("=" * 55)
        input("  Press ENTER when iRacing wizard is on Brake step...")

        print("  Holding at 0% for 1s...")
        dll.SetAxis(VJOY_MIN, 1, AXIS_Z)
        time.sleep(1)

        print("  → Pressing to 100%...")
        for i in range(100):
            val = int(VJOY_MIN + (VJOY_MAX - VJOY_MIN) * (i / 99))
            dll.SetAxis(val, 1, AXIS_Z)
            time.sleep(0.02)
        time.sleep(0.5)

        print("  → Releasing to 0%...")
        for i in range(100):
            val = int(VJOY_MAX - (VJOY_MAX - VJOY_MIN) * (i / 99))
            dll.SetAxis(val, 1, AXIS_Z)
            time.sleep(0.02)

        print("  ✓ Brake done — holding at 0%")
        print("  Click 'Next' in iRacing\n")

        # ── DONE ──
        print("=" * 55)
        print("  ✓ All axes calibrated!")
        print("  Finish the wizard in iRacing.")
        print()
        print("  Then in Controls, also map:")
        print("    vJoy Button 1 → Shift Up")
        print("    vJoy Button 2 → Shift Down")
        print("=" * 55)

        input("\n  Press ENTER to release vJoy device...")

    except KeyboardInterrupt:
        print("\n\n  Stopped by user.")

    # Cleanup
    dll.SetAxis(VJOY_CENTER, 1, AXIS_X)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Y)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Z)
    dll.RelinquishVJD(1)
    print("  vJoy device released. Done.")


if __name__ == "__main__":
    main()
