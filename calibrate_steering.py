"""
calibrate_steering.py

Calibrates vJoy steering axis (X) for iRacing's Calibration Wizard.
Uses ctypes DLL directly.

Usage:
  1. Open iRacing → Options → Controls → Calibrate → Steering step
  2. Run: python calibrate_steering.py
"""

import ctypes
import time
import sys

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"
VJOY_MIN = 1
VJOY_MAX = 32768
VJOY_CENTER = 16384
AXIS_X = 0x30


def main():
    print("=" * 55)
    print("  vJoy Steering Calibration")
    print("=" * 55)

    try:
        dll = ctypes.WinDLL(VJOY_DLL)
    except OSError:
        print(f"\nERROR: Cannot load {VJOY_DLL}")
        sys.exit(1)

    dll.RelinquishVJD(1)
    if not dll.AcquireVJD(1):
        print("\nERROR: Cannot acquire vJoy Device 1")
        sys.exit(1)

    print("\n✓ vJoy Device 1 acquired")

    # Start at center
    dll.SetAxis(VJOY_CENTER, 1, AXIS_X)
    print("  Steering at CENTER (50%)\n")

    try:
        input("  Press ENTER when iRacing wizard is on Steering step...")

        print("  Holding CENTER for 1s...")
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

        print("\n  ✓ Steering calibrated! Click 'Next' in iRacing.")
        input("  Press ENTER to release vJoy...")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    dll.SetAxis(VJOY_CENTER, 1, AXIS_X)
    dll.RelinquishVJD(1)
    print("  Done.")


if __name__ == "__main__":
    main()
