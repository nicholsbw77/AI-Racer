"""
calibrate_pedals.py

Calibrates vJoy throttle (Y) and brake (Z) axes for iRacing's Calibration Wizard.
Uses ctypes DLL directly.

Usage:
  1. Open iRacing → Options → Controls → Calibrate → Throttle step
  2. Run: python calibrate_pedals.py
"""

import ctypes
import time
import sys

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"
VJOY_MIN = 1
VJOY_MAX = 32768
AXIS_Y = 0x31  # Throttle
AXIS_Z = 0x32  # Brake


def sweep_pedal(dll, axis_id, name):
    """Sweep a pedal axis: 0% → 100% → 0%"""
    input(f"  Press ENTER when iRacing wizard is on {name} step...")

    print(f"  Holding {name} at 0% for 1s...")
    dll.SetAxis(VJOY_MIN, 1, axis_id)
    time.sleep(1)

    print(f"  → Pressing {name} to 100%...")
    for i in range(100):
        val = int(VJOY_MIN + (VJOY_MAX - VJOY_MIN) * (i / 99))
        dll.SetAxis(val, 1, axis_id)
        time.sleep(0.02)
    time.sleep(0.5)

    print(f"  → Releasing {name} to 0%...")
    for i in range(100):
        val = int(VJOY_MAX - (VJOY_MAX - VJOY_MIN) * (i / 99))
        dll.SetAxis(val, 1, axis_id)
        time.sleep(0.02)

    print(f"\n  ✓ {name} calibrated! Click 'Next' in iRacing.\n")


def main():
    print("=" * 55)
    print("  vJoy Pedals Calibration (Throttle + Brake)")
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

    # Start pedals at zero
    dll.SetAxis(VJOY_MIN, 1, AXIS_Y)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Z)
    print("  Throttle at 0%, Brake at 0%\n")

    try:
        sweep_pedal(dll, AXIS_Y, "THROTTLE")
        sweep_pedal(dll, AXIS_Z, "BRAKE")

        print("=" * 55)
        print("  ✓ Both pedals calibrated!")
        print("=" * 55)
        input("\n  Press ENTER to release vJoy...")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    dll.SetAxis(VJOY_MIN, 1, AXIS_Y)
    dll.SetAxis(VJOY_MIN, 1, AXIS_Z)
    dll.RelinquishVJD(1)
    print("  Done.")


if __name__ == "__main__":
    main()
