"""
calibrate_buttons.py

Pulses vJoy buttons for iRacing's shift paddle calibration.
Uses ctypes DLL directly.

Usage:
  1. Open iRacing → Options → Controls → Calibrate → Shift Up step
  2. Run: python calibrate_buttons.py
"""

import ctypes
import time
import sys

VJOY_DLL = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"


def main():
    print("=" * 55)
    print("  vJoy Button Calibration (Paddle Shifters)")
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

    print("\n✓ vJoy Device 1 acquired\n")

    try:
        # ── SHIFT UP ──
        print("=" * 55)
        print("  SHIFT UP (Button 1)")
        print("=" * 55)
        input("  Press ENTER when iRacing wizard is on Shift Up step...")

        print("  Pulsing button 16...")
        dll.SetBtn(1, 1, 1)
        time.sleep(0.3)
        dll.SetBtn(0, 1, 1)

        print("  ✓ Shift Up done! Click 'Next' in iRacing.\n")

        # ── SHIFT DOWN ──
        print("=" * 55)
        print("  SHIFT DOWN (Button 2)")
        print("=" * 55)
        input("  Press ENTER when iRacing wizard is on Shift Down step...")

        print("  Pulsing button 8...")
        dll.SetBtn(1, 1, 2)
        time.sleep(0.3)
        dll.SetBtn(0, 1, 2)

        print("  ✓ Shift Down done! Click 'Next' in iRacing.\n")

        print("=" * 55)
        print("  ✓ Both shifter buttons calibrated!")
        print("=" * 55)
        input("\n  Press ENTER to release vJoy...")

    except KeyboardInterrupt:
        print("\n\n  Stopped.")

    dll.SetBtn(0, 1, 1)
    dll.SetBtn(0, 1, 2)
    dll.RelinquishVJD(1)
    print("  Done.")


if __name__ == "__main__":
    main()
