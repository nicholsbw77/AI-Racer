"""
setup_vjoy_calib.py

Detects the vJoy device GUID and adds a calibration entry to iRacing's
joyCalib.yaml so you can skip the Calibration Wizard.

Usage:
  python setup_vjoy_calib.py
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

# --- Config ---
IRACING_CALIB = Path(r"G:\Documents\iRacing\joyCalib.yaml")

# vJoy axis range: 0x1 (1) to 0x8000 (32768), center = 16384
VJOY_MIN = 1
VJOY_MAX = 32768
VJOY_CENTER = 16384


def find_vjoy_guid():
    """Try to find vJoy's InstanceGUID from Windows registry."""
    try:
        import winreg
        # vJoy devices are listed under HID
        # Try common registry paths
        paths = [
            r"SYSTEM\CurrentControlSet\Services\vjoy\Enum",
        ]
        for path in paths:
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
                count = winreg.QueryValueEx(key, "Count")[0]
                if count > 0:
                    print(f"  Found {count} vJoy device(s) in registry")
                winreg.CloseKey(key)
            except FileNotFoundError:
                continue
    except Exception as e:
        print(f"  Registry scan note: {e}")

    # Also try via pyvjoy to confirm device works
    try:
        import pyvjoy
        j = pyvjoy.VJoyDevice(1)
        print("  ✓ pyvjoy can open Device 1")
        # Sweep axis briefly to confirm
        j.set_axis(0x30, VJOY_CENTER)
        print("  ✓ Axis write succeeded")
        return True
    except Exception as e:
        print(f"  ✗ pyvjoy error: {e}")
        return False


def build_vjoy_entry(guid: str) -> dict:
    """Build a vJoy calibration entry matching iRacing's format.

    Axes:
      X  (Axis 1) = Steering      (centered)
      Y  (Axis 2) = Throttle      (zero-based)
      Z  (Axis 3) = Brake         (zero-based)
    Buttons (mapped in iRacing Controls, not in calibration):
      Button 1 = Shift Up   (paddle right)
      Button 2 = Shift Down (paddle left)
    """
    return {
        "DeviceName": "vJoy Device",
        "InstanceGUID": guid,
        "AxisList": [
            {
                "Axis": 1,
                "AxisName": "X Axis",
                "CalibMin": VJOY_MIN,
                "CalibCenter": VJOY_CENTER,
                "CalibMax": VJOY_MAX,
            },
            {
                "Axis": 2,
                "AxisName": "Y Axis",
                "CalibMin": VJOY_MIN,
                "CalibCenter": 0,
                "CalibMax": VJOY_MAX,
            },
            {
                "Axis": 3,
                "AxisName": "Z Axis",
                "CalibMin": VJOY_MIN,
                "CalibCenter": 0,
                "CalibMax": VJOY_MAX,
            },
        ],
    }


def main():
    print("=" * 50)
    print("  vJoy Calibration Setup for iRacing")
    print("=" * 50)

    # Step 1: Verify vJoy is working
    print("\n[1] Checking vJoy...")
    vjoy_ok = find_vjoy_guid()

    if not vjoy_ok:
        print("\n⚠  vJoy doesn't seem to be working.")
        print("   Install vJoy and enable Device 1 first.")
        resp = input("   Continue anyway? (y/n): ").strip().lower()
        if resp != "y":
            sys.exit(1)

    # Step 2: Read existing joyCalib.yaml
    print(f"\n[2] Reading {IRACING_CALIB}...")
    if not IRACING_CALIB.exists():
        print(f"  ✗ File not found: {IRACING_CALIB}")
        print("  Make sure iRacing docs path is correct.")
        sys.exit(1)

    with open(IRACING_CALIB, "r") as f:
        calib = yaml.safe_load(f)

    devices = calib.get("CalibrationInfo", {}).get("DeviceList", [])
    print(f"  Found {len(devices)} existing device(s):")
    for d in devices:
        print(f"    - {d['DeviceName']} ({d['InstanceGUID']})")

    # Check if vJoy already exists
    vjoy_existing = [d for d in devices if "vjoy" in d.get("DeviceName", "").lower()]
    if vjoy_existing:
        print(f"\n  vJoy entry already exists! GUID: {vjoy_existing[0]['InstanceGUID']}")
        resp = input("  Overwrite it? (y/n): ").strip().lower()
        if resp == "y":
            devices = [d for d in devices if "vjoy" not in d.get("DeviceName", "").lower()]
        else:
            print("  Keeping existing entry. Done.")
            return

    # Step 3: Get vJoy GUID
    print("\n[3] We need the vJoy InstanceGUID.")
    print("    To find it:")
    print("    1. Open Windows 'Set up USB game controllers' (joy.cpl)")
    print("    2. Or check Device Manager → Human Interface Devices → vJoy")
    print()
    print("    Common vJoy GUIDs look like: {12345678-1234-1234-1234-123456789ABC}")
    print()
    guid = input("    Paste your vJoy GUID (or press Enter to use a placeholder): ").strip()

    if not guid:
        guid = "{00000000-0000-0000-0000-000000000000}"
        print(f"    Using placeholder: {guid}")
        print("    ⚠  You may need to update this later if iRacing doesn't recognize it.")

    # Step 4: Backup and write
    print(f"\n[4] Backing up and writing...")
    backup = IRACING_CALIB.with_suffix(
        f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    shutil.copy2(IRACING_CALIB, backup)
    print(f"  Backup saved: {backup}")

    vjoy_entry = build_vjoy_entry(guid)
    devices.append(vjoy_entry)
    calib["CalibrationInfo"]["DeviceList"] = devices

    with open(IRACING_CALIB, "w") as f:
        yaml.dump(calib, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ vJoy entry added to {IRACING_CALIB}")
    print()
    print("  vJoy calibration values:")
    print(f"    Steering (X):  min={VJOY_MIN}, center={VJOY_CENTER}, max={VJOY_MAX}")
    print(f"    Throttle (Y):  min={VJOY_MIN}, center=0, max={VJOY_MAX}")
    print(f"    Brake    (Z):  min={VJOY_MIN}, center=0, max={VJOY_MAX}")
    print(f"    Shift Up:      vJoy Button 1  (map in iRacing Controls)")
    print(f"    Shift Down:    vJoy Button 2  (map in iRacing Controls)")
    print()
    print("  ✓ Done! Restart iRacing for changes to take effect.")
    print("    Then map vJoy axes in Options → Controls.")


if __name__ == "__main__":
    main()
