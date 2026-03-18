"""
vjoy_test.py

Quick diagnostic to verify vJoy is working.
Run this while watching the "Monitor vJoy" app to see if axes move.

Tests multiple axis access methods since pyvjoy versions differ.
"""

import time
import sys

try:
    import pyvjoy
except ImportError:
    print("ERROR: pyvjoy not installed. Run: pip install pyvjoy")
    sys.exit(1)

print(f"pyvjoy version/location: {pyvjoy.__file__}")
print()

# Try to acquire device
try:
    j = pyvjoy.VJoyDevice(1)
    print("✓ vJoy Device 1 acquired successfully")
except Exception as e:
    print(f"✗ Failed to acquire vJoy Device 1: {e}")
    print("  - Is vJoy installed?")
    print("  - Is Device 1 enabled in 'Configure vJoy'?")
    sys.exit(1)

# Show what constants pyvjoy exposes
print("\n--- pyvjoy constants ---")
for name in dir(pyvjoy):
    if "HID" in name or "AXIS" in name or "Usage" in name.lower():
        print(f"  {name} = {getattr(pyvjoy, name, '?')}")

print("\n--- VJoyDevice methods ---")
for name in dir(j):
    if not name.startswith("_"):
        print(f"  {name}")

# Test 1: Try HID_USAGE constants with set_axis
print("\n\n========== TEST 1: set_axis with HID_USAGE ==========")
print("Watch 'Monitor vJoy' — axes should move.\n")

axes_to_test = [
    (0x30, "X / Steering"),
    (0x31, "Y / Throttle"),
    (0x32, "Z / Brake"),
]

for axis_id, name in axes_to_test:
    try:
        print(f"  Moving axis {name} (0x{axis_id:02X}) to MAX (32768)...", end=" ")
        j.set_axis(axis_id, 0x8000)
        time.sleep(1)
        print("MIN (1)...", end=" ")
        j.set_axis(axis_id, 0x1)
        time.sleep(1)
        print("CENTER (16384)...", end=" ")
        j.set_axis(axis_id, 0x4000)
        time.sleep(0.5)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")

# Test 2: Try using data object directly
print("\n\n========== TEST 2: Using data object + update ==========")
try:
    j.data.wAxisX = 0x8000
    j.update()
    print("  Set wAxisX to MAX via data.wAxisX + update()")
    time.sleep(1)

    j.data.wAxisX = 0x1
    j.update()
    print("  Set wAxisX to MIN via data.wAxisX + update()")
    time.sleep(1)

    j.data.wAxisX = 0x4000
    j.data.wAxisY = 0x8000
    j.data.wAxisZ = 0x8000
    j.update()
    print("  Set X=CENTER, Y=MAX, Z=MAX via data + update()")
    time.sleep(1)

    j.data.wAxisY = 0x1
    j.data.wAxisZ = 0x1
    j.update()
    print("  Set Y=MIN, Z=MIN via data + update()")
    time.sleep(1)

    print("  ✓ data+update method works!")
except Exception as e:
    print(f"  ✗ data+update method failed: {e}")

# Reset
print("\n--- Resetting all axes to center/zero ---")
try:
    j.data.wAxisX = 0x4000
    j.data.wAxisY = 0x1
    j.data.wAxisZ = 0x1
    j.update()
    print("Done.")
except:
    pass

print("\n============================================")
print("Did you see axes move in 'Monitor vJoy'?")
print("  If YES to Test 1 → set_axis method works")
print("  If YES to Test 2 → data+update method works")
print("  If NEITHER moved → vJoy driver issue")
print("============================================")
