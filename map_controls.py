"""
map_controls.py

Holds each vJoy axis/button active one at a time so you can map them
in iRacing's Options → Controls screen.

Usage:
  1. Open iRacing → Options → Controls
  2. Run: python map_controls.py
  3. Follow the prompts — click the iRacing control, then press Enter here
"""

import ctypes
import time

dll = ctypes.WinDLL(r"C:\Program Files\vJoy\x64\vJoyInterface.dll")
dll.RelinquishVJD(1)

if not dll.AcquireVJD(1):
    print("ERROR: Could not acquire vJoy Device 1")
    exit(1)

print("vJoy Device 1 acquired.\n")
print("In iRacing: Options → Controls")
print("Click the control you want to map, then press ENTER here.\n")

# Reset everything to neutral
dll.SetAxis(16384, 1, 0x30)  # steering center
dll.SetAxis(1, 1, 0x31)      # throttle off
dll.SetAxis(1, 1, 0x32)      # brake off

steps = [
    ("STEERING", "Click 'Steering' in iRacing, press ENTER, then move detected",
     lambda: [dll.SetAxis(1, 1, 0x30), time.sleep(0.3), dll.SetAxis(32767, 1, 0x30), time.sleep(0.3), dll.SetAxis(16384, 1, 0x30)]),

    ("THROTTLE", "Click 'Throttle' in iRacing, press ENTER",
     lambda: [dll.SetAxis(1, 1, 0x31), time.sleep(0.2), dll.SetAxis(32767, 1, 0x31), time.sleep(0.5), dll.SetAxis(1, 1, 0x31)]),

    ("BRAKE", "Click 'Brake' in iRacing, press ENTER",
     lambda: [dll.SetAxis(1, 1, 0x32), time.sleep(0.2), dll.SetAxis(32767, 1, 0x32), time.sleep(0.5), dll.SetAxis(1, 1, 0x32)]),

    ("SHIFT UP", "Click 'Shift Up' in iRacing, press ENTER",
     lambda: [dll.SetBtn(1, 1, 1), time.sleep(0.3), dll.SetBtn(0, 1, 1)]),

    ("SHIFT DOWN", "Click 'Shift Down' in iRacing, press ENTER",
     lambda: [dll.SetBtn(1, 1, 2), time.sleep(0.3), dll.SetBtn(0, 1, 2)]),
]

for name, prompt, action in steps:
    print(f"--- {name} ---")
    input(f"  {prompt}: ")
    print(f"  Sending {name}...")
    action()
    print(f"  Done! iRacing should have detected it.\n")

# Reset to neutral
dll.SetAxis(16384, 1, 0x30)
dll.SetAxis(1, 1, 0x31)
dll.SetAxis(1, 1, 0x32)

dll.RelinquishVJD(1)
print("All controls mapped. You can now run: python orchestrator.py --auto")
