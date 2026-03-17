"""
inspect_ibt.py

Diagnostic tool for iRacing .ibt telemetry files.
Run this first to verify a file can be read and all required channels exist.

Usage:
  python inspect_ibt.py data/cadillacctsvr_lagunaseca 2023-08-16 00-09-26 (...).ibt
  python inspect_ibt.py data/   # inspect first .ibt found in folder
"""

import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

try:
    import irsdk
    IRSDK_AVAILABLE = True
except ImportError:
    IRSDK_AVAILABLE = False

from ibt_loader import (
    load_ibt_file,
    get_session_info,
    parse_combo_from_filename,
    REQUIRED_CHANNELS,
    IBT_CHANNEL_MAP,
)


def inspect(filepath: str):
    path = Path(filepath)

    # If a folder was given, pick the first .ibt inside it
    if path.is_dir():
        ibt_files = sorted(path.glob("*.ibt"))
        if not ibt_files:
            print(f"No .ibt files found in {path}")
            sys.exit(1)
        path = ibt_files[0]
        print(f"(Auto-selected first file: {path.name})\n")

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    if not IRSDK_AVAILABLE:
        print("ERROR: pyirsdk is not installed.")
        print("Install it: pip install pyirsdk")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Inspecting: {path.name}")
    print(f"{'='*60}\n")

    # --- Filename parsing ---
    car_id, track_id = parse_combo_from_filename(str(path))
    print(f"Filename parse:")
    print(f"  Car   : {car_id}")
    print(f"  Track : {track_id}")
    print()

    # --- Session info ---
    info = get_session_info(str(path))
    if info:
        print("Session info (from .ibt YAML header):")
        for k in ("track_name", "track_id", "track_length", "car_id", "car_name", "sample_hz"):
            if k in info:
                print(f"  {k:15s}: {info[k]}")
        print()

    # --- Channel availability ---
    channels = info.get("channels", [])
    if channels:
        print(f"Available channels ({len(channels)} total):")
        for ch in channels:
            print(f"  {ch}")
        print()

    # --- Required channel check ---
    print("Required channel check:")
    all_present = True
    for iracing_name, canonical_name in IBT_CHANNEL_MAP.items():
        present = iracing_name in channels
        required = iracing_name in REQUIRED_CHANNELS
        status = "✓" if present else ("✗ REQUIRED" if required else "- optional")
        print(f"  {iracing_name:30s} → {canonical_name:15s} {status}")
        if not present and required:
            all_present = False
    print()

    if not all_present:
        print("ERROR: Missing required channels — this file cannot be used for training.")
        sys.exit(1)

    # --- Load data and show value ranges ---
    print("Loading data...")
    df = load_ibt_file(str(path))

    if df is None:
        print("ERROR: load_ibt_file() returned None — check warnings above.")
        sys.exit(1)

    print(f"Loaded {len(df):,} frames\n")

    if "session_time" in df.columns and len(df) > 1:
        duration = df["session_time"].iloc[-1] - df["session_time"].iloc[0]
        print(f"Session duration : {duration:.1f}s ({duration/60:.1f} min)")

    if "sample_hz" in info:
        print(f"Sample rate      : {info['sample_hz']} Hz")
    elif len(df) > 1 and "session_time" in df.columns:
        dt = df["session_time"].diff().dropna()
        dt = dt[dt > 0]
        if len(dt):
            hz = round(1.0 / float(dt.median()))
            print(f"Sample rate      : ~{hz} Hz (estimated)")

    if "lapIndex" in df.columns:
        laps = df["lapIndex"].astype(int)
        n_laps = laps.nunique()
        print(f"Laps detected    : {n_laps}")
        lap_counts = laps.value_counts().sort_index()
        for lap_num, count in lap_counts.items():
            print(f"  Lap {lap_num:3d}: {count:,} frames")

    print()
    print("Value ranges (raw, before normalization):")
    print(f"  {'Column':<20} {'Min':>10} {'Max':>10} {'Mean':>10}")
    print(f"  {'-'*54}")
    for col in df.columns:
        if df[col].dtype in [np.float32, np.float64, float]:
            print(f"  {col:<20} {df[col].min():>10.4f} {df[col].max():>10.4f} {df[col].mean():>10.4f}")

    print()
    print("✓ File looks good — ready for preprocessing.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_ibt.py <path_to_file.ibt>")
        print("       python inspect_ibt.py <folder_containing_ibt_files>")
        sys.exit(1)

    inspect(sys.argv[1])


if __name__ == "__main__":
    main()
