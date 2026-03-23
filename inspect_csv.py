"""
scripts/inspect_csv.py

Quick diagnostic to inspect a VRS CSV file and verify column detection.
Run this first to confirm your VRS exports are being parsed correctly.

Usage:
  python scripts/inspect_csv.py path/to/your_lap.csv
"""

import sys
from pathlib import Path

import pandas as pd
import yaml
from loader import load_vrs_csv, normalize_features, COLUMN_CANDIDATES


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_csv.py <path_to_vrs_csv>")
        sys.exit(1)

    fpath = sys.argv[1]
    print(f"\nInspecting: {fpath}\n")

    # Try raw load first to show all columns
    try:
        raw = pd.read_csv(fpath, sep="\t", nrows=5)
        if len(raw.columns) < 3:
            raw = pd.read_csv(fpath, sep=",", nrows=5)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        sys.exit(1)

    print(f"Detected {len(raw.columns)} columns:")
    for col in raw.columns:
        print(f"  {col}: {raw[col].dtype}  (sample: {raw[col].iloc[0] if len(raw) > 0 else 'N/A'})")

    print("\n--- Column mapping detection ---")
    for key, candidates in COLUMN_CANDIDATES.items():
        found = None
        for c in candidates:
            if c in raw.columns:
                found = c
                break
        status = f"✓ '{found}'" if found else "✗ NOT FOUND"
        print(f"  {key:15s} → {status}")

    print("\n--- Loading with loader.py ---")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = load_vrs_csv(fpath)
    if df is None:
        print("FAILED: load_vrs_csv returned None")
        print("Check that throttle, brake, steering, speed, lap_dist columns are present")
        sys.exit(1)

    print(f"Loaded {len(df)} valid bins")
    print(f"Columns after mapping: {list(df.columns)}")

    df, _ = normalize_features(df, cfg)
    print("\nNormalized value ranges:")
    for col in df.columns:
        if df[col].dtype in ["float32", "float64"]:
            print(f"  {col:20s}: [{df[col].min():.3f}, {df[col].max():.3f}]")

    if "lapIndex" in df.columns:
        n_laps = df["lapIndex"].nunique()
        print(f"\nLaps in file: {n_laps}")

    if "lap_time" in df.columns:
        lap_times = df.groupby("lapIndex")["lap_time"].max()
        print(f"Lap times (s): {lap_times.values}")
        print(f"Best lap: {lap_times.min():.3f}s")

    print("\n✓ File looks good! Ready for preprocessing.")


if __name__ == "__main__":
    main()
