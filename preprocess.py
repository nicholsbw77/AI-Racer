"""
scripts/preprocess.py

Converts raw VRS CSV exports to processed, normalized datasets.
Run this once before training.

Usage:
  python scripts/preprocess.py --input data/raw/ --output data/processed/

Expects input structure:
  data/raw/
    <trackId>_<carId>/
      session_001.csv
      session_002.csv
      ...

Outputs:
  data/processed/
    <trackId>_<carId>/
      *.csv   (cleaned, normalized, engineered)
      meta.yaml  (dataset stats for this combo)
"""

import sys
import os
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.loader import (
    load_vrs_csv,
    normalize_features,
    engineer_features,
    filter_clean_laps,
    compute_lap_times,
    STATE_FEATURES,
    ACTION_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _resolve_sequence_history(cfg)
    return cfg


def _resolve_sequence_history(cfg: dict):
    train_cfg = cfg["training"]
    if train_cfg.get("sequence_history") == "auto":
        hz = int(train_cfg.get("data_hz", 60))
        ms = float(train_cfg.get("context_window_ms", 250))
        train_cfg["sequence_history"] = max(1, round(ms / 1000.0 * hz))
        logger.info(
            f"sequence_history=auto resolved to {train_cfg['sequence_history']} "
            f"frames ({ms:.0f}ms @ {hz}Hz)"
        )


def preprocess_combo(input_dir: Path, output_dir: Path, cfg: dict) -> dict:
    """Process one track/car folder. Returns stats dict."""
    import glob

    csv_files = sorted(glob.glob(str(input_dir / "*.csv")))
    if not csv_files:
        logger.warning(f"No CSV files in {input_dir}")
        return {}

    logger.info(f"Processing {input_dir.name}: {len(csv_files)} files")

    all_dfs = []
    skipped = 0

    for i, fpath in enumerate(csv_files):
        df = load_vrs_csv(fpath)
        if df is None:
            skipped += 1
            continue

        df = normalize_features(df, cfg)
        df = engineer_features(df)

        # Re-index laps globally
        if "lapIndex" in df.columns:
            df["lapIndex"] = df["lapIndex"] + i * 10000

        all_dfs.append(df)

    if not all_dfs:
        logger.warning(f"No valid data in {input_dir}")
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute and save lap times before filtering
    lap_times = compute_lap_times(combined)
    personal_best = lap_times.min() if len(lap_times) > 0 else None
    total_laps = len(lap_times)

    # Filter to clean laps
    threshold = cfg["training"]["clean_lap_threshold"]
    combined = filter_clean_laps(combined, threshold)
    clean_laps = len(compute_lap_times(combined))

    # Normalize speed globally for this combo
    speed_max = combined["speed"].max()
    if speed_max > 0:
        combined["speed"] = combined["speed"] / speed_max
        combined["speed_delta"] = combined["speed_delta"] / speed_max

    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.parquet"
    combined.to_parquet(output_path, index=False)

    # Save metadata
    meta = {
        "combo_name": input_dir.name,
        "total_files": len(csv_files),
        "skipped_files": skipped,
        "total_laps": int(total_laps),
        "clean_laps": int(clean_laps),
        "total_bins": len(combined),
        "personal_best_s": float(personal_best) if personal_best else None,
        "speed_max_ms": float(speed_max),
        "features": STATE_FEATURES,
        "actions": ACTION_FEATURES,
    }

    with open(output_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    logger.info(
        f"  ✓ {clean_laps}/{total_laps} clean laps, "
        f"{len(combined):,} bins → {output_path}"
    )
    if personal_best:
        mins = int(personal_best // 60)
        secs = personal_best % 60
        logger.info(f"  Personal best: {mins}:{secs:06.3f}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess VRS telemetry CSVs")
    parser.add_argument("--input", default="data/raw/",
                        help="Input folder with track/car subfolders")
    parser.add_argument("--output", default="data/processed/",
                        help="Output folder for processed datasets")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        logger.error(f"Input folder not found: {input_root}")
        sys.exit(1)

    combos = [d for d in input_root.iterdir() if d.is_dir()]
    if not combos:
        logger.error(f"No track/car subfolders found in {input_root}")
        logger.error("Expected: data/raw/<trackId>_<carId>/lap_001.csv ...")
        sys.exit(1)

    logger.info(f"Found {len(combos)} track/car combo(s)")

    all_meta = {}
    for combo_dir in sorted(combos):
        out_dir = output_root / combo_dir.name
        meta = preprocess_combo(combo_dir, out_dir, cfg)
        if meta:
            all_meta[combo_dir.name] = meta

    # Summary
    logger.info("\n" + "="*50)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*50)
    total_bins = sum(m.get("total_bins", 0) for m in all_meta.values())
    total_laps = sum(m.get("clean_laps", 0) for m in all_meta.values())
    logger.info(f"Combos processed: {len(all_meta)}")
    logger.info(f"Total clean laps: {total_laps:,}")
    logger.info(f"Total training bins: {total_bins:,}")
    logger.info(f"Output: {output_root}")


if __name__ == "__main__":
    main()
