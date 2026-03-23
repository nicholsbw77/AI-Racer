"""
preprocess.py

Converts raw iRacing .ibt telemetry files to processed, normalized datasets.
Run this once before training.

Usage:
  python preprocess.py --input data/ --output data/processed/
  python preprocess.py --input data/ --output data/processed/ --build-map

Scans the input folder for .ibt files, groups them by car+track combo
(parsed from the iRacing filename convention), normalizes and engineers
features, then writes one data.parquet + meta.yaml per combo.

Enhanced with:
  - Automatic TrackMap building during preprocessing
  - Track map saved alongside processed data for live inference

iRacing .ibt filename convention:
  {car}_{track} {YYYY-MM-DD HH-MM-SS} ({utc_timestamp}).ibt
  e.g. cadillacctsvr_lagunaseca 2023-08-16 00-09-26 (2023_08_18 13_28_45 UTC).ibt

Outputs:
  data/processed/
    {car}_{track}/
      data.parquet   (cleaned, normalized, feature-engineered)
      meta.yaml      (dataset stats)
      track_map.yaml (segment profiles for GPS-free positioning)
"""

import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from loader import (
    normalize_features,
    engineer_features,
    filter_clean_laps,
    compute_lap_times,
    STATE_FEATURES,
    ACTION_FEATURES,
)
from ibt_loader import load_ibt_file, parse_combo_from_filename
from track_map import TrackMap

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


def group_ibt_files(input_root: Path) -> dict:
    """
    Scan input_root for .ibt files and group them by {car}_{track} combo.
    Returns dict: combo_name -> list of Path objects.
    """
    ibt_files = sorted(input_root.glob("**/*.ibt"))
    if not ibt_files:
        return {}

    groups = defaultdict(list)
    for fpath in ibt_files:
        car_id, track_id = parse_combo_from_filename(str(fpath))
        # Sanitize for use as a directory name (replace spaces with underscores)
        track_safe = track_id.replace(" ", "_")
        combo_name = f"{car_id}_{track_safe}"
        groups[combo_name].append(fpath)

    return dict(groups)


def preprocess_combo(
    combo_name: str,
    ibt_files: list,
    output_dir: Path,
    cfg: dict,
    build_map: bool = True,
) -> dict:
    """
    Process all .ibt files for one track/car combo.
    Returns stats dict, or empty dict on failure.
    """
    logger.info(f"\nProcessing {combo_name}: {len(ibt_files)} file(s)")

    all_dfs = []
    skipped = 0

    for i, fpath in enumerate(ibt_files):
        df = load_ibt_file(str(fpath))
        if df is None:
            skipped += 1
            continue

        df = normalize_features(df, cfg)
        df = engineer_features(df)

        # Re-index lapIndex globally across files to avoid collisions
        if "lapIndex" in df.columns:
            df["lapIndex"] = df["lapIndex"].astype(int) + i * 10000

        all_dfs.append(df)

    if not all_dfs:
        logger.warning(f"No valid data for {combo_name}")
        return {}

    combined = pd.concat(all_dfs, ignore_index=True)

    # Lap time stats before filtering
    lap_times = compute_lap_times(combined)
    personal_best = float(lap_times.min()) if len(lap_times) > 0 else None
    total_laps = len(lap_times)

    # Filter to clean laps only
    threshold = cfg["training"]["clean_lap_threshold"]
    combined = filter_clean_laps(combined, threshold)
    clean_laps = len(compute_lap_times(combined))

    if len(combined) == 0:
        logger.warning(f"No clean laps in {combo_name}")
        return {}

    # Build track map BEFORE speed normalization (needs raw speeds)
    track_map_path = None
    if build_map:
        track_cfg = cfg.get("track", {})
        num_segments = track_cfg.get("num_segments", 100)
        track_map = TrackMap(num_segments=num_segments)
        track_map.build_from_dataframe(combined, combo_name)
        if personal_best:
            track_map.personal_best_s = personal_best

        output_dir.mkdir(parents=True, exist_ok=True)
        track_map_path = str(output_dir / "track_map.yaml")
        track_map.save(track_map_path)

        # Also save to checkpoints dir for live inference
        ckpt_map_dir = Path(cfg["paths"]["checkpoints"]) / combo_name
        ckpt_map_dir.mkdir(parents=True, exist_ok=True)
        track_map.save(str(ckpt_map_dir / "track_map.yaml"))

    # Normalize speed globally for this combo
    speed_max = float(combined["speed"].max())
    if speed_max > 0:
        combined["speed"] = combined["speed"] / speed_max
        combined["speed_delta"] = combined["speed_delta"] / speed_max

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.parquet"
    combined.to_parquet(output_path, index=False)

    meta = {
        "combo_name": combo_name,
        "source_files": [str(f) for f in ibt_files],
        "total_files": len(ibt_files),
        "skipped_files": skipped,
        "total_laps": int(total_laps),
        "clean_laps": int(clean_laps),
        "total_frames": len(combined),
        "personal_best_s": personal_best,
        "speed_max_ms": speed_max,
        "data_hz": cfg["training"].get("data_hz", 60),
        "sequence_history": cfg["training"]["sequence_history"],
        "features": STATE_FEATURES,
        "actions": ACTION_FEATURES,
        "track_map": track_map_path,
    }

    with open(output_dir / "meta.yaml", "w") as f:
        yaml.dump(meta, f, default_flow_style=False)

    logger.info(
        f"  -> {clean_laps}/{total_laps} clean laps, "
        f"{len(combined):,} frames -> {output_path}"
    )
    if personal_best:
        mins = int(personal_best // 60)
        secs = personal_best % 60
        logger.info(f"  Personal best: {mins}:{secs:06.3f}")
    if track_map_path:
        logger.info(f"  Track map: {track_map_path}")

    return meta


def main():
    parser = argparse.ArgumentParser(description="Preprocess iRacing .ibt telemetry files")
    parser.add_argument("--input", default="data/",
                        help="Folder containing .ibt files (searched recursively)")
    parser.add_argument("--output", default="data/processed/",
                        help="Output folder for processed datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--combo", default=None,
                        help="Only process this combo name (e.g. cadillacctsvr_lagunaseca)")
    parser.add_argument("--no-map", action="store_true",
                        help="Skip building track maps")
    args = parser.parse_args()

    cfg = load_config(args.config)

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        logger.error(f"Input folder not found: {input_root}")
        sys.exit(1)

    groups = group_ibt_files(input_root)
    if not groups:
        logger.error(f"No .ibt files found in {input_root}")
        sys.exit(1)

    logger.info(f"Found {sum(len(v) for v in groups.values())} .ibt files "
                f"across {len(groups)} track/car combo(s):")
    for combo, files in sorted(groups.items()):
        logger.info(f"  {combo}: {len(files)} file(s)")

    # Filter to specific combo if requested
    if args.combo:
        groups = {k: v for k, v in groups.items() if args.combo.lower() in k.lower()}
        if not groups:
            logger.error(f"No combo matching '{args.combo}' found")
            sys.exit(1)

    build_map = cfg.get("track", {}).get("auto_build_map", True) and not args.no_map

    all_meta = {}
    for combo_name, ibt_files in sorted(groups.items()):
        out_dir = output_root / combo_name
        meta = preprocess_combo(combo_name, ibt_files, out_dir, cfg, build_map=build_map)
        if meta:
            all_meta[combo_name] = meta

    logger.info("\n" + "=" * 50)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 50)
    total_frames = sum(m.get("total_frames", 0) for m in all_meta.values())
    total_laps = sum(m.get("clean_laps", 0) for m in all_meta.values())
    logger.info(f"Combos processed  : {len(all_meta)}")
    logger.info(f"Total clean laps  : {total_laps:,}")
    logger.info(f"Total frames      : {total_frames:,}")
    logger.info(f"Track maps built  : {sum(1 for m in all_meta.values() if m.get('track_map'))}")
    logger.info(f"Output            : {output_root}")


if __name__ == "__main__":
    main()
