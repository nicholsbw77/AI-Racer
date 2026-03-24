"""
trainer/loader.py

Loads VRS/SRT tab-separated CSV exports and engineers the feature vectors
used for behavior cloning training.

Enhanced with:
  - Extended state features (yaw_rate, slip_angle)
  - Track-aware feature columns for segment context
  - TrackMap integration for building track maps during preprocessing

VRS CSV format notes (from SRT docs):
  - Tab-separated (not comma)
  - SI units: speed in m/s, distance in meters, angles in radians
  - 'validBin' = 1 for usable rows
  - 'lapFlag' = 0 for normal laps (ignore lapFlag=1 race-start laps)
  - 'lap_distance' = meters from start/finish line
  - Binned by distance (not time) - consistent bin positions across laps
  - carId, trackId columns identify the combo
"""

import os
import glob
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column name candidates - VRS exports vary slightly by version/car
# We try each in order and use the first one found.
# ---------------------------------------------------------------------------
COLUMN_CANDIDATES = {
    "speed":      ["speed", "Speed", "VX", "velocity_x", "v"],
    "throttle":   ["throttle", "Throttle", "throttleInput", "gas"],
    "brake":      ["brake", "Brake", "brakeInput", "brakeRaw"],
    "steering":   ["steeringWheelAngle", "steering", "SteeringWheelAngle", "steer"],
    "gear":       ["gear", "Gear", "currentGear"],
    "rpm":        ["rpm", "RPM", "engineRPM"],
    "lat_g":      ["accelerationY", "lateralAcceleration", "latG", "gLat", "LatAccel"],
    "lon_g":      ["accelerationX", "longitudinalAcceleration", "lonG", "gLon", "LongAccel"],
    "track_pos":  ["trackPosition", "track_position", "trackLat", "lanePosition"],
    "lap_dist":   ["lap_distance", "lapDistance", "distanceOnTrack"],
    "lap_time":   ["lap_time", "lapTime", "currentLapTime"],
    "yaw_rate":   ["YawRate", "yaw_rate", "yawRate"],
    "velocity_y": ["VelocityY", "velocity_y", "velY", "lateralVelocity"],
}

SYSTEM_COLS = ["validBin", "lapFlag", "lapIndex", "lapNum", "trackId", "carId",
               "trackLength", "binIndex"]


def _resolve_column(df: pd.DataFrame, key: str) -> Optional[str]:
    """Return the first matching column name from candidates for this key."""
    for candidate in COLUMN_CANDIDATES.get(key, []):
        if candidate in df.columns:
            return candidate
    return None


def load_vrs_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a single VRS/SRT CSV export.
    Returns a cleaned DataFrame with normalized column names,
    or None if the file can't be parsed.
    """
    try:
        # VRS uses tab separator, UTF-8 encoding
        df = pd.read_csv(filepath, sep="\t", encoding="utf-8", low_memory=False)
    except Exception:
        # Fallback: try comma-separated (some VRS versions)
        try:
            df = pd.read_csv(filepath, sep=",", encoding="utf-8", low_memory=False)
        except Exception as e:
            logger.warning(f"Could not parse {filepath}: {e}")
            return None

    # Filter to valid bins only
    if "validBin" in df.columns:
        df = df[df["validBin"] == 1].copy()

    # Filter out race-start laps (lapFlag=1) - they have inconsistent distance
    if "lapFlag" in df.columns:
        df = df[df["lapFlag"] == 0].copy()

    if len(df) == 0:
        logger.warning(f"No valid bins in {filepath}")
        return None

    # Resolve and rename columns to canonical names
    col_map = {}
    missing = []
    for key in COLUMN_CANDIDATES:
        resolved = _resolve_column(df, key)
        if resolved:
            col_map[resolved] = key
        else:
            missing.append(key)

    if missing:
        logger.debug(f"{filepath}: missing columns {missing}")

    # Require at minimum throttle, brake, steering, speed, lap_dist
    required = {"throttle", "brake", "steering", "speed", "lap_dist"}
    resolved_keys = set(col_map.values())
    if not required.issubset(resolved_keys):
        logger.warning(f"Skipping {filepath}: missing required columns "
                       f"{required - resolved_keys}")
        return None

    df = df.rename(columns=col_map)

    # Preserve system columns for metadata
    keep_cols = list(col_map.values()) + [c for c in SYSTEM_COLS if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    return df


def normalize_features(df: pd.DataFrame, cfg: dict):
    """
    Normalize raw telemetry values to [-1, 1] or [0, 1] ranges.
    Modifies df in-place and returns (df, norm_consts) where norm_consts
    is a dict of detected normalization constants (steering_lock_radians, rpm_max).
    """
    feat_cfg = cfg.get("features", {})
    norm_consts = {}

    # Speed: normalize to [0, 1] using track max (computed per-session later)
    # Here we just ensure it's positive
    df["speed"] = df["speed"].clip(lower=0.0)

    # Gear: normalize to [0, 1] assuming max 7 gears (common in iRacing)
    df["gear"] = df["gear"].clip(0, 7) / 7.0

    # RPM: normalize 0-1 using observed max
    if "rpm" in df.columns:
        rpm_max = float(df["rpm"].max())
        norm_consts["rpm_max"] = rpm_max
        if rpm_max > 0:
            df["rpm"] = df["rpm"] / rpm_max
        else:
            df["rpm"] = 0.0

    # Throttle/brake: already 0-1 in VRS, but clamp just in case
    df["throttle"] = df["throttle"].clip(0.0, 1.0)
    df["brake"] = df["brake"].clip(0.0, 1.0)

    # Steering: normalize radians to [-1, 1]
    steering_lock = feat_cfg.get("steering_lock_radians")
    if steering_lock is None:
        # Auto-detect from data: use 99th percentile absolute value
        steering_lock = float(df["steering"].abs().quantile(0.99))
        if steering_lock < 0.1:
            steering_lock = np.pi  # fallback ~180 degrees
        logger.debug(f"Auto-detected steering lock: {np.degrees(steering_lock):.1f} deg")
    norm_consts["steering_lock_radians"] = float(steering_lock)

    df["steering"] = (df["steering"] / steering_lock).clip(-1.0, 1.0)

    # G-forces: normalize using typical racing limits (+/-4g)
    for col in ["lat_g", "lon_g"]:
        if col in df.columns:
            df[col] = (df[col] / 40.0).clip(-1.0, 1.0)  # 40 m/s^2 ~ 4g

    # Track position: should already be -1 to +1, clamp it
    if "track_pos" in df.columns:
        df["track_pos"] = df["track_pos"].clip(-1.0, 1.0)
    else:
        df["track_pos"] = 0.0

    # Yaw rate: normalize to [-1, 1] using +/-3 rad/s range
    if "yaw_rate" in df.columns:
        df["yaw_rate"] = (df["yaw_rate"] / 3.0).clip(-1.0, 1.0)
    else:
        df["yaw_rate"] = 0.0

    # Lateral velocity -> slip angle proxy
    if "velocity_y" in df.columns and "speed" in df.columns:
        safe_speed = df["speed"].clip(lower=1.0)
        df["slip_angle"] = np.arctan2(df["velocity_y"], safe_speed)
        df["slip_angle"] = (df["slip_angle"] / 0.5).clip(-1.0, 1.0)
    else:
        df["slip_angle"] = 0.0

    # Lap distance pct: compute from lap_distance and trackLength
    if "trackLength" in df.columns and "lap_dist" in df.columns:
        track_len = df["trackLength"].iloc[0]
        if track_len > 0:
            df["lap_dist_pct"] = (df["lap_dist"] / track_len).clip(0.0, 1.0)
        else:
            df["lap_dist_pct"] = 0.0
    elif "lap_dist_pct" not in df.columns and "lap_dist" in df.columns:
        # Normalize by observed max distance
        dist_max = df["lap_dist"].max()
        df["lap_dist_pct"] = (df["lap_dist"] / dist_max).clip(0.0, 1.0) if dist_max > 0 else 0.0

    return df, norm_consts


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that help the model predict inputs more accurately.
    """
    # Speed delta (acceleration proxy) - rate of change
    df["speed_delta"] = df["speed"].diff().fillna(0.0).clip(-0.5, 0.5)

    # Braking zone flag: are we decelerating hard?
    df["heavy_braking"] = ((df["brake"] > 0.3) & (df["speed_delta"] < -0.05)).astype(float)

    # Throttle application flag
    df["full_throttle"] = (df["throttle"] > 0.95).astype(float)

    # Steering magnitude (helps model understand cornering intensity)
    df["steering_abs"] = df["steering"].abs()

    # Steering rate of change (helps smoothness prediction)
    df["steering_delta"] = df["steering"].diff().fillna(0.0).clip(-0.3, 0.3)

    return df


def compute_lap_times(df: pd.DataFrame) -> pd.Series:
    """
    Return a Series indexed by lapIndex with lap time in seconds.
    """
    if "lap_time" not in df.columns or "lapIndex" not in df.columns:
        return pd.Series(dtype=float)

    # Lap time at the last valid bin of each lap
    return df.groupby("lapIndex")["lap_time"].max()


def filter_clean_laps(df: pd.DataFrame, threshold: float = 1.01) -> pd.DataFrame:
    """
    Keep only laps within `threshold` * personal best lap time.
    This ensures we only train on quality representative laps.
    """
    if "lapIndex" not in df.columns:
        return df

    lap_times = compute_lap_times(df)
    if len(lap_times) == 0:
        return df

    personal_best = lap_times.min()
    cutoff = personal_best * threshold

    valid_laps = lap_times[lap_times <= cutoff].index
    filtered = df[df["lapIndex"].isin(valid_laps)].copy()

    n_total = lap_times.shape[0]
    n_kept = len(valid_laps)
    logger.info(f"Lap filter: kept {n_kept}/{n_total} laps "
                f"(PB={personal_best:.3f}s, cutoff={cutoff:.3f}s)")

    return filtered


def load_track_car_dataset(
    folder: str,
    cfg: dict,
) -> Optional[pd.DataFrame]:
    """
    Load all CSV files from a track/car folder, clean, normalize, and combine.

    Expected folder structure:
      data/raw/<trackId>_<carId>/
        lap_001.csv
        lap_002.csv
        ...

    Returns a single DataFrame ready for Dataset construction.
    """
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {folder}")
        return None

    dfs = []
    for i, fpath in enumerate(sorted(csv_files)):
        df = load_vrs_csv(fpath)
        if df is None:
            continue
        df, _ = normalize_features(df, cfg)
        df = engineer_features(df)

        # Re-index lapIndex globally across files
        if "lapIndex" in df.columns:
            df["lapIndex"] = df["lapIndex"] + i * 10000

        dfs.append(df)

    if not dfs:
        logger.warning(f"No valid data loaded from {folder}")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Filter to clean laps
    threshold = cfg.get("training", {}).get("clean_lap_threshold", 1.01)
    combined = filter_clean_laps(combined, threshold)

    if len(combined) == 0:
        logger.warning(f"No clean laps found in {folder}")
        return None

    # Normalize speed to [0,1] using global max for this track/car combo
    speed_max = combined["speed"].max()
    if speed_max > 0:
        combined["speed"] = combined["speed"] / speed_max
        combined["speed_delta"] = combined["speed_delta"] / speed_max

    logger.info(f"Loaded {len(combined)} bins from {len(dfs)} files in {folder}")
    return combined


# ---------------------------------------------------------------------------
# Feature and action column definitions
# ---------------------------------------------------------------------------

STATE_FEATURES = [
    "lap_dist_pct",     # where on track (0-1)
    "speed",            # current speed (0-1 normalized)
    "speed_delta",      # acceleration proxy
    "gear",             # gear (0-1 normalized)
    "rpm",              # engine RPM (0-1)
    "lat_g",            # lateral G-force
    "lon_g",            # longitudinal G-force
    "track_pos",        # lateral offset from centerline
    "steering_abs",     # steering magnitude (cornering context)
    "heavy_braking",    # braking zone flag
    "full_throttle",    # full throttle flag
    "yaw_rate",         # yaw rotation rate (NEW)
    "slip_angle",       # estimated slip angle (NEW)
]

ACTION_FEATURES = [
    "throttle",
    "brake",
    "steering",
]

# These action features are appended to state as history context
HISTORY_ACTIONS = ["throttle", "brake", "steering", "steering_delta"]
