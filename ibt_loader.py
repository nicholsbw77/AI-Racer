"""
ibt_loader.py

Reads iRacing .ibt binary telemetry files into DataFrames compatible
with the rest of the training pipeline (same canonical column names
as loader.py).

iRacing .ibt channel names used:
  Speed             - m/s (vehicle speed)
  Throttle          - 0-1
  Brake             - 0-1
  SteeringWheelAngle - radians
  Gear              - integer (-1=reverse, 0=neutral, 1-n=forward)
  RPM               - engine RPM
  LatAccel          - m/s² lateral acceleration
  LongAccel         - m/s² longitudinal acceleration
  LapDistPct        - 0-1 normalized distance around track
  Lap               - lap counter (increments at start/finish)
  SessionTime       - seconds since session start

Usage:
  from ibt_loader import load_ibt_file, parse_combo_from_filename
  df = load_ibt_file("data/cadillacctsvr_lagunaseca 2023-08-16 00-09-26.ibt")
"""

import re
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import irsdk
    IRSDK_AVAILABLE = True
except ImportError:
    IRSDK_AVAILABLE = False
    logger.warning("pyirsdk not available - cannot read .ibt files")


# ---------------------------------------------------------------------------
# iRacing channel name → our canonical name
# ---------------------------------------------------------------------------
IBT_CHANNEL_MAP = {
    "Speed":              "speed",            # m/s
    "Throttle":           "throttle",         # 0-1
    "Brake":              "brake",            # 0-1
    "SteeringWheelAngle": "steering",         # radians
    "Gear":               "gear",             # integer
    "RPM":                "rpm",
    "LatAccel":           "lat_g",            # m/s²
    "LongAccel":          "lon_g",            # m/s²
    "LapDistPct":         "lap_dist_pct",     # 0-1
    "Lap":                "lapIndex",         # lap counter (increments at S/F)
    "SessionTime":        "session_time",     # seconds
    # Lap timing — LapCurrentLapTime resets to 0 at S/F; max within a
    # lapIndex group = completed lap time.  Required by compute_lap_times().
    "LapCurrentLapTime":  "lap_time",         # seconds, current lap elapsed
    "LapLastLapTime":     "lap_last_time",    # seconds, previous lap completed
    "LapBestLapTime":     "lap_best_time",    # seconds, session best so far
}

REQUIRED_CHANNELS = {"Speed", "Throttle", "Brake", "SteeringWheelAngle", "LapDistPct"}


def _get_available_channels(ibt) -> set:
    """Get channel names from an open IBT object, supporting multiple pyirsdk versions."""
    if hasattr(ibt, "var_headers_names") and ibt.var_headers_names:
        return set(ibt.var_headers_names)
    if hasattr(ibt, "var_headers_dict") and ibt.var_headers_dict:
        return set(ibt.var_headers_dict.keys())
    return set()


def load_ibt_file(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load a single .ibt file into a cleaned DataFrame.

    Returns a DataFrame with canonical column names ready for
    normalize_features() and engineer_features() in loader.py,
    or None if the file cannot be read.
    """
    if not IRSDK_AVAILABLE:
        logger.error("pyirsdk is required to read .ibt files. Install it: pip install pyirsdk")
        return None

    path = Path(filepath)
    if not path.exists():
        logger.warning(f"File not found: {filepath}")
        return None

    try:
        ibt = irsdk.IBT()
        ibt.open(str(path))
    except Exception as e:
        logger.warning(f"Failed to open {filepath}: {e}")
        return None

    # Check required channels are present
    available = _get_available_channels(ibt)
    missing = REQUIRED_CHANNELS - available
    if missing:
        logger.warning(f"Skipping {path.name}: missing channels {missing}")
        ibt.close()
        return None

    # Read all channels we care about
    data = {}
    for iracing_name, canonical_name in IBT_CHANNEL_MAP.items():
        try:
            values = ibt.get_all(iracing_name)
            if values is not None:
                # Some iRacing channels return per-wheel/per-tire data as
                # lists of tuples.  np.array(..., dtype=float32) may fail
                # or produce an object array.  Handle both cases.
                try:
                    arr = np.array(values, dtype=np.float32)
                except (ValueError, TypeError):
                    # Ragged or nested — try extracting first element per tick
                    arr = np.array([v[0] if hasattr(v, '__len__') else v
                                    for v in values], dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr[:, 0] if arr.shape[1] > 0 else arr.flatten()
                elif arr.ndim == 1 and arr.dtype == object:
                    # Object array of lists/tuples — take first element
                    arr = np.array([v[0] if hasattr(v, '__len__') else v
                                    for v in arr], dtype=np.float32)
                data[canonical_name] = arr
        except Exception:
            logger.debug(f"Could not read channel {iracing_name} from {path.name}")

    # Extract track length before closing the file
    track_length_m = _get_track_length(ibt) if hasattr(ibt, "session_info") else None

    ibt.close()

    if not data:
        logger.warning(f"No data extracted from {filepath}")
        return None

    # Build DataFrame — align all arrays to the shortest length
    min_len = min(len(v) for v in data.values())
    if min_len == 0:
        logger.warning(f"Empty channels in {filepath}")
        return None

    df = pd.DataFrame({k: v[:min_len] for k, v in data.items()})
    if track_length_m and track_length_m > 0 and "lap_dist_pct" in df.columns:
        df["lap_dist"] = df["lap_dist_pct"] * track_length_m
        df["trackLength"] = track_length_m
    elif "lap_dist_pct" in df.columns:
        # Fallback: use pct as proxy (normalize_features handles this)
        df["lap_dist"] = df["lap_dist_pct"]

    # Gear: clip to valid range (iRacing uses -1 for reverse)
    if "gear" in df.columns:
        df["gear"] = df["gear"].clip(-1, 8)

    # Remove rows where the car is clearly not on track:
    # LapDistPct stuck at 0.0 before session starts
    if "lap_dist_pct" in df.columns and "session_time" in df.columns:
        # Keep frames after first movement (speed > 0.5 m/s or LapDistPct > 0.01)
        if "speed" in df.columns:
            first_move = df[(df["speed"] > 0.5) | (df["lap_dist_pct"] > 0.01)].index
            if len(first_move) > 0:
                df = df.iloc[first_move[0]:].reset_index(drop=True)

    if len(df) == 0:
        logger.warning(f"No valid frames in {filepath}")
        return None

    logger.info(f"Loaded {len(df)} frames from {path.name}")
    return df


def _get_track_length(ibt) -> Optional[float]:
    """Extract track length in meters from iRacing session YAML."""
    try:
        info = ibt.session_info
        if info and "WeekendInfo" in info:
            length_str = info["WeekendInfo"].get("TrackLength", "")
            # Format: "4.01 km" or "2.49 mi"
            match = re.match(r"([\d.]+)\s*(km|mi)", str(length_str))
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                if unit == "km":
                    return value * 1000.0
                elif unit == "mi":
                    return value * 1609.344
    except Exception:
        pass
    return None


def get_session_info(filepath: str) -> dict:
    """
    Return session metadata from a .ibt file without loading all channel data.
    Useful for building combo names and checking content before full load.
    """
    if not IRSDK_AVAILABLE:
        return {}

    try:
        ibt = irsdk.IBT()
        ibt.open(str(filepath))
        info = {}

        if hasattr(ibt, "session_info") and ibt.session_info:
            si = ibt.session_info
            weekend = si.get("WeekendInfo", {})
            info["track_name"] = weekend.get("TrackDisplayName", "")
            info["track_id"]   = weekend.get("TrackName", "")
            info["track_length"] = weekend.get("TrackLength", "")

            # Car info lives under DriverInfo → Drivers → first entry
            drivers = si.get("DriverInfo", {}).get("Drivers", [])
            if drivers:
                d = drivers[0]
                info["car_id"]   = d.get("CarPath", "")
                info["car_name"] = d.get("CarScreenNameShort", "")

        info["channels"] = sorted(_get_available_channels(ibt))

        # Sample rate from first channel header
        if hasattr(ibt, "var_headers") and ibt.var_headers:
            # iRacing tick rate is in the disk header, not per-channel
            # Estimate from SessionTime delta
            try:
                times = ibt.get_all("SessionTime")
                if times is not None and len(times) > 1:
                    deltas = np.diff(np.array(times[:100]))
                    avg_dt = float(np.median(deltas[deltas > 0]))
                    info["sample_hz"] = round(1.0 / avg_dt) if avg_dt > 0 else None
            except Exception:
                pass

        ibt.close()
        return info

    except Exception as e:
        logger.warning(f"Could not read session info from {filepath}: {e}")
        return {}


def parse_combo_from_filename(filepath: str) -> Tuple[str, str]:
    """
    Parse car and track from iRacing .ibt filename convention.

    iRacing names files as:
      {car}_{track} {YYYY-MM-DD HH-MM-SS} ({utc_timestamp}).ibt

    Examples:
      cadillacctsvr_lagunaseca 2023-08-16 00-09-26 (...).ibt
        → car="cadillacctsvr", track="lagunaseca"
      mercedesamgevogt3_watkinsglen 2021 cupcircuit 2023-08-14 22-41-51 (...).ibt
        → car="mercedesamgevogt3", track="watkinsglen 2021 cupcircuit"

    Returns (car_id, track_id) strings, or ("unknown", "unknown") on failure.
    """
    name = Path(filepath).stem

    # Strip the UTC timestamp suffix: " (2023_08_15 06_35_07 UTC)"
    name = re.sub(r"\s*\([\d_]+ [\d_]+ UTC\)", "", name).strip()

    # Strip the local datetime: " 2023-08-14 22-41-51"
    name = re.sub(r"\s+\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}$", "", name).strip()

    # Split on first underscore: everything before = car, after = track
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()

    return "unknown", name


def load_ibt_files(folder: str, cfg: dict) -> Optional[pd.DataFrame]:
    """
    Load all .ibt files from a folder, normalize, engineer features,
    and return a combined DataFrame ready for TelemetryDataset.

    This is the .ibt equivalent of load_track_car_dataset() in loader.py.
    """
    from loader import normalize_features, engineer_features, filter_clean_laps

    ibt_files = sorted(Path(folder).glob("*.ibt"))
    if not ibt_files:
        logger.warning(f"No .ibt files found in {folder}")
        return None

    dfs = []
    for i, fpath in enumerate(ibt_files):
        df = load_ibt_file(str(fpath))
        if df is None:
            continue

        df, _ = normalize_features(df, cfg)
        df = engineer_features(df)

        # Re-index lapIndex globally across files
        if "lapIndex" in df.columns:
            df["lapIndex"] = df["lapIndex"].astype(int) + i * 10000

        dfs.append(df)

    if not dfs:
        logger.warning(f"No valid data loaded from {folder}")
        return None

    combined = pd.concat(dfs, ignore_index=True)

    threshold = cfg.get("training", {}).get("clean_lap_threshold", 1.01)
    combined = filter_clean_laps(combined, threshold)

    if len(combined) == 0:
        logger.warning(f"No clean laps found in {folder}")
        return None

    speed_max = combined["speed"].max()
    if speed_max > 0:
        combined["speed"] = combined["speed"] / speed_max
        combined["speed_delta"] = combined["speed_delta"] / speed_max

    logger.info(f"Loaded {len(combined)} bins from {len(dfs)} files in {folder}")
    return combined
