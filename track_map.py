"""
track_map.py

Builds a virtual track map from recorded telemetry data, enabling track
boundary awareness for the AI driver.

iRacing does not directly expose lateral track position in .ibt files, so
this module reconstructs a centerline profile from available telemetry
channels (speed, steering, lat_g, lon_g, lap_dist_pct) and derives:
  - Speed profile with safe min/max envelope
  - Curvature profile from lateral G / speed^2
  - Corner detection (apex, direction, entry speed)
  - Braking zone detection (from sustained heavy braking)
  - Lateral position estimation from dynamic state comparison

Usage:
  # Build from processed data
  from track_map import TrackMap
  track_map = TrackMap.build_from_dataframe(df)
  track_map.save("checkpoints/cadillacctsvr_lagunaseca/track_map.json")

  # Load and query
  track_map = TrackMap.load("checkpoints/.../track_map.json")
  corners = track_map.get_corners()
  braking_zones = track_map.get_braking_zones()

Standalone:
  python track_map.py --data data/processed/some_combo/data.parquet
"""

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Corner:
    """Detected corner on the track."""
    start_pct: float       # lap_dist_pct where corner entry begins
    apex_pct: float        # lap_dist_pct of maximum curvature
    end_pct: float         # lap_dist_pct where corner exit ends
    max_curvature: float   # peak curvature value at apex
    direction: str         # "left" or "right"
    suggested_entry_speed: float  # observed average speed at corner entry


@dataclass
class BrakingZone:
    """Detected braking zone on the track."""
    start_pct: float       # lap_dist_pct where braking begins
    end_pct: float         # lap_dist_pct where braking ends
    entry_speed: float     # average speed at braking zone start
    exit_speed: float      # average speed at braking zone end
    brake_intensity: float # average brake pressure in zone (0-1)


@dataclass
class BinData:
    """Aggregated telemetry statistics for one track segment."""
    lap_dist_pct: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    typical_speed: float = 0.0
    speed_std: float = 0.0
    speed_min: float = 0.0
    speed_max: float = 0.0
    typical_steering: float = 0.0
    steering_std: float = 0.0
    typical_throttle: float = 0.0
    typical_brake: float = 0.0
    typical_lat_g: float = 0.0
    lat_g_std: float = 0.0
    typical_lon_g: float = 0.0
    track_width_estimate: float = 0.0


# ---------------------------------------------------------------------------
# TrackMap
# ---------------------------------------------------------------------------

class TrackMap:
    """
    Represents a learned track layout built from recorded telemetry.

    The track is divided into N bins by lap_dist_pct. Each bin stores
    aggregated statistics (speed, curvature, steering, etc.) computed
    from all observed laps in the training data.
    """

    def __init__(self, n_bins: int = 200):
        self.n_bins: int = n_bins
        self.bins: List[BinData] = [BinData() for _ in range(n_bins)]
        self._corners: Optional[List[Corner]] = None
        self._braking_zones: Optional[List[BrakingZone]] = None

    # ------------------------------------------------------------------
    # Build from telemetry
    # ------------------------------------------------------------------

    @classmethod
    def build_from_dataframe(
        cls,
        df,
        n_bins: int = 200,
    ) -> "TrackMap":
        """
        Build a TrackMap from a processed telemetry DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns: lap_dist_pct, speed, steering, lat_g, lon_g,
            throttle, brake.  Additional columns (gear, rpm, etc.) are ignored.
        n_bins : int
            Number of equal-width segments to divide the track into.
            Default 200 gives ~0.5 % resolution.

        Returns
        -------
        TrackMap
            Populated track map ready for queries.
        """
        import pandas as pd  # deferred so numpy-only callers don't need pandas

        required = {"lap_dist_pct", "speed", "steering", "lat_g", "lon_g",
                     "throttle", "brake"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        tm = cls(n_bins=n_bins)

        # Bin edges from 0 to 1
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Assign each row to a bin
        pct = df["lap_dist_pct"].values
        bin_idx = np.clip(
            np.digitize(pct, bin_edges[1:]),  # 0-based bin index
            0,
            n_bins - 1,
        )

        speed = df["speed"].values.astype(np.float64)
        steering = df["steering"].values.astype(np.float64)
        lat_g = df["lat_g"].values.astype(np.float64)
        lon_g = df["lon_g"].values.astype(np.float64)
        throttle = df["throttle"].values.astype(np.float64)
        brake = df["brake"].values.astype(np.float64)

        eps = 1e-6  # avoid division by zero

        for i in range(n_bins):
            mask = bin_idx == i
            if not np.any(mask):
                # Empty bin — interpolate later
                tm.bins[i].lap_dist_pct = float(bin_centers[i])
                continue

            s = speed[mask]
            st = steering[mask]
            lg = lat_g[mask]
            lng = lon_g[mask]
            thr = throttle[mask]
            brk = brake[mask]

            bd = tm.bins[i]
            bd.lap_dist_pct = float(bin_centers[i])

            # Speed statistics
            bd.typical_speed = float(np.mean(s))
            bd.speed_std = float(np.std(s))
            bd.speed_min = float(np.min(s))
            bd.speed_max = float(np.max(s))

            # Steering
            bd.typical_steering = float(np.mean(st))
            bd.steering_std = float(np.std(st))

            # Lateral / longitudinal G
            bd.typical_lat_g = float(np.mean(lg))
            bd.lat_g_std = float(np.std(lg))
            bd.typical_lon_g = float(np.mean(lng))

            # Curvature: lat_g / speed^2  (positive = left, negative = right)
            speed_sq = s ** 2 + eps
            curvatures = lg / speed_sq
            bd.curvature = float(np.mean(curvatures))

            # Heading: integrate curvature as a rough proxy.
            # We reconstruct relative heading changes from lat_g / speed.
            # heading_rate = lat_g / speed  (rad/s, but we bin by distance)
            heading_rates = lg / (s + eps)
            bd.heading = float(np.mean(heading_rates))

            # Controls
            bd.typical_throttle = float(np.mean(thr))
            bd.typical_brake = float(np.mean(brk))

            # Track width estimate: higher variance in lateral G indicates a
            # wider section where drivers take different lines.
            bd.track_width_estimate = float(np.std(lg)) + float(np.std(st)) * 0.5

        # Fill any empty bins by linear interpolation from neighbors
        tm._interpolate_empty_bins()

        # Pre-compute derived structures
        tm._corners = None
        tm._braking_zones = None
        tm._compute_expected_headings()

        logger.info(
            f"TrackMap built: {n_bins} bins, "
            f"{len(tm.get_corners())} corners, "
            f"{len(tm.get_braking_zones())} braking zones"
        )
        return tm

    def _interpolate_empty_bins(self) -> None:
        """Fill bins that had no data via linear interpolation."""
        fields = [
            "heading", "curvature", "typical_speed", "speed_std",
            "speed_min", "speed_max", "typical_steering", "steering_std",
            "typical_throttle", "typical_brake", "typical_lat_g",
            "lat_g_std", "typical_lon_g", "track_width_estimate",
        ]
        n = self.n_bins
        for fname in fields:
            vals = np.array([getattr(self.bins[i], fname) for i in range(n)])
            populated = np.array([
                i for i in range(n) if vals[i] != 0.0 or fname == "curvature"
            ])
            if len(populated) < 2:
                continue

            # Only interpolate truly empty bins (speed == 0 means no data)
            empty_mask = np.array([
                self.bins[i].typical_speed == 0.0 and self.bins[i].lap_dist_pct != 0.0
                for i in range(n)
            ])
            if not np.any(empty_mask):
                continue

            populated_vals = vals[populated]
            populated_pcts = np.array([
                self.bins[i].lap_dist_pct for i in populated
            ])
            all_pcts = np.array([self.bins[i].lap_dist_pct for i in range(n)])

            interp_vals = np.interp(all_pcts, populated_pcts, populated_vals)
            for i in range(n):
                if empty_mask[i]:
                    setattr(self.bins[i], fname, float(interp_vals[i]))

    def _compute_expected_headings(self) -> None:
        """Compute expected relative heading at each bin by integrating heading rates.

        The result is a relative heading profile — the absolute offset is unknown
        until calibrated against live Yaw data.  The profile lets us compute
        heading *error* (how far the car's yaw deviates from expected).
        """
        n = self.n_bins
        bin_width = 1.0 / n  # fraction of lap per bin

        # heading field stores heading_rate (rad/s-ish, actually rad per unit distance).
        # Integrate to get cumulative heading at each bin.
        heading_rates = np.array([self.bins[i].heading for i in range(n)])

        # Cumulative sum gives relative heading at each bin start
        self._expected_headings = np.zeros(n)
        cumulative = 0.0
        for i in range(n):
            self._expected_headings[i] = cumulative
            cumulative += heading_rates[i] * bin_width

        logger.debug(f"Expected heading profile computed ({n} bins, "
                     f"total rotation={cumulative:.2f} rad)")

    def get_expected_heading(self, lap_dist_pct: float) -> float:
        """Return the expected relative heading at a track position (radians)."""
        if not hasattr(self, '_expected_headings') or self._expected_headings is None:
            return 0.0
        idx = int(lap_dist_pct * self.n_bins)
        idx = max(0, min(idx, self.n_bins - 1))
        return float(self._expected_headings[idx])

    # ------------------------------------------------------------------
    # Speed profile
    # ------------------------------------------------------------------

    def get_speed_profile(self) -> np.ndarray:
        """
        Return the speed profile as an array of shape (n_bins, 4).

        Columns: [lap_dist_pct, safe_min_speed, typical_speed, safe_max_speed]

        safe_min = mean - 2*std (clipped to 0)
        safe_max = mean + 2*std
        """
        result = np.zeros((self.n_bins, 4), dtype=np.float64)
        for i, b in enumerate(self.bins):
            safe_min = max(0.0, b.typical_speed - 2.0 * b.speed_std)
            safe_max = b.typical_speed + 2.0 * b.speed_std
            result[i] = [b.lap_dist_pct, safe_min, b.typical_speed, safe_max]
        return result

    # ------------------------------------------------------------------
    # Curvature profile
    # ------------------------------------------------------------------

    def get_curvature_profile(self) -> np.ndarray:
        """
        Return the curvature profile as an array of shape (n_bins, 2).

        Columns: [lap_dist_pct, curvature]

        High absolute curvature = sharp corner = need to slow down.
        Positive curvature = left turn, negative = right turn.
        """
        result = np.zeros((self.n_bins, 2), dtype=np.float64)
        for i, b in enumerate(self.bins):
            result[i] = [b.lap_dist_pct, b.curvature]
        return result

    # ------------------------------------------------------------------
    # Corner detection
    # ------------------------------------------------------------------

    def get_corners(self, min_curvature: float = 0.005) -> List[Corner]:
        """
        Detect corners from the curvature profile.

        Parameters
        ----------
        min_curvature : float
            Minimum absolute curvature to qualify as a corner.

        Returns
        -------
        list of Corner
            Sorted by start_pct.
        """
        if self._corners is not None:
            return self._corners

        curvatures = np.array([b.curvature for b in self.bins])
        abs_curv = np.abs(curvatures)

        # Identify bins above the curvature threshold
        above = abs_curv >= min_curvature

        corners: List[Corner] = []
        i = 0
        n = self.n_bins
        while i < n:
            if not above[i]:
                i += 1
                continue

            # Found start of a corner region
            start = i
            while i < n and above[i]:
                i += 1
            end = i - 1  # inclusive

            # Find apex: bin with max absolute curvature in this region
            region_abs = abs_curv[start:end + 1]
            apex_offset = int(np.argmax(region_abs))
            apex = start + apex_offset

            max_curv = float(abs_curv[apex])
            direction = "left" if curvatures[apex] > 0 else "right"

            # Suggested entry speed: average observed speed at corner start
            suggested_entry_speed = float(self.bins[start].typical_speed)

            corners.append(Corner(
                start_pct=float(self.bins[start].lap_dist_pct),
                apex_pct=float(self.bins[apex].lap_dist_pct),
                end_pct=float(self.bins[end].lap_dist_pct),
                max_curvature=max_curv,
                direction=direction,
                suggested_entry_speed=suggested_entry_speed,
            ))

        self._corners = corners
        return corners

    # ------------------------------------------------------------------
    # Braking zone detection
    # ------------------------------------------------------------------

    def get_braking_zones(self, min_brake: float = 0.15, min_bins: int = 2) -> List[BrakingZone]:
        """
        Detect braking zones from the telemetry profile.

        A braking zone is a contiguous region where the typical brake
        pressure exceeds ``min_brake`` for at least ``min_bins`` segments.

        Parameters
        ----------
        min_brake : float
            Minimum typical brake value to consider as active braking.
        min_bins : int
            Minimum number of consecutive bins to qualify as a braking zone.

        Returns
        -------
        list of BrakingZone
            Sorted by start_pct.
        """
        if self._braking_zones is not None:
            return self._braking_zones

        brakes = np.array([b.typical_brake for b in self.bins])
        above = brakes >= min_brake

        zones: List[BrakingZone] = []
        i = 0
        n = self.n_bins
        while i < n:
            if not above[i]:
                i += 1
                continue

            start = i
            while i < n and above[i]:
                i += 1
            end = i - 1  # inclusive

            if (end - start + 1) < min_bins:
                continue

            entry_speed = float(self.bins[start].typical_speed)
            exit_speed = float(self.bins[end].typical_speed)
            avg_brake = float(np.mean(brakes[start:end + 1]))

            zones.append(BrakingZone(
                start_pct=float(self.bins[start].lap_dist_pct),
                end_pct=float(self.bins[end].lap_dist_pct),
                entry_speed=entry_speed,
                exit_speed=exit_speed,
                brake_intensity=avg_brake,
            ))

        self._braking_zones = zones
        return zones

    # ------------------------------------------------------------------
    # Track position estimation
    # ------------------------------------------------------------------

    def estimate_track_pos(
        self,
        lap_dist_pct: float,
        speed: float,
        steering: float,
        lat_g: float,
    ) -> float:
        """
        Estimate lateral track position from the current dynamic state.

        Compares the current steering and lateral G against the typical
        values at this track position to infer a lateral offset.

        Parameters
        ----------
        lap_dist_pct : float
            Current position on track (0 to 1).
        speed : float
            Current speed (same units/normalization as training data).
        steering : float
            Current steering input (normalized -1 to 1).
        lat_g : float
            Current lateral acceleration (normalized).

        Returns
        -------
        float
            Estimated lateral position in [-1, 1].
            0 = on the typical racing line,
            negative = inside / left of line,
            positive = outside / right of line.
        """
        b = self._get_bin(lap_dist_pct)

        # Steering deviation: how much are we steering differently from normal?
        steer_dev = 0.0
        if b.steering_std > 1e-6:
            steer_dev = (steering - b.typical_steering) / b.steering_std
        else:
            steer_dev = steering - b.typical_steering

        # Lateral G deviation: compare current lat_g to expected
        eps = 1e-6
        expected_lat_g = b.curvature * (speed ** 2 + eps)
        lat_g_dev = 0.0
        if b.lat_g_std > 1e-6:
            lat_g_dev = (lat_g - b.typical_lat_g) / b.lat_g_std
        else:
            lat_g_dev = lat_g - b.typical_lat_g

        # Combine: weighted sum of deviations, clipped to [-1, 1]
        # Steering deviation is the stronger signal for lateral offset
        # Deviations are already z-scores, so the combined value is
        # naturally in a useful range — no further scaling needed.
        track_pos = np.clip(0.6 * steer_dev + 0.4 * lat_g_dev, -1.0, 1.0)

        return float(track_pos)

    def _get_bin(self, lap_dist_pct: float) -> BinData:
        """Return the BinData for the segment containing ``lap_dist_pct``."""
        idx = int(lap_dist_pct * self.n_bins)
        idx = max(0, min(idx, self.n_bins - 1))
        return self.bins[idx]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_typical_speed_at(self, lap_dist_pct: float) -> float:
        """Return typical speed at a given track position."""
        return self._get_bin(lap_dist_pct).typical_speed

    def get_curvature_at(self, lap_dist_pct: float) -> float:
        """Return curvature at a given track position."""
        return self._get_bin(lap_dist_pct).curvature

    def get_speed_range_at(self, lap_dist_pct: float) -> Tuple[float, float]:
        """Return (min, max) observed speed at a given track position."""
        b = self._get_bin(lap_dist_pct)
        return (b.speed_min, b.speed_max)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the track map to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path (typically ``track_map.json``).
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_bins": self.n_bins,
            "bins": [asdict(b) for b in self.bins],
        }

        with open(p, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"TrackMap saved to {p}")

    @classmethod
    def load(cls, path: str) -> "TrackMap":
        """
        Load a track map from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a ``track_map.json`` file.

        Returns
        -------
        TrackMap
        """
        with open(path) as f:
            data = json.load(f)

        n_bins = data["n_bins"]
        tm = cls(n_bins=n_bins)

        for i, bd_dict in enumerate(data["bins"]):
            tm.bins[i] = BinData(**bd_dict)

        tm._compute_expected_headings()
        logger.info(f"TrackMap loaded from {path}: {n_bins} bins")
        return tm

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the track map."""
        corners = self.get_corners()
        braking = self.get_braking_zones()
        sp = self.get_speed_profile()

        lines = [
            f"TrackMap: {self.n_bins} bins",
            f"  Speed range : {sp[:, 1].min():.3f} – {sp[:, 3].max():.3f}",
            f"  Avg speed   : {sp[:, 2].mean():.3f}",
            f"  Corners     : {len(corners)}",
            f"  Braking zones: {len(braking)}",
        ]

        if corners:
            lines.append("")
            lines.append("  Corners:")
            for i, c in enumerate(corners):
                lines.append(
                    f"    {i + 1:2d}. {c.direction:5s}  "
                    f"pct={c.start_pct:.3f}→{c.apex_pct:.3f}→{c.end_pct:.3f}  "
                    f"curv={c.max_curvature:.4f}  "
                    f"entry_spd={c.suggested_entry_speed:.3f}"
                )

        if braking:
            lines.append("")
            lines.append("  Braking zones:")
            for i, bz in enumerate(braking):
                lines.append(
                    f"    {i + 1:2d}. pct={bz.start_pct:.3f}→{bz.end_pct:.3f}  "
                    f"spd={bz.entry_speed:.3f}→{bz.exit_speed:.3f}  "
                    f"brake={bz.brake_intensity:.2f}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    """Build a track map from a parquet file and print diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build a track map from processed telemetry data."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to a processed data.parquet file.",
    )
    parser.add_argument(
        "--bins", type=int, default=200,
        help="Number of track segments (default: 200).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for track_map.json (default: same dir as data).",
    )
    args = parser.parse_args()

    import pandas as pd

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading {data_path} ...")
    df = pd.read_parquet(data_path)
    logger.info(f"  {len(df):,} rows, columns: {list(df.columns)}")

    tm = TrackMap.build_from_dataframe(df, n_bins=args.bins)

    print()
    print("=" * 60)
    print(tm.summary())
    print("=" * 60)

    # Speed profile summary
    sp = tm.get_speed_profile()
    print()
    print("Speed profile (sample points):")
    print(f"  {'pct':>6s}  {'min':>8s}  {'typical':>8s}  {'max':>8s}")
    indices = np.linspace(0, len(sp) - 1, 10, dtype=int)
    for idx in indices:
        row = sp[idx]
        print(f"  {row[0]:6.3f}  {row[1]:8.4f}  {row[2]:8.4f}  {row[3]:8.4f}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = data_path.parent / "track_map.json"
    tm.save(str(out_path))


if __name__ == "__main__":
    main()
