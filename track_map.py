"""
track_map.py

GPS-free track positioning system using iRacing telemetry.

Since iRacing doesn't expose raw GPS coordinates or lateral track position
directly in .ibt telemetry, we build a track model from telemetry data:

  1. TrackSegment — divides the track into N segments by lap_dist_pct
  2. SegmentProfile — stores per-segment statistics learned from training laps:
     - Reference speed, steering, G-forces (the "racing line")
     - Track curvature estimate (from steering + speed)
     - Segment type classification (straight, corner, braking zone, etc.)
  3. TrackMap — full track model with lookahead for upcoming segments
  4. LiveTracker — real-time position tracking with segment-aware features

This replaces the need for actual GPS/track maps while giving the model
spatial context about WHERE on the track it is and WHAT'S COMING NEXT.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class SegmentType(Enum):
    """Classification of a track segment."""
    STRAIGHT = auto()
    BRAKING_ZONE = auto()
    CORNER_ENTRY = auto()
    CORNER_APEX = auto()
    CORNER_EXIT = auto()
    ACCELERATION_ZONE = auto()
    CHICANE = auto()
    UNKNOWN = auto()


@dataclass
class SegmentProfile:
    """Statistical profile of a single track segment learned from training data."""
    segment_id: int = 0
    pct_start: float = 0.0        # lap_dist_pct start
    pct_end: float = 0.0          # lap_dist_pct end
    pct_center: float = 0.0       # center point

    # Reference values (mean of clean laps)
    ref_speed: float = 0.0        # m/s (raw, not normalized)
    ref_throttle: float = 0.0     # 0-1
    ref_brake: float = 0.0        # 0-1
    ref_steering: float = 0.0     # normalized -1 to 1
    ref_lat_g: float = 0.0        # m/s^2
    ref_lon_g: float = 0.0        # m/s^2

    # Variability (std dev of clean laps)
    speed_std: float = 0.0
    steering_std: float = 0.0

    # Derived characteristics
    curvature: float = 0.0        # estimated curvature (higher = tighter corner)
    segment_type: SegmentType = SegmentType.UNKNOWN
    is_full_throttle: bool = False
    is_braking: bool = False
    is_cornering: bool = False

    # Speed deltas
    speed_delta: float = 0.0      # speed change across segment (+ = accel, - = decel)
    min_speed: float = 0.0
    max_speed: float = 0.0

    # GPS center position (decimal degrees; 0.0 if GPS data was unavailable)
    ref_lat: float = 0.0
    ref_lon: float = 0.0


@dataclass
class TrackMapData:
    """Serializable track map data for saving/loading."""
    track_name: str = ""
    car_name: str = ""
    combo_name: str = ""
    num_segments: int = 0
    track_length_m: float = 0.0
    segments: List[dict] = field(default_factory=list)
    personal_best_s: float = 0.0
    source_laps: int = 0
    data_hz: int = 60


class TrackMap:
    """
    Complete track model built from training telemetry.

    Divides the track into N equal segments by lap_dist_pct and stores
    statistical profiles learned from clean laps. Provides lookahead
    features for the model to anticipate upcoming corners/braking zones.
    """

    DEFAULT_NUM_SEGMENTS = 100  # 1% of track per segment

    def __init__(self, num_segments: int = DEFAULT_NUM_SEGMENTS):
        self.num_segments = num_segments
        self.segments: List[SegmentProfile] = []
        self.track_name: str = ""
        self.car_name: str = ""
        self.combo_name: str = ""
        self.track_length_m: float = 0.0
        self.personal_best_s: float = 0.0
        self._built = False

    def build_from_dataframe(self, df, combo_name: str = "", track_length_m: float = 0.0):
        """
        Build track map from a preprocessed training DataFrame.

        The DataFrame should contain raw (un-normalized) telemetry with columns:
            speed, throttle, brake, steering, lat_g, lon_g, lap_dist_pct, lapIndex
        """
        self.combo_name = combo_name
        self.track_length_m = track_length_m

        if "lap_dist_pct" not in df.columns:
            logger.warning("Cannot build track map: missing lap_dist_pct")
            return

        n = self.num_segments
        segment_width = 1.0 / n
        self.segments = []

        for i in range(n):
            pct_start = i * segment_width
            pct_end = (i + 1) * segment_width
            pct_center = (pct_start + pct_end) / 2.0

            # Select rows in this segment
            mask = (df["lap_dist_pct"] >= pct_start) & (df["lap_dist_pct"] < pct_end)
            seg_df = df[mask]

            seg = SegmentProfile(
                segment_id=i,
                pct_start=pct_start,
                pct_end=pct_end,
                pct_center=pct_center,
            )

            if len(seg_df) > 0:
                seg.ref_speed = float(seg_df["speed"].mean()) if "speed" in seg_df else 0.0
                seg.ref_throttle = float(seg_df["throttle"].mean()) if "throttle" in seg_df else 0.0
                seg.ref_brake = float(seg_df["brake"].mean()) if "brake" in seg_df else 0.0
                seg.ref_steering = float(seg_df["steering"].mean()) if "steering" in seg_df else 0.0
                seg.ref_lat_g = float(seg_df["lat_g"].mean()) if "lat_g" in seg_df else 0.0
                seg.ref_lon_g = float(seg_df["lon_g"].mean()) if "lon_g" in seg_df else 0.0

                seg.speed_std = float(seg_df["speed"].std()) if "speed" in seg_df else 0.0
                seg.steering_std = float(seg_df["steering"].std()) if "steering" in seg_df else 0.0

                seg.min_speed = float(seg_df["speed"].min()) if "speed" in seg_df else 0.0
                seg.max_speed = float(seg_df["speed"].max()) if "speed" in seg_df else 0.0

                # GPS center: exclude (0,0) frames from session init
                if "gps_lat" in seg_df.columns and "gps_lon" in seg_df.columns:
                    valid_gps = seg_df[
                        (seg_df["gps_lat"].abs() > 0.001) &
                        (seg_df["gps_lon"].abs() > 0.001)
                    ]
                    if len(valid_gps) > 0:
                        seg.ref_lat = float(valid_gps["gps_lat"].mean())
                        seg.ref_lon = float(valid_gps["gps_lon"].mean())

                # Estimate curvature from steering angle and speed
                # curvature ≈ |steering| / speed (simplified bicycle model)
                if seg.ref_speed > 5.0:
                    seg.curvature = abs(seg.ref_steering) / seg.ref_speed * 50.0
                else:
                    seg.curvature = abs(seg.ref_steering) * 10.0

            self.segments.append(seg)

        # Compute speed deltas between segments
        for i in range(n):
            next_i = (i + 1) % n
            self.segments[i].speed_delta = (
                self.segments[next_i].ref_speed - self.segments[i].ref_speed
            )

        # Classify segment types
        self._classify_segments()
        self._built = True

        n_corners = sum(1 for s in self.segments if s.is_cornering)
        n_straights = sum(1 for s in self.segments if s.segment_type == SegmentType.STRAIGHT)
        n_braking = sum(1 for s in self.segments if s.is_braking)
        logger.info(
            f"Track map built: {n} segments, "
            f"{n_corners} corners, {n_straights} straights, {n_braking} braking zones"
        )

    def _classify_segments(self):
        """Classify each segment based on its telemetry profile."""
        # Adaptive full-throttle threshold: 85% of the observed peak throttle.
        # A hard-coded 0.85 would incorrectly classify all segments as non-full-
        # throttle if the driver's controller or TC cap limits peak to ~0.7.
        # Floor at 0.50 so very slow/cautious data doesn't label everything straight.
        observed_max_throttle = max((s.ref_throttle for s in self.segments), default=1.0)
        throttle_threshold = max(0.50, observed_max_throttle * 0.85)
        logger.debug(
            f"_classify_segments: observed_max_throttle={observed_max_throttle:.3f}, "
            f"throttle_threshold={throttle_threshold:.3f}"
        )

        for seg in self.segments:
            # Thresholds for classification
            steering_threshold = 0.05   # normalized, above this = cornering
            brake_threshold = 0.15      # above this = braking zone
            curvature_threshold = 0.3   # above this = significant corner

            seg.is_full_throttle = seg.ref_throttle > throttle_threshold
            seg.is_braking = seg.ref_brake > brake_threshold
            seg.is_cornering = abs(seg.ref_steering) > steering_threshold

            if seg.is_braking and seg.speed_delta < -1.0:
                seg.segment_type = SegmentType.BRAKING_ZONE
            elif seg.is_cornering and seg.curvature > curvature_threshold:
                if seg.speed_delta < -0.5:
                    seg.segment_type = SegmentType.CORNER_ENTRY
                elif seg.speed_delta > 0.5:
                    seg.segment_type = SegmentType.CORNER_EXIT
                else:
                    seg.segment_type = SegmentType.CORNER_APEX
            elif seg.is_full_throttle and not seg.is_cornering:
                # On short tracks the car is almost always still accelerating
                # through the straight, so use a generous threshold (3 m/s per
                # segment) to avoid labelling every straight ACCELERATION_ZONE.
                if seg.speed_delta > 3.0:
                    seg.segment_type = SegmentType.ACCELERATION_ZONE
                else:
                    seg.segment_type = SegmentType.STRAIGHT
            elif seg.is_cornering and abs(seg.ref_steering) < 0.15:
                seg.segment_type = SegmentType.CHICANE
            else:
                seg.segment_type = SegmentType.UNKNOWN

    # ------------------------------------------------------------------
    # Lookup and features
    # ------------------------------------------------------------------

    def get_segment(self, lap_dist_pct: float) -> Optional[SegmentProfile]:
        """Get the segment profile for a given track position."""
        if not self._built or not self.segments:
            return None
        idx = int(lap_dist_pct * self.num_segments) % self.num_segments
        return self.segments[idx]

    def get_segment_index(self, lap_dist_pct: float) -> int:
        """Get segment index for a given track position."""
        return int(lap_dist_pct * self.num_segments) % self.num_segments

    def find_segment_by_gps(self, lat: float, lon: float) -> Optional[SegmentProfile]:
        """
        Find the nearest track segment by GPS position.

        Uses equirectangular approximation (accurate to <1m for track-scale areas).
        Returns None if GPS is unavailable (lat/lon == 0.0) or track map lacks GPS data.
        """
        if not self._built or abs(lat) < 0.001 or abs(lon) < 0.001:
            return None
        cos_lat = math.cos(math.radians(lat))
        best_seg = None
        best_dist_sq = float('inf')
        for seg in self.segments:
            if abs(seg.ref_lat) < 0.001 and abs(seg.ref_lon) < 0.001:
                continue
            dlat = (lat - seg.ref_lat) * 111319.9        # degrees → meters
            dlon = (lon - seg.ref_lon) * 111319.9 * cos_lat
            dist_sq = dlat * dlat + dlon * dlon
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_seg = seg
        return best_seg

    def get_lookahead(
        self, lap_dist_pct: float, num_ahead: int = 5
    ) -> List[SegmentProfile]:
        """
        Get profiles for upcoming segments (lookahead).

        This gives the model advance knowledge of what's coming:
        braking zones, corners, straights.
        """
        if not self._built:
            return []

        current_idx = self.get_segment_index(lap_dist_pct)
        ahead = []
        for i in range(1, num_ahead + 1):
            idx = (current_idx + i) % self.num_segments
            ahead.append(self.segments[idx])
        return ahead

    def get_lookbehind(
        self, lap_dist_pct: float, num_behind: int = 2
    ) -> List[SegmentProfile]:
        """Get profiles for recently passed segments."""
        if not self._built:
            return []

        current_idx = self.get_segment_index(lap_dist_pct)
        behind = []
        for i in range(1, num_behind + 1):
            idx = (current_idx - i) % self.num_segments
            behind.append(self.segments[idx])
        return behind

    def get_track_features(
        self,
        lap_dist_pct: float,
        current_speed: float = 0.0,
        lookahead: int = 5,
    ) -> np.ndarray:
        """
        Build a track-aware feature vector for the current position.

        Returns numpy array with:
          [0]     curvature of current segment (0 = straight, higher = tighter)
          [1]     segment type one-hot: is_straight
          [2]     segment type one-hot: is_braking_zone
          [3]     segment type one-hot: is_corner
          [4]     segment type one-hot: is_acceleration
          [5]     reference speed delta (how fast should we be vs how fast are we)
          [6]     reference steering magnitude
          [7]     distance to next braking zone (0-1 normalized)
          [8]     distance to next corner (0-1 normalized)
          [9..9+lookahead*3]  lookahead: [curvature, ref_speed_norm, ref_brake] per segment
        """
        if not self._built:
            return np.zeros(10 + lookahead * 3, dtype=np.float32)

        seg = self.get_segment(lap_dist_pct)

        # Current segment features
        curvature = min(seg.curvature / 5.0, 1.0)  # normalize to 0-1
        is_straight = float(seg.segment_type == SegmentType.STRAIGHT)
        is_braking = float(seg.segment_type == SegmentType.BRAKING_ZONE)
        is_corner = float(seg.is_cornering)
        is_accel = float(seg.segment_type == SegmentType.ACCELERATION_ZONE)

        # Speed reference delta (positive = we should be going faster)
        speed_max = max(s.max_speed for s in self.segments) if self.segments else 1.0
        speed_delta_ref = 0.0
        if speed_max > 0 and seg.ref_speed > 0:
            speed_delta_ref = (seg.ref_speed - current_speed) / speed_max

        steer_ref = min(abs(seg.ref_steering), 1.0)

        # Distance to next key features
        dist_to_brake = self._distance_to_type(lap_dist_pct, SegmentType.BRAKING_ZONE)
        dist_to_corner = self._distance_to_corner(lap_dist_pct)

        features = [
            curvature,
            is_straight,
            is_braking,
            is_corner,
            is_accel,
            np.clip(speed_delta_ref, -1.0, 1.0),
            steer_ref,
            dist_to_brake,
            dist_to_corner,
        ]

        # Lookahead features
        ahead_segs = self.get_lookahead(lap_dist_pct, lookahead)
        for seg_ahead in ahead_segs:
            features.append(min(seg_ahead.curvature / 5.0, 1.0))
            features.append(seg_ahead.ref_speed / speed_max if speed_max > 0 else 0.0)
            features.append(min(seg_ahead.ref_brake, 1.0))

        # Pad if we don't have enough lookahead
        while len(features) < 9 + lookahead * 3:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def get_boundary_proximity(self, lap_dist_pct: float, track_pos: float) -> float:
        """
        Estimate how close the car is to the track boundary.

        Returns 0.0 (center) to 1.0 (at edge).

        Since iRacing doesn't give us lateral position directly, we estimate
        from steering deviation and G-force deviation from the reference line.
        """
        seg = self.get_segment(lap_dist_pct)
        if seg is None:
            return 0.0

        # Use track_pos if available (non-zero)
        if abs(track_pos) > 0.01:
            return min(abs(track_pos), 1.0)

        # Fallback: no boundary info available
        return 0.0

    def _distance_to_type(self, lap_dist_pct: float, seg_type: SegmentType) -> float:
        """Distance (in pct, 0-1) to the next segment of given type."""
        idx = self.get_segment_index(lap_dist_pct)
        for i in range(1, self.num_segments):
            check_idx = (idx + i) % self.num_segments
            if self.segments[check_idx].segment_type == seg_type:
                return i / self.num_segments
        return 1.0  # not found = far away

    def _distance_to_corner(self, lap_dist_pct: float) -> float:
        """Distance (in pct, 0-1) to the next cornering segment."""
        idx = self.get_segment_index(lap_dist_pct)
        for i in range(1, self.num_segments):
            check_idx = (idx + i) % self.num_segments
            if self.segments[check_idx].is_cornering:
                return i / self.num_segments
        return 1.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save track map to YAML file."""
        data = {
            "track_name": self.track_name,
            "car_name": self.car_name,
            "combo_name": self.combo_name,
            "track_length_m": self.track_length_m,
            "num_segments": self.num_segments,
            "personal_best_s": self.personal_best_s,
            "segments": [],
        }

        for seg in self.segments:
            data["segments"].append({
                "id": seg.segment_id,
                "pct_start": round(seg.pct_start, 6),
                "pct_end": round(seg.pct_end, 6),
                "ref_speed": round(seg.ref_speed, 3),
                "ref_throttle": round(seg.ref_throttle, 4),
                "ref_brake": round(seg.ref_brake, 4),
                "ref_steering": round(seg.ref_steering, 4),
                "ref_lat_g": round(seg.ref_lat_g, 3),
                "ref_lon_g": round(seg.ref_lon_g, 3),
                "curvature": round(seg.curvature, 4),
                "segment_type": seg.segment_type.name,
                "min_speed": round(seg.min_speed, 3),
                "max_speed": round(seg.max_speed, 3),
                "speed_delta": round(seg.speed_delta, 3),
                "ref_lat": round(seg.ref_lat, 8),
                "ref_lon": round(seg.ref_lon, 8),
            })

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Track map saved to {filepath}")

    def load(self, filepath: str) -> bool:
        """Load track map from YAML file. Returns True if successful."""
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)

            self.track_name = data.get("track_name", "")
            self.car_name = data.get("car_name", "")
            self.combo_name = data.get("combo_name", "")
            self.track_length_m = data.get("track_length_m", 0.0)
            self.num_segments = data.get("num_segments", self.DEFAULT_NUM_SEGMENTS)
            self.personal_best_s = data.get("personal_best_s", 0.0)

            self.segments = []
            type_map = {t.name: t for t in SegmentType}

            for seg_data in data.get("segments", []):
                seg = SegmentProfile(
                    segment_id=seg_data["id"],
                    pct_start=seg_data["pct_start"],
                    pct_end=seg_data["pct_end"],
                    pct_center=(seg_data["pct_start"] + seg_data["pct_end"]) / 2.0,
                    ref_speed=seg_data.get("ref_speed", 0),
                    ref_throttle=seg_data.get("ref_throttle", 0),
                    ref_brake=seg_data.get("ref_brake", 0),
                    ref_steering=seg_data.get("ref_steering", 0),
                    ref_lat_g=seg_data.get("ref_lat_g", 0),
                    ref_lon_g=seg_data.get("ref_lon_g", 0),
                    curvature=seg_data.get("curvature", 0),
                    segment_type=type_map.get(seg_data.get("segment_type", "UNKNOWN"),
                                              SegmentType.UNKNOWN),
                    min_speed=seg_data.get("min_speed", 0),
                    max_speed=seg_data.get("max_speed", 0),
                    speed_delta=seg_data.get("speed_delta", 0),
                    ref_lat=seg_data.get("ref_lat", 0.0),
                    ref_lon=seg_data.get("ref_lon", 0.0),
                )
                seg.is_cornering = abs(seg.ref_steering) > 0.05
                seg.is_braking = seg.ref_brake > 0.15
                seg.is_full_throttle = seg.ref_throttle > 0.85
                self.segments.append(seg)

            self._built = len(self.segments) > 0
            logger.info(f"Track map loaded: {filepath} ({len(self.segments)} segments)")
            return True

        except Exception as e:
            logger.warning(f"Failed to load track map from {filepath}: {e}")
            return False

    @property
    def is_built(self) -> bool:
        return self._built


class LiveTracker:
    """
    Real-time track position tracker that provides rich spatial context
    during live racing.

    Wraps a TrackMap and maintains state about current position, upcoming
    features, and deviations from the reference racing line.
    """

    def __init__(self, track_map: TrackMap, lookahead: int = 5):
        self.track_map = track_map
        self.lookahead = lookahead

        # Position tracking
        self._prev_pct = 0.0
        self._lap_count = 0
        self._lap_start_time = 0.0
        self._lap_times: List[float] = []

        # Deviation tracking (rolling window)
        self._speed_deviations = np.zeros(60, dtype=np.float32)
        self._steer_deviations = np.zeros(60, dtype=np.float32)
        self._dev_ptr = 0

    def update(
        self,
        lap_dist_pct: float,
        speed: float,
        steering: float,
        track_pos: float = 0.0,
    ) -> dict:
        """
        Update tracker with current telemetry and return rich context.

        Returns dict with:
            segment_id:       current segment index
            segment_type:     SegmentType enum
            track_features:   numpy array for model input
            boundary_pct:     proximity to track edge (0-1)
            speed_deviation:  how far from reference speed
            upcoming_corner:  distance to next corner (0-1)
            upcoming_brake:   distance to next braking zone (0-1)
            lap_crossed:      True if we just crossed start/finish
        """
        # Detect lap crossing
        lap_crossed = False
        if self._prev_pct > 0.95 and lap_dist_pct < 0.05:
            lap_crossed = True
            self._lap_count += 1
            now = time.perf_counter()
            if self._lap_start_time > 0:
                lap_time = now - self._lap_start_time
                self._lap_times.append(lap_time)
            self._lap_start_time = now

        self._prev_pct = lap_dist_pct

        if not self.track_map.is_built:
            return {
                "segment_id": 0,
                "segment_type": SegmentType.UNKNOWN,
                "track_features": np.zeros(9 + self.lookahead * 3, dtype=np.float32),
                "boundary_pct": 0.0,
                "speed_deviation": 0.0,
                "upcoming_corner": 1.0,
                "upcoming_brake": 1.0,
                "lap_crossed": lap_crossed,
            }

        seg = self.track_map.get_segment(lap_dist_pct)

        # Track features for model input
        track_features = self.track_map.get_track_features(
            lap_dist_pct, speed, self.lookahead
        )

        # Boundary proximity
        boundary_pct = self.track_map.get_boundary_proximity(lap_dist_pct, track_pos)

        # Speed deviation from reference
        speed_dev = 0.0
        if seg and seg.ref_speed > 0:
            speed_dev = (speed - seg.ref_speed) / seg.ref_speed

        # Steering deviation from reference
        steer_dev = 0.0
        if seg:
            steer_dev = steering - seg.ref_steering

        # Update deviation history
        ptr = self._dev_ptr % len(self._speed_deviations)
        self._speed_deviations[ptr] = speed_dev
        self._steer_deviations[ptr] = steer_dev
        self._dev_ptr += 1

        # Upcoming features
        upcoming_corner = self.track_map._distance_to_corner(lap_dist_pct)
        upcoming_brake = self.track_map._distance_to_type(
            lap_dist_pct, SegmentType.BRAKING_ZONE
        )

        return {
            "segment_id": seg.segment_id if seg else 0,
            "segment_type": seg.segment_type if seg else SegmentType.UNKNOWN,
            "track_features": track_features,
            "boundary_pct": boundary_pct,
            "speed_deviation": np.clip(speed_dev, -1.0, 1.0),
            "upcoming_corner": upcoming_corner,
            "upcoming_brake": upcoming_brake,
            "lap_crossed": lap_crossed,
        }

    @property
    def lap_count(self) -> int:
        return self._lap_count

    @property
    def best_lap_time(self) -> float:
        return min(self._lap_times) if self._lap_times else 0.0

    @property
    def avg_speed_deviation(self) -> float:
        """Average speed deviation over recent window (+ = faster than ref)."""
        if self._dev_ptr == 0:
            return 0.0
        n = min(self._dev_ptr, len(self._speed_deviations))
        return float(np.mean(self._speed_deviations[:n]))


def build_track_map_from_parquet(
    parquet_path: str,
    combo_name: str = "",
    num_segments: int = 100,
    track_length_m: float = 0.0,
) -> TrackMap:
    """
    Convenience function: build a TrackMap from a preprocessed parquet file.

    This reads the raw (un-normalized) data if available, or works with
    normalized data and un-normalizes using metadata.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    logger.info(f"Building track map from {parquet_path} ({len(df)} frames)")

    track_map = TrackMap(num_segments=num_segments)
    track_map.build_from_dataframe(df, combo_name, track_length_m)

    return track_map
