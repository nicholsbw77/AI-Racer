# Telemetry Requirements for a Lap-Turning AI Model

How much data does a behavior cloning model actually need to turn consistent,
clean laps in iRacing? This document covers the minimum viable telemetry,
recommended data volumes, and the signals that matter most.

---

## 1. Minimum Viable Telemetry Channels

These are the **absolute minimum** channels needed to complete a lap:

| Channel | iRacing Name | Unit | Why It's Essential |
|---------|-------------|------|-------------------|
| Speed | `Speed` | m/s | Core state — model needs to know how fast it's going |
| Throttle | `Throttle` | 0-1 | Training target + history context |
| Brake | `Brake` | 0-1 | Training target + braking zone learning |
| Steering | `SteeringWheelAngle` | rad | Training target + cornering context |
| Track Position | `LapDistPct` | 0-1 | WHERE on track — this is the "GPS" |
| Gear | `Gear` | int | Shift timing, speed/RPM relationship |
| RPM | `RPM` | float | Shift points, engine state |

**With just these 7 channels, a model CAN learn to turn laps** — but it will be
rough, inconsistent, and crash frequently.

---

## 2. Recommended Telemetry for Reliable Laps

Adding these channels dramatically improves consistency:

| Channel | iRacing Name | Unit | What It Adds |
|---------|-------------|------|-------------|
| Lateral G | `LatAccel` | m/s² | Cornering load — prevents oversteer |
| Longitudinal G | `LongAccel` | m/s² | Braking/accel intensity |
| Yaw Rate | `YawRate` | rad/s | Spin detection, car rotation awareness |
| Lateral Velocity | `VelocityY` | m/s | Slip angle — is the car sliding? |
| Lap Counter | `Lap` | int | Lap boundary detection for clean data segmentation |
| Session Time | `SessionTime` | s | Timing for lap time computation |

**Total recommended: 13 channels** (7 minimum + 6 dynamics).

The dynamics channels (G-forces, yaw rate, slip) are what let the model
understand the car's **physical state**, not just its **position and speed**.
Without them, the model can't distinguish between:
- "I'm going 100 km/h on a straight" vs "I'm going 100 km/h mid-corner pulling 2G"
- "I'm braking normally" vs "I'm locking up and sliding"

---

## 3. How Much Training Data Is Needed?

### Per Track/Car Combo

| Data Level | Laps | Frames (60Hz) | Real Time | Quality |
|-----------|------|---------------|-----------|---------|
| **Bare minimum** | 5-10 | ~18,000-36,000 | 5-10 min | Can complete laps but inconsistent, frequent off-tracks |
| **Functional** | 20-30 | ~72,000-108,000 | 20-30 min | Consistent laps, occasional wobbles in complex corners |
| **Good** | 50-80 | ~180,000-288,000 | 50-80 min | Clean laps within 3-5% of personal best |
| **Excellent** | 100-200 | ~360,000-720,000 | 1.5-3.5 hrs | Near-expert performance, handles edge cases |

### Key Insight: Quality > Quantity

**10 clean laps beat 100 sloppy laps.** The `clean_lap_threshold: 1.01` filter
(keep only laps within 1% of personal best) is critical. Training on inconsistent
data teaches the model to be inconsistent.

### Data Requirements by Track Complexity

| Track Type | Min Clean Laps | Notes |
|-----------|---------------|-------|
| Oval (Daytona, etc) | 5-10 | Simple, repetitive — needs very little data |
| Short circuit (Okayama) | 15-25 | Few corners but tight, needs precision |
| Medium circuit (Laguna Seca) | 30-50 | Mix of corner types, elevation changes |
| Long circuit (Bathurst, Spa) | 50-100 | Many unique corners, each needs examples |
| Street circuit (Long Beach) | 60-120 | Tight walls, no margin — needs lots of data |

---

## 4. Recording Rate: 60Hz vs 360Hz

### 60Hz (Default .ibt)

- **1 sample every 16.7ms**
- At 200 km/h (55 m/s), the car moves **0.92 meters** between samples
- **Sufficient for most behavior cloning** — steering, throttle, and brake
  don't change dramatically in 17ms
- **~3,600 samples per lap** (at 60s/lap)

### 360Hz (Optional iRacing setting)

- **1 sample every 2.8ms**
- At 200 km/h, the car moves **0.15 meters** between samples
- **Useful for**: high-speed corner entry, trail braking precision,
  catching slides at the limit
- **~21,600 samples per lap** — 6x more data
- Enable in `app.ini`: `[DataServerSubscriptions] iRacingDataRate=360`

### Recommendation

**Start with 60Hz.** It's enough to learn the racing line, braking points,
and cornering. Only move to 360Hz if you need sub-second precision in
transitions (e.g., trail braking, fast chicanes).

---

## 5. Feature Vector Breakdown

The model's input state vector contains three parts:

### Part A: Current State (13 features)

| # | Feature | Range | Source |
|---|---------|-------|--------|
| 0 | `lap_dist_pct` | [0, 1] | Where on track |
| 1 | `speed` | [0, 1] | Current speed (normalized) |
| 2 | `speed_delta` | [-0.5, 0.5] | Acceleration/deceleration |
| 3 | `gear` | [0, 1] | Current gear (normalized) |
| 4 | `rpm` | [0, 1] | Engine RPM (normalized) |
| 5 | `lat_g` | [-1, 1] | Lateral G-force |
| 6 | `lon_g` | [-1, 1] | Longitudinal G-force |
| 7 | `track_pos` | [-1, 1] | Lateral track position |
| 8 | `steering_abs` | [0, 1] | Steering magnitude |
| 9 | `heavy_braking` | {0, 1} | Braking zone flag |
| 10 | `full_throttle` | {0, 1} | Full throttle flag |
| 11 | `yaw_rate` | [-1, 1] | Yaw rotation rate |
| 12 | `slip_angle` | [-1, 1] | Estimated slip angle |

### Part B: Action History (4 features x N frames)

| Feature | Frames | Total |
|---------|--------|-------|
| `throttle` | 15 (at 60Hz = 250ms) | 15 |
| `brake` | 15 | 15 |
| `steering` | 15 | 15 |
| `steering_delta` | 15 | 15 |
| **Subtotal** | | **60** |

### Part C: Track Features (optional, from TrackMap)

| # | Feature | Range | Source |
|---|---------|-------|--------|
| 0 | curvature | [0, 1] | Current segment curvature |
| 1 | is_straight | {0, 1} | Segment type |
| 2 | is_braking_zone | {0, 1} | Segment type |
| 3 | is_corner | {0, 1} | Segment type |
| 4 | is_acceleration | {0, 1} | Segment type |
| 5 | speed_delta_ref | [-1, 1] | Ref speed vs actual |
| 6 | steering_ref | [0, 1] | Reference steering magnitude |
| 7 | dist_to_brake | [0, 1] | Distance to next braking zone |
| 8 | dist_to_corner | [0, 1] | Distance to next corner |
| 9-23 | lookahead[0..4] x 3 | various | curvature, ref_speed, ref_brake for 5 ahead segments |
| **Subtotal** | | **24** |

### Total Input Dimension

| Config | Dimension | Notes |
|--------|-----------|-------|
| Without track features | 13 + 60 = **73** | Basic model |
| With track features | 13 + 60 + 24 = **97** | Track-aware model |
| With 360Hz (90 history frames) | 13 + 360 + 24 = **397** | High-res model |

---

## 6. What Makes the Difference for Lap Completion

Ranked by importance for actually completing laps without crashing:

### Critical (must have)
1. **`lap_dist_pct`** — The single most important feature. Without knowing
   WHERE you are on track, the model can't know WHAT to do.
2. **`speed`** — The model must know how fast it's going.
3. **`steering` history** — Temporal context prevents oscillation.
4. **Clean training data** — Train on your best laps only.

### Very Important (big quality improvement)
5. **`lat_g` / `lon_g`** — Physics awareness prevents over-driving.
6. **Track lookahead** — Knowing a corner is coming prevents late braking.
7. **`brake` history** — Smooth trail-braking transitions.
8. **Sufficient laps (30+)** — The model needs to see each corner many times.

### Important (refinement)
9. **`yaw_rate`** — Catch slides before they become spins.
10. **`speed_delta`** — Helps the model predict momentum.
11. **`gear` / `rpm`** — Shift timing matters for traction.
12. **EMA smoothing** — Prevents jittery outputs from ruining stability.

### Nice to Have (marginal gains)
13. **`slip_angle`** — Fine control at the limit of adhesion.
14. **`track_pos`** — Lateral position (hard to get from iRacing).
15. **360Hz data** — Only matters for aggressive driving styles.
16. **Smoothness penalty** — Reduces steering oscillation in training.

---

## 7. Quick-Start Data Collection Guide

### Step 1: Record Training Laps
1. Open iRacing, load a practice session
2. iRacing automatically records `.ibt` files to `~/Documents/iRacing/telemetry/`
3. Drive **30-50 clean laps** at a consistent pace (no experiments/crashes)
4. Copy `.ibt` files to `AI-Racer/data/`

### Step 2: Preprocess
```bash
python preprocess.py --input data/ --output data/processed/
```
This will:
- Parse car/track from filenames
- Normalize and engineer features
- Filter to clean laps (within 1% of your PB)
- Build a TrackMap with 100 segments
- Output `.parquet` + `meta.yaml` + `track_map.yaml`

### Step 3: Check Your Data
After preprocessing, check `meta.yaml`:
- `clean_laps` should be **15+** (absolute minimum for a functional model)
- `total_frames` should be **50,000+** for decent results
- `personal_best_s` should match your actual PB

### Step 4: Train
```bash
python train.py --data data/processed/ --all
```

### Step 5: Test
```bash
python orchestrator.py --auto
```

---

## 8. Troubleshooting: Common Data Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Model weaves/oscillates | Not enough history context or smoothness weight too low | Increase `context_window_ms` to 400-500, increase `smoothness_weight` to 0.25 |
| Brakes too late | Not enough braking zone examples | Record more laps with consistent braking, ensure `brake_loss_weight >= 1.2` |
| Misses apexes | `lap_dist_pct` not providing enough spatial resolution | Enable track map features (`--build-map`), increase `num_segments` to 200 |
| Spins on corner exit | No yaw rate / slip angle data | Ensure `YawRate` and `VelocityY` are in your .ibt channels |
| Jerky throttle | EMA alpha too high | Reduce `ema_alpha` to 0.15-0.20 for smoother output |
| Won't complete a lap | Insufficient data OR data quality issues | Need 15+ clean laps; check data isn't all pit laps or warmup laps |
