# AI Racer — Build List

iRacing behavior cloning bot: reads `.ibt` telemetry → trains PyTorch MLP → outputs via vJoy at 360Hz.

---

## Status Key
- [ ] Not started
- [~] In progress
- [x] Done

---

## 0 — Foundation (done)
- [x] Project structure and module layout
- [x] `config.yaml` — all hyperparameters and paths
- [x] `requirements.txt`
- [x] `.gitignore` (excludes venv, `.ibt`, checkpoints)
- [x] GitHub repo: https://github.com/nicholsbw77/AI-Racer

---

## 1 — Data Pipeline  ← START HERE

The biggest gap: all data is `.ibt` (iRacing binary) but the current
loader only handles VRS CSV exports. Must fix before anything else works.

- [ ] **`ibt_loader.py`** — Read `.ibt` files into DataFrames
  - Use `irsdk` library to open `.ibt` replay files
  - Extract: Speed, Throttle, Brake, SteeringWheelAngle, Gear, RPM,
    LatAccel, LongAccel, LapDistPct, SessionTime, Lap
  - Map to the same canonical column names used in `loader.py`
    (`speed`, `throttle`, `brake`, `steering`, `gear`, `rpm`,
    `lat_g`, `lon_g`, `lap_dist_pct`, `lap_time`)
  - Handle per-lap segmentation (detect lap crossings)
  - Return a cleaned DataFrame compatible with `normalize_features()`

- [ ] **`inspect_ibt.py`** — Diagnostic script for `.ibt` files
  - Replace `inspect_csv.py` for the new data format
  - Print all available channels, sample rates, session info
  - Verify all required channels are present
  - Usage: `python inspect_ibt.py data/some_file.ibt`

- [ ] **Update `loader.py`** — Add `.ibt` loading path
  - `load_ibt_file(filepath)` function alongside `load_vrs_csv()`
  - Auto-detect file type by extension in `load_track_car_dataset()`
  - Update `COLUMN_CANDIDATES` if iRacing channel names differ

- [ ] **Update `preprocess.py`** — Handle `.ibt` source files
  - Scan `data/` folder for `.ibt` files grouped by `{car}_{track}`
  - (Filenames already follow pattern: `{car}_{track} {datetime}.ibt`)
  - Parse car/track from filename
  - Output to `data/processed/{track}_{car}/data.parquet` + `meta.yaml`

- [ ] **Add `irsdk` to `requirements.txt`**
  - `pyirsdk>=1.4.0` (uncomment existing commented line)

---

## 2 — Training

- [ ] Smoke-test full pipeline on one `.ibt` file end-to-end
  - Run `inspect_ibt.py` on a sample file
  - Run `preprocess.py` → verify `.parquet` output
  - Run `train.py` on processed data → verify loss decreases

- [ ] Tune `config.yaml` based on actual data
  - Confirm `sequence_history` is appropriate for `.ibt` sample rate
  - `.ibt` files are recorded at 60Hz (not 360Hz) — may need adjustment
  - Update `steering_lock_radians` per car (MX-5=450°, AMG GT3=450°, Cadillac CTS-VR=?)

- [ ] Train models for each track/car combo in `data/`
  - `mercedesamgevogt3` @ Watkins Glen
  - `mercedesamgevogt3` @ Phillipis Island
  - `mercedesamgevogt3` @ Bathurst
  - `mercedesamgevogt3` @ Skidpad
  - `cadillacctsvr` @ Laguna Seca
  - `cadillacctsvr` @ Daytona Road
  - `mercedesamggt4` @ Suzuka
  - `mx5` @ Okayama
  - `dirtsprint winged 360` @ Limaland
  - `dirtmodified bigblock` @ Eldora
  - `dirtsprint nonwinged 410` @ Port Royal

---

## 3 — Live Agent

Requires iRacing running on Windows with pyirsdk connected.

- [ ] Install and configure vJoy driver
  - Download from sourceforge.net/projects/vjoystick
  - Enable Device 1 with at least 3 axes (X=steering, Y=throttle, Z=brake)

- [ ] Map vJoy Device 1 in iRacing controls options

- [ ] Test `controller.py` in mock mode
  - `python orchestrator.py --mock --combo test`

- [ ] Test full live loop
  - `python orchestrator.py --auto`

---

## 4 — Track Navigation & Staying On Track

- [x] **`track_pos` estimation** — Fixed hardcoded 0.0 in telemetry.py.
  Now estimates lateral position from dynamics (yaw rate, lat_g, steering).
  Falls back to track map integration when available.
- [x] **Track boundary features** — Added `near_edge`, `on_rumble`,
  `track_pos_sign` to state vector. Model now sees when it's near edges.
- [x] **`safety_controller.py`** — Safety layer between model and controller:
  - Off-track recovery (hard brake + steer toward center)
  - Edge proximity warnings (progressive throttle reduction + corrective steering)
  - Excessive lateral G protection
  - Configurable blend factors (safety never fully overrides model)
- [x] **`track_map.py`** — Builds virtual track profile from telemetry data:
  - Curvature, speed, and steering profiles per track segment
  - Corner detection, braking zone identification
  - Speed envelope (safe min/max per position)
  - Track position estimation from dynamics
- [x] **`track_simulator.py`** — Offline 2D track simulator:
  - Bicycle model physics (no iRacing needed)
  - Built-in tracks: oval, road course, figure-eight
  - State vector compatible with trained models
  - PID demo controller for testing
- [x] **Boundary-aware training loss** — Added `boundary_weight` loss term
  that increases steering error penalty near track edges

---

## 5 — Quality / Nice-to-have

- [ ] `evaluate.py` — offline lap simulation: replay `.ibt`, compare
  predicted inputs vs actual, compute MSE per channel
- [ ] Lap time predictor: given model outputs, estimate lap delta vs PB
- [ ] Tensorboard / W&B logging in `train.py`
- [ ] GitHub Actions CI: lint + unit tests on push
- [ ] DAgger (Dataset Aggregation) for iterative improvement against covariate shift
- [ ] Track-specific speed limit profiles in safety controller

---

## Known Issues / Notes

- **`.ibt` sample rate**: iRacing `.ibt` files record at **60Hz**, not 360Hz.
  The live agent runs at 360Hz via `wait_for_data()`. Training data will be
  at 60Hz — the model will still work but `sequence_history=15` covers
  250ms at 60Hz vs 42ms at 360Hz. May want to reduce to `sequence_history=5`
  for 60Hz training data.

- **`track_pos` estimation**: Now estimated from dynamics rather than
  hardcoded to 0.0. Accuracy improves significantly when a TrackMap is
  built from training data. Without a track map, uses lateral G integration.

- **`inspect_csv.py`**: Imports from `trainer.loader` using a `trainer/`
  subpackage path. Current file layout is flat — either reorganize into
  `trainer/` and `agent/` subpackages, or update imports.

- **File naming**: `.ibt` filenames encode car and track
  (e.g. `cadillacctsvr_lagunaseca 2023-08-16 ...ibt`).
  The preprocessor can parse car/track directly from filenames.
