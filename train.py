"""
trainer/train.py

Behavior cloning training loop.

Usage:
  python trainer/train.py --data data/processed/ --track sebring --car mx5
  python trainer/train.py --data data/processed/ --all   # train all track/car combos

Checkpoints saved to: checkpoints/<trackId>_<carId>/best.pt
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from loader import load_track_car_dataset
from dataset import TelemetryDataset, split_dataset, build_track_features_for_dataset
from model import DrivingPolicyNet, BehaviorCloningLoss
from track_map import TrackMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _resolve_sequence_history(cfg)
    return cfg


def _resolve_sequence_history(cfg: dict):
    """
    If sequence_history is 'auto', compute it from context_window_ms and data_hz.
    Modifies cfg in-place.
    """
    train_cfg = cfg["training"]
    if train_cfg.get("sequence_history") == "auto":
        hz = int(train_cfg.get("data_hz", 60))
        ms = float(train_cfg.get("context_window_ms", 250))
        train_cfg["sequence_history"] = max(1, round(ms / 1000.0 * hz))
        logger.info(
            f"sequence_history=auto resolved to {train_cfg['sequence_history']} "
            f"frames ({ms:.0f}ms @ {hz}Hz)"
        )


def train_one_combo(
    combo_folder: str,
    combo_name: str,
    cfg: dict,
    device: torch.device,
) -> bool:
    """
    Train a model for one track/car combination.
    Returns True on success.
    """
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {combo_name}")
    logger.info(f"{'='*60}")

    # --- Data ---
    parquet_path = os.path.join(combo_folder, "data.parquet")
    if os.path.exists(parquet_path):
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df):,} frames from {parquet_path}")
    else:
        df = load_track_car_dataset(combo_folder, cfg)
    if df is None or len(df) == 0:
        logger.warning(f"Skipping {combo_name}: no data")
        return False

    # --- Normalization constants from meta.yaml ---
    # These must be saved in the checkpoint so inference uses the EXACT same
    # scales as training (speed_max, steering_lock, rpm_max).
    norm_constants = {}
    meta_path = Path(combo_folder) / "meta.yaml"
    if meta_path.exists():
        with open(meta_path) as _mf:
            _meta = yaml.safe_load(_mf)
        norm_constants = {
            "speed_max_ms":           _meta.get("speed_max_ms"),
            "steering_lock_radians":  _meta.get("steering_lock_radians"),
            "rpm_max":                _meta.get("rpm_max"),
        }
        logger.info(
            f"Norm constants from meta.yaml: "
            f"speed_max={norm_constants['speed_max_ms']:.2f} m/s  "
            f"steer_lock={norm_constants['steering_lock_radians']:.4f} rad  "
            f"rpm_max={norm_constants['rpm_max']:.0f}"
        )

    # --- Track features ---
    track_features = None
    track_cfg = cfg.get("track", {})
    lookahead = track_cfg.get("lookahead_segments", 5)

    # Look for track map in checkpoints or processed data folder
    for map_dir in [
        Path(cfg["paths"]["checkpoints"]) / combo_name,
        Path(combo_folder),
    ]:
        map_path = map_dir / "track_map.yaml"
        if map_path.exists():
            tmap = TrackMap()
            if tmap.load(str(map_path)):
                track_features = build_track_features_for_dataset(
                    df, tmap, lookahead=lookahead
                )
                logger.info(
                    f"Track features: {track_features.shape[1]} dims "
                    f"(lookahead={lookahead}) from {map_path}"
                )
            break

    try:
        full_dataset = TelemetryDataset(
            df, train_cfg["sequence_history"],
            track_features=track_features,
        )
    except ValueError as e:
        logger.warning(f"Skipping {combo_name}: {e}")
        return False

    train_set, val_set = split_dataset(
        full_dataset,
        val_fraction=train_cfg["val_split"],
        seed=train_cfg["seed"],
    )

    logger.info(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")
    logger.info(f"Input dim: {full_dataset.input_dim}  |  Output dim: {full_dataset.output_dim}")

    # num_workers=0 on Windows avoids slow multiprocessing spawn overhead;
    # dataset fits in memory so DataLoader overhead is minimal
    num_workers = 0 if sys.platform == "win32" else 4
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=train_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- Model ---
    model = DrivingPolicyNet(
        input_dim=full_dataset.input_dim,
        hidden_dims=tuple(model_cfg["hidden_dims"]),
        dropout=model_cfg["dropout"],
    ).to(device)

    logger.info(f"Model parameters: {model.parameter_count:,}")

    criterion = BehaviorCloningLoss(
        throttle_weight=train_cfg["throttle_loss_weight"],
        brake_weight=train_cfg["brake_loss_weight"],
        steering_weight=train_cfg["steering_loss_weight"],
        smoothness_weight=train_cfg["smoothness_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg["lr_factor"],
        patience=train_cfg["lr_patience"],
        verbose=True,
    )

    # --- Checkpoint dir ---
    checkpoint_dir = Path(cfg["paths"]["checkpoints"]) / combo_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"

    # --- Training loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 50

    for epoch in range(1, train_cfg["epochs"] + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_components = {"throttle": 0, "brake": 0, "steering": 0, "smoothness": 0, "boundary": 0}
        t0 = time.time()

        for batch_state, batch_action in train_loader:
            batch_state = batch_state.to(device, non_blocking=True)
            batch_action = batch_action.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred_thr, pred_brk, pred_str = model(batch_state)

            # Use previous steering from history as smoothness reference
            # History index: last steering value is at state[n_state_features + 2]
            # (history is [t-1: throttle, brake, steering, steer_delta, ...])
            prev_str = batch_state[:, full_dataset.state_arr.shape[1] + 2:
                                      full_dataset.state_arr.shape[1] + 3]

            loss, components = criterion(
                pred_thr, pred_brk, pred_str, batch_action, prev_str
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            for k in components:
                train_components[k] += components[k]

        n_batches = len(train_loader)
        train_loss = train_loss_sum / n_batches

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch_state, batch_action in val_loader:
                batch_state = batch_state.to(device, non_blocking=True)
                batch_action = batch_action.to(device, non_blocking=True)

                pred_thr, pred_brk, pred_str = model(batch_state)
                loss, _ = criterion(pred_thr, pred_brk, pred_str, batch_action)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        scheduler.step(val_loss)

        # Logging
        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{train_cfg['epochs']}  "
                f"train={train_loss:.5f}  val={val_loss:.5f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({elapsed:.1f}s)"
            )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "input_dim": full_dataset.input_dim,
                "output_dim": full_dataset.output_dim,
                "cfg": cfg,
                "combo_name": combo_name,
                "norm": norm_constants,
            }, best_path)
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(no improvement for {early_stop_patience} epochs)")
            break

    # Save last checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        "input_dim": full_dataset.input_dim,
        "output_dim": full_dataset.output_dim,
        "cfg": cfg,
        "combo_name": combo_name,
        "norm": norm_constants,
    }, last_path)

    logger.info(f"✓ Best val loss: {best_val_loss:.5f}  Checkpoint: {best_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Train iRacing behavior cloning model")
    parser.add_argument("--data", default="data/processed/",
                        help="Root folder containing track/car subfolders")
    parser.add_argument("--track", default=None,
                        help="Track ID filter (e.g. 'sebring')")
    parser.add_argument("--car", default=None,
                        help="Car ID filter (e.g. 'mx5')")
    parser.add_argument("--combo", default=None,
                        help="Combo name filter (e.g. 'cadillacctsvr_summitpoint')")
    parser.add_argument("--all", action="store_true",
                        help="Train all track/car combos found in data folder")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cuda', 'cpu'")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path(args.data)
    if not data_root.exists():
        logger.error(f"Data folder not found: {data_root}")
        sys.exit(1)

    # Find combos to train
    if args.all:
        combos = [d for d in data_root.iterdir() if d.is_dir()]
    elif args.combo:
        combos = [
            d for d in data_root.iterdir()
            if d.is_dir() and args.combo.lower() in d.name.lower()
        ]
    elif args.track or args.car:
        pattern = f"{args.track or '*'}_{args.car or '*'}"
        combos = list(data_root.glob(pattern))
        if not combos:
            # Try to find any folder containing the filters
            combos = [
                d for d in data_root.iterdir()
                if d.is_dir()
                and (args.track is None or args.track.lower() in d.name.lower())
                and (args.car is None or args.car.lower() in d.name.lower())
            ]
    else:
        logger.error("Specify --combo, --track/--car, or --all")
        sys.exit(1)

    if not combos:
        logger.error(f"No matching track/car combos found in {data_root}")
        sys.exit(1)

    logger.info(f"Found {len(combos)} combo(s) to train")

    success = 0
    for combo_path in sorted(combos):
        ok = train_one_combo(
            str(combo_path),
            combo_path.name,
            cfg,
            device,
        )
        if ok:
            success += 1

    logger.info(f"\nCompleted: {success}/{len(combos)} combos trained successfully")


if __name__ == "__main__":
    main()
