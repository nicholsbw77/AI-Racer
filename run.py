"""
run.py

Turnkey launcher for the iRacing AI bot.
Performs pre-flight checks, then starts the orchestrator.

Usage:
  python run.py                    # auto-detect track/car from iRacing
  python run.py --combo cadillacctsvr_lagunaseca
  python run.py --mock             # dry run without iRacing/vJoy
  python run.py --check            # run checks only, don't start bot

Pre-flight checks:
  1. Python version and required packages
  2. Config file exists and is valid
  3. Checkpoint exists for the requested combo
  4. Track map exists (optional but recommended)
  5. Normalization constants present in checkpoint
  6. vJoy driver installed and enabled (Windows only)
  7. iRacing running and connected (unless --mock)
"""

import sys
import os
import logging
import argparse
from pathlib import Path

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 9):
        logger.error(f"Python 3.9+ required, got {v.major}.{v.minor}")
        return False
    logger.info(f"  Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_packages() -> bool:
    missing = []
    for pkg in ["torch", "numpy", "yaml", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"  Missing packages: {missing}")
        logger.error(f"  Run: pip install -r requirements.txt")
        return False
    import torch
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    logger.info(f"  PyTorch {torch.__version__} ({device})")
    return True


def check_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.error(f"  Config not found: {config_path}")
        return None
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    required_keys = ["training", "model", "inference", "paths"]
    for k in required_keys:
        if k not in cfg:
            logger.error(f"  Config missing section: {k}")
            return None
    logger.info(f"  Config OK ({config_path})")
    return cfg


def check_checkpoint(cfg: dict, combo_name: str) -> bool:
    if combo_name is None:
        logger.info("  Checkpoint: will auto-detect from iRacing session")
        return True
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / combo_name
    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists():
        logger.error(f"  No checkpoint: {best_pt}")
        logger.error(f"  Train first: python train.py --data data/processed/ --all")
        return False

    import torch
    ckpt = torch.load(best_pt, map_location="cpu")
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", "?")
    logger.info(f"  Checkpoint OK: epoch={epoch}, val_loss={val_loss:.5f}" if isinstance(val_loss, float) else f"  Checkpoint OK: epoch={epoch}")

    # Check norm constants
    norm = ckpt.get("norm", {})
    if norm.get("speed_max_ms"):
        logger.info(f"  Norm constants: speed_max={norm['speed_max_ms']:.1f}m/s, "
                     f"steer_lock={norm.get('steering_lock_radians', 'N/A')}")
    else:
        logger.warning("  Norm constants not in checkpoint (older format)")
        logger.warning("  Track position estimation may be less accurate")
        logger.warning("  Re-train to include norm constants")

    # Check track map
    track_map_path = ckpt_dir / "track_map.json"
    if track_map_path.exists():
        logger.info(f"  Track map: {track_map_path}")
    else:
        logger.warning("  No track map found — track position estimation will use fallback")
        logger.warning("  Re-train to auto-generate track map")

    return True


def check_vjoy() -> bool:
    if sys.platform != "win32":
        logger.info("  vJoy: skipped (not Windows)")
        return True
    dll_path = r"C:\Program Files\vJoy\x64\vJoyInterface.dll"
    if not os.path.exists(dll_path):
        logger.error(f"  vJoy not installed: {dll_path}")
        logger.error("  Download: https://sourceforge.net/projects/vjoystick/")
        return False
    logger.info("  vJoy driver found")
    return True


def check_iracing() -> bool:
    try:
        import irsdk
        ir = irsdk.IRSDK()
        if ir.startup() and ir.is_connected:
            logger.info("  iRacing connected")
            ir.shutdown()
            return True
        else:
            logger.warning("  iRacing not running — start iRacing first")
            return False
    except ImportError:
        logger.warning("  pyirsdk not installed — cannot check iRacing connection")
        return True


def list_available_combos(cfg: dict):
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    if not ckpt_dir.exists():
        logger.info("\n  No checkpoints directory found")
        return
    combos = [d.name for d in ckpt_dir.iterdir()
              if d.is_dir() and (d / "best.pt").exists()]
    if combos:
        logger.info(f"\n  Available combos ({len(combos)}):")
        for c in sorted(combos):
            has_map = (ckpt_dir / c / "track_map.json").exists()
            map_str = " [+track_map]" if has_map else ""
            logger.info(f"    - {c}{map_str}")
    else:
        logger.info("\n  No trained models found. Run training first.")


def run_preflight(cfg: dict, combo_name: str, mock: bool) -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    logger.info("=" * 50)
    logger.info("PRE-FLIGHT CHECKS")
    logger.info("=" * 50)

    ok = True

    logger.info("[1/5] Environment")
    ok &= check_python_version()
    ok &= check_packages()

    logger.info("[2/5] Configuration")
    if cfg is None:
        return False

    logger.info("[3/5] Model checkpoint")
    ok &= check_checkpoint(cfg, combo_name)

    logger.info("[4/5] vJoy controller")
    if mock:
        logger.info("  Skipped (mock mode)")
    else:
        ok &= check_vjoy()

    logger.info("[5/5] iRacing connection")
    if mock:
        logger.info("  Skipped (mock mode)")
    else:
        ir_ok = check_iracing()
        if not ir_ok:
            logger.info("  (bot will wait for iRacing to start)")

    list_available_combos(cfg)

    logger.info("=" * 50)
    if ok:
        logger.info("ALL CHECKS PASSED — ready to race")
    else:
        logger.error("CHECKS FAILED — fix issues above before starting")
    logger.info("=" * 50)

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="iRacing AI Bot — Turnkey Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # auto-detect, go live
  python run.py --combo cadillacctsvr_lagunaseca
  python run.py --mock                   # test without hardware
  python run.py --check                  # pre-flight only
  python run.py --list                   # list trained combos
        """,
    )
    parser.add_argument("--combo", default=None,
                        help="Track/car combo name (e.g. 'cadillacctsvr_lagunaseca')")
    parser.add_argument("--mock", action="store_true",
                        help="Dry run without iRacing or vJoy")
    parser.add_argument("--check", action="store_true",
                        help="Run pre-flight checks only, don't start bot")
    parser.add_argument("--list", action="store_true",
                        help="List available trained combos and exit")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = check_config(args.config)
    if cfg is None:
        sys.exit(1)

    if args.list:
        list_available_combos(cfg)
        return

    ok = run_preflight(cfg, args.combo, args.mock)

    if args.check:
        sys.exit(0 if ok else 1)

    if not ok:
        logger.error("Fix pre-flight issues before starting. Use --check to re-test.")
        sys.exit(1)

    # Hand off to orchestrator
    logger.info("\nStarting bot...\n")

    from orchestrator import BotOrchestrator, load_config
    full_cfg = load_config(args.config)
    bot = BotOrchestrator(full_cfg, mock=args.mock)
    try:
        bot.start(args.combo)
    except Exception as e:
        logger.exception(f"Bot crashed: {e}")
        bot.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
