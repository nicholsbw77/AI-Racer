"""
pit_exit_gui.py

GUI to configure per-track pit exit autopilot parameters.
Reads/writes JSON files in the pit_exit_configs/ folder.

Usage:
  python pit_exit_gui.py
"""

import json
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
    QComboBox, QMessageBox,
)

CONFIG_DIR = Path(__file__).parent / "pit_exit_configs"
ROOT_CONFIG = Path(__file__).parent / "pit_exit_config.json"

DEFAULTS = {
    "stall_pullout_left_dur":  0.0,
    "stall_pullout_right_dur": 0.0,
    "stall_pullout_steer":     0.35,
    "stall_pullout_throttle":  0.20,
    "straight_duration":       5.0,
    "straight_throttle":       0.20,
    "turn_angle":             -5.0,
    "turn_duration":           1.2,
    "turn_throttle":           0.20,
    "ramp_duration":           0.5,
    "cruise_until_lap_pct":    0.04,
    "cruise_throttle":         0.20,
    "cruise_steering":         0.0,
    "launch_min_speed_ms":    20.0,
}

FIELD_INFO = {
    # (label, hint, min, max, step, decimals)
    "stall_pullout_left_dur":  ("Left-turn duration (sec)", "secs to steer left out of stall", 0.0, 10.0, 0.1, 1),
    "stall_pullout_right_dur": ("Right-turn duration (sec)", "secs to steer right out of stall", 0.0, 10.0, 0.1, 1),
    "stall_pullout_steer":     ("Steer amount (0-1)", "steering magnitude during pullout", 0.0, 1.0, 0.05, 2),
    "stall_pullout_throttle":  ("Pullout throttle", "", 0.0, 1.0, 0.05, 2),
    "straight_duration":       ("Straight duration (sec)", "drive straight down pit lane", 0.0, 60.0, 0.5, 1),
    "straight_throttle":       ("Straight throttle", "", 0.0, 1.0, 0.05, 2),
    "turn_angle":              ("Turn angle (deg)", "negative=left  positive=right", -180.0, 180.0, 5.0, 1),
    "turn_duration":           ("Turn duration (sec)", "", 0.0, 10.0, 0.1, 1),
    "turn_throttle":           ("Turn throttle", "", 0.0, 1.0, 0.05, 2),
    "ramp_duration":           ("Ramp duration (sec)", "smoothly unwind steer back to 0", 0.0, 10.0, 0.5, 1),
    "cruise_until_lap_pct":    ("Cruise until lap %", "0=off-pit-road  0.42=back straight", 0.0, 1.0, 0.01, 3),
    "cruise_throttle":         ("Cruise throttle", "", 0.0, 1.0, 0.05, 2),
    "cruise_steering":         ("Cruise steering", "constant steer during cruise (oval pit roads)", -1.0, 1.0, 0.01, 2),
    "launch_min_speed_ms":     ("Launch min speed (m/s)", "0=skip launch, hand straight to model", 0.0, 100.0, 1.0, 1),
}

GROUPS = [
    ("Stall Pullout  (leave pit box)", [
        "stall_pullout_left_dur", "stall_pullout_right_dur",
        "stall_pullout_steer", "stall_pullout_throttle",
    ]),
    ("Straight Phase  (drive down pit lane)", [
        "straight_duration", "straight_throttle",
    ]),
    ("Turn Phase  (pit lane exit turn)", [
        "turn_angle", "turn_duration", "turn_throttle",
    ]),
    ("Ramp Phase  (unwind steering)", [
        "ramp_duration",
    ]),
    ("Cruise Phase  (after merge onto track)", [
        "cruise_until_lap_pct", "cruise_throttle", "cruise_steering",
    ]),
    ("Model Handoff", [
        "launch_min_speed_ms",
    ]),
]


def list_configs():
    CONFIG_DIR.mkdir(exist_ok=True)
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            saved = json.load(f)
        # strip comment keys
        saved = {k: v for k, v in saved.items() if not k.startswith("_")}
        return {**DEFAULTS, **saved}
    return dict(DEFAULTS)


def save_config(name: str, cfg: dict):
    CONFIG_DIR.mkdir(exist_ok=True)
    path = CONFIG_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    # Keep root pit_exit_config.json in sync with the last-saved config
    with open(ROOT_CONFIG, "w") as f:
        json.dump(cfg, f, indent=2)


class PitExitGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pit Exit Config")
        self.setMinimumWidth(480)

        self._spinboxes: dict = {}
        root = QVBoxLayout()
        root.setSpacing(8)

        # Title
        title = QLabel("Pit Exit Autopilot")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # Config selector row
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Config:"))
        self.combo = QComboBox()
        self._refresh_combo()
        self.combo.currentTextChanged.connect(self._on_combo_changed)
        sel_row.addWidget(self.combo, stretch=1)
        new_btn = QPushButton("New…")
        new_btn.clicked.connect(self._new_config)
        sel_row.addWidget(new_btn)
        root.addLayout(sel_row)

        # Parameter groups
        for group_label, keys in GROUPS:
            box = QGroupBox(group_label)
            grid = QGridLayout()
            grid.setVerticalSpacing(4)
            for row, key in enumerate(keys):
                label_text, hint, lo, hi, step, dec = FIELD_INFO[key]
                grid.addWidget(QLabel(label_text + ":"), row, 0)
                sb = QDoubleSpinBox()
                sb.setRange(lo, hi)
                sb.setSingleStep(step)
                sb.setDecimals(dec)
                sb.setValue(DEFAULTS[key])
                grid.addWidget(sb, row, 1)
                if hint:
                    lbl = QLabel(hint)
                    lbl.setStyleSheet("color: gray; font-size: 10px;")
                    grid.addWidget(lbl, row, 2)
                self._spinboxes[key] = sb
            box.setLayout(grid)
            root.addWidget(box)

        # Buttons
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        root.addLayout(btn_row)

        self.status = QLabel("")
        self.status.setStyleSheet("color: green;")
        self.status.setAlignment(Qt.AlignCenter)
        root.addWidget(self.status)

        self.setLayout(root)

        # Load initial config
        if self.combo.count() > 0:
            self._load_into_ui(self.combo.currentText())

    def _refresh_combo(self):
        self.combo.blockSignals(True)
        current = self.combo.currentText()
        self.combo.clear()
        for name in list_configs():
            self.combo.addItem(name)
        # restore selection if possible
        idx = self.combo.findText(current)
        if idx >= 0:
            self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)

    def _on_combo_changed(self, name: str):
        if name:
            self._load_into_ui(name)

    def _load_into_ui(self, name: str):
        cfg = load_config(name)
        for key, sb in self._spinboxes.items():
            sb.setValue(float(cfg.get(key, DEFAULTS[key])))
        self.status.setText(f"Loaded: {name}")
        QTimer.singleShot(2000, lambda: self.status.setText(""))

    def _save(self):
        name = self.combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, "No config", "Select or create a config first.")
            return
        cfg = {key: sb.value() for key, sb in self._spinboxes.items()}
        save_config(name, cfg)
        self._refresh_combo()
        self.status.setText(f"Saved → pit_exit_configs/{name}.json  +  pit_exit_config.json")
        QTimer.singleShot(3000, lambda: self.status.setText(""))

    def _reset(self):
        for key, sb in self._spinboxes.items():
            sb.setValue(DEFAULTS[key])
        self.status.setText("Reset to defaults (not saved)")
        QTimer.singleShot(3000, lambda: self.status.setText(""))

    def _new_config(self):
        from PyQt5.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "New Config", "Config name (e.g. bmwlmdh_charlotte_2025_oval):"
        )
        if ok and name.strip():
            name = name.strip()
            save_config(name, dict(DEFAULTS))
            self._refresh_combo()
            idx = self.combo.findText(name)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            self._load_into_ui(name)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitExitGUI()
    window.show()
    sys.exit(app.exec_())
