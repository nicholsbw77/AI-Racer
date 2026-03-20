"""
pit_exit_gui.py

PyQt5 GUI to configure pit exit autopilot parameters.
Saves settings to pit_exit_config.json, which the orchestrator reads at runtime.

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
    QLabel, QDoubleSpinBox, QPushButton, QFrame, QGroupBox,
)

CONFIG_PATH = Path(__file__).parent / "pit_exit_config.json"

DEFAULTS = {
    "straight_duration": 8.0,    # seconds to drive straight after pit exit
    "turn_angle": -60.0,         # degrees (negative = left, positive = right)
    "turn_duration": 1.5,        # seconds to hold the turn
    "turn_throttle": 0.35,       # throttle during turn
    "straight_throttle": 0.40,   # throttle during straight
    "ramp_duration": 3.0,        # seconds to blend from turn to model handoff
    "cruise_until_lap_pct": 0.15, # keep autopilot until this lap %
    "cruise_throttle": 0.50,     # throttle during cruise phase
    "pit_exit_track_pos": 0.60,  # known track offset at pit exit (+ = right of line)
    "stall_pullout_left_dur": 1.2,    # seconds turning left out of stall
    "stall_pullout_right_dur": 0.8,   # seconds turning right to straighten
    "stall_pullout_steer": 0.35,      # steering magnitude for pullout
    "stall_pullout_throttle": 0.35,   # gentle throttle during pullout
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except (json.JSONDecodeError, IOError):
            pass
    return dict(DEFAULTS)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


class PitExitGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pit Exit Autopilot Config")
        self.setFixedWidth(420)

        cfg = load_config()

        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Title
        title = QLabel("Pit Exit Autopilot")
        title.setFont(QFont("", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Stall Pullout Phase ---
        pullout_group = QGroupBox("Stall Pullout (leave pit box)")
        pg = QGridLayout()

        pg.addWidget(QLabel("Left turn duration (sec):"), 0, 0)
        self.pullout_left_dur = QDoubleSpinBox()
        self.pullout_left_dur.setRange(0.0, 5.0)
        self.pullout_left_dur.setSingleStep(0.1)
        self.pullout_left_dur.setDecimals(1)
        self.pullout_left_dur.setValue(cfg.get("stall_pullout_left_dur", 1.2))
        pg.addWidget(self.pullout_left_dur, 0, 1)

        pg.addWidget(QLabel("Right turn duration (sec):"), 1, 0)
        self.pullout_right_dur = QDoubleSpinBox()
        self.pullout_right_dur.setRange(0.0, 5.0)
        self.pullout_right_dur.setSingleStep(0.1)
        self.pullout_right_dur.setDecimals(1)
        self.pullout_right_dur.setValue(cfg.get("stall_pullout_right_dur", 0.8))
        pg.addWidget(self.pullout_right_dur, 1, 1)

        pg.addWidget(QLabel("Steering amount:"), 2, 0)
        self.pullout_steer = QDoubleSpinBox()
        self.pullout_steer.setRange(0.0, 1.0)
        self.pullout_steer.setSingleStep(0.05)
        self.pullout_steer.setDecimals(2)
        self.pullout_steer.setValue(cfg.get("stall_pullout_steer", 0.35))
        pg.addWidget(self.pullout_steer, 2, 1)

        pg.addWidget(QLabel("Pullout throttle:"), 3, 0)
        self.pullout_throttle = QDoubleSpinBox()
        self.pullout_throttle.setRange(0.0, 1.0)
        self.pullout_throttle.setSingleStep(0.05)
        self.pullout_throttle.setDecimals(2)
        self.pullout_throttle.setValue(cfg.get("stall_pullout_throttle", 0.35))
        pg.addWidget(self.pullout_throttle, 3, 1)

        pullout_hint = QLabel("swerve left then right to clear pit stall wall")
        pullout_hint.setStyleSheet("color: gray; font-size: 10px;")
        pg.addWidget(pullout_hint, 4, 0, 1, 3)

        pullout_group.setLayout(pg)
        layout.addWidget(pullout_group)

        # --- Straight Phase ---
        straight_group = QGroupBox("Straight Phase")
        sg = QGridLayout()

        sg.addWidget(QLabel("Time before turn (sec):"), 0, 0)
        self.straight_dur = QDoubleSpinBox()
        self.straight_dur.setRange(0.0, 60.0)
        self.straight_dur.setSingleStep(0.5)
        self.straight_dur.setDecimals(1)
        self.straight_dur.setValue(cfg["straight_duration"])
        sg.addWidget(self.straight_dur, 0, 1)

        sg.addWidget(QLabel("Straight throttle:"), 1, 0)
        self.straight_thr = QDoubleSpinBox()
        self.straight_thr.setRange(0.0, 1.0)
        self.straight_thr.setSingleStep(0.05)
        self.straight_thr.setDecimals(2)
        self.straight_thr.setValue(cfg["straight_throttle"])
        sg.addWidget(self.straight_thr, 1, 1)

        straight_group.setLayout(sg)
        layout.addWidget(straight_group)

        # --- Turn Phase ---
        turn_group = QGroupBox("Turn Phase")
        tg = QGridLayout()

        tg.addWidget(QLabel("Steering angle (deg):"), 0, 0)
        self.turn_angle = QDoubleSpinBox()
        self.turn_angle.setRange(-180.0, 180.0)
        self.turn_angle.setSingleStep(5.0)
        self.turn_angle.setDecimals(1)
        self.turn_angle.setValue(cfg["turn_angle"])
        tg.addWidget(self.turn_angle, 0, 1)
        hint = QLabel("negative = left, positive = right")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        tg.addWidget(hint, 0, 2)

        tg.addWidget(QLabel("Turn duration (sec):"), 1, 0)
        self.turn_dur = QDoubleSpinBox()
        self.turn_dur.setRange(0.1, 10.0)
        self.turn_dur.setSingleStep(0.1)
        self.turn_dur.setDecimals(1)
        self.turn_dur.setValue(cfg["turn_duration"])
        tg.addWidget(self.turn_dur, 1, 1)

        tg.addWidget(QLabel("Turn throttle:"), 2, 0)
        self.turn_thr = QDoubleSpinBox()
        self.turn_thr.setRange(0.0, 1.0)
        self.turn_thr.setSingleStep(0.05)
        self.turn_thr.setDecimals(2)
        self.turn_thr.setValue(cfg["turn_throttle"])
        tg.addWidget(self.turn_thr, 2, 1)

        turn_group.setLayout(tg)
        layout.addWidget(turn_group)

        # --- Ramp / Handoff section ---
        ramp_group = QGroupBox("Model Handoff Ramp")
        rg = QGridLayout()

        rg.addWidget(QLabel("Ramp duration (sec):"), 0, 0)
        self.ramp_dur = QDoubleSpinBox()
        self.ramp_dur.setRange(0.0, 10.0)
        self.ramp_dur.setSingleStep(0.5)
        self.ramp_dur.setDecimals(1)
        self.ramp_dur.setValue(cfg["ramp_duration"])
        rg.addWidget(self.ramp_dur, 0, 1)
        ramp_hint = QLabel("blends from turn to full model control")
        ramp_hint.setStyleSheet("color: gray; font-size: 10px;")
        rg.addWidget(ramp_hint, 0, 2)

        ramp_group.setLayout(rg)
        layout.addWidget(ramp_group)

        # --- Cruise Phase ---
        cruise_group = QGroupBox("Cruise Phase (post-merge)")
        cg = QGridLayout()

        cg.addWidget(QLabel("Cruise until lap %:"), 0, 0)
        self.cruise_pct = QDoubleSpinBox()
        self.cruise_pct.setRange(0.0, 1.0)
        self.cruise_pct.setSingleStep(0.01)
        self.cruise_pct.setDecimals(3)
        self.cruise_pct.setValue(cfg.get("cruise_until_lap_pct", 0.15))
        cg.addWidget(self.cruise_pct, 0, 1)
        cruise_hint = QLabel("autopilot stays active until this lap %")
        cruise_hint.setStyleSheet("color: gray; font-size: 10px;")
        cg.addWidget(cruise_hint, 0, 2)

        cg.addWidget(QLabel("Cruise throttle:"), 1, 0)
        self.cruise_thr = QDoubleSpinBox()
        self.cruise_thr.setRange(0.0, 1.0)
        self.cruise_thr.setSingleStep(0.05)
        self.cruise_thr.setDecimals(2)
        self.cruise_thr.setValue(cfg.get("cruise_throttle", 0.50))
        cg.addWidget(self.cruise_thr, 1, 1)

        cg.addWidget(QLabel("Pit exit track offset:"), 2, 0)
        self.pit_track_pos = QDoubleSpinBox()
        self.pit_track_pos.setRange(-1.0, 1.0)
        self.pit_track_pos.setSingleStep(0.05)
        self.pit_track_pos.setDecimals(2)
        self.pit_track_pos.setValue(cfg.get("pit_exit_track_pos", 0.60))
        cg.addWidget(self.pit_track_pos, 2, 1)
        pos_hint = QLabel("+ = right of racing line at pit exit")
        pos_hint.setStyleSheet("color: gray; font-size: 10px;")
        cg.addWidget(pos_hint, 2, 2)

        cruise_group.setLayout(cg)
        layout.addWidget(cruise_group)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset Defaults")
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)

        layout.addLayout(btn_layout)

        # Status label
        self.status = QLabel("")
        self.status.setStyleSheet("color: green;")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

        self.setLayout(layout)

    def _save(self):
        cfg = {
            "straight_duration": self.straight_dur.value(),
            "turn_angle": self.turn_angle.value(),
            "turn_duration": self.turn_dur.value(),
            "turn_throttle": self.turn_thr.value(),
            "straight_throttle": self.straight_thr.value(),
            "ramp_duration": self.ramp_dur.value(),
            "cruise_until_lap_pct": self.cruise_pct.value(),
            "cruise_throttle": self.cruise_thr.value(),
            "pit_exit_track_pos": self.pit_track_pos.value(),
            "stall_pullout_left_dur": self.pullout_left_dur.value(),
            "stall_pullout_right_dur": self.pullout_right_dur.value(),
            "stall_pullout_steer": self.pullout_steer.value(),
            "stall_pullout_throttle": self.pullout_throttle.value(),
        }
        save_config(cfg)
        self.status.setText(f"Saved to {CONFIG_PATH.name}")
        QTimer.singleShot(3000, lambda: self.status.setText(""))

    def _reset(self):
        self.straight_dur.setValue(DEFAULTS["straight_duration"])
        self.turn_angle.setValue(DEFAULTS["turn_angle"])
        self.turn_dur.setValue(DEFAULTS["turn_duration"])
        self.turn_thr.setValue(DEFAULTS["turn_throttle"])
        self.straight_thr.setValue(DEFAULTS["straight_throttle"])
        self.ramp_dur.setValue(DEFAULTS["ramp_duration"])
        self.cruise_pct.setValue(DEFAULTS["cruise_until_lap_pct"])
        self.cruise_thr.setValue(DEFAULTS["cruise_throttle"])
        self.pit_track_pos.setValue(DEFAULTS["pit_exit_track_pos"])
        self.pullout_left_dur.setValue(DEFAULTS["stall_pullout_left_dur"])
        self.pullout_right_dur.setValue(DEFAULTS["stall_pullout_right_dur"])
        self.pullout_steer.setValue(DEFAULTS["stall_pullout_steer"])
        self.pullout_throttle.setValue(DEFAULTS["stall_pullout_throttle"])
        self.status.setText("Reset to defaults (not saved yet)")
        QTimer.singleShot(3000, lambda: self.status.setText(""))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PitExitGUI()
    window.show()
    sys.exit(app.exec_())
