"""
pit_exit_gui.py

Tkinter GUI for configuring pit exit sequences per track/car combo.
Saves JSON configs to pit_exit_configs/ that the orchestrator reads
to drive the car out of the pits through a series of timed phases:

  STALL_PULLOUT_LEFT -> STALL_PULLOUT_RIGHT -> STRAIGHT -> TURN -> RAMP -> CRUISE -> HANDOFF

Usage:
  python pit_exit_gui.py
  python pit_exit_gui.py --combo summit_point_mx5_cup
"""

import json
import math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path


CONFIG_DIR = Path(__file__).parent / "pit_exit_configs"

DEFAULT_CONFIG = {
    "straight_duration": 5.0,
    "turn_angle": -5.0,
    "turn_duration": 1.2,
    "turn_throttle": 0.2,
    "straight_throttle": 0.2,
    "ramp_duration": 0.5,
    "cruise_until_lap_pct": 0.04,
    "cruise_throttle": 0.2,
    "pit_exit_track_pos": 0.0,
    "stall_pullout_left_dur": 0.0,
    "stall_pullout_right_dur": 0.0,
    "stall_pullout_steer": 0.35,
    "stall_pullout_throttle": 0.2,
}

# Slider definitions: (key, label, min, max, resolution, group)
SLIDER_DEFS = [
    # Stall Pullout
    ("stall_pullout_left_dur",  "Left Duration (s)",    0.0, 10.0, 0.1,  "Stall Pullout"),
    ("stall_pullout_right_dur", "Right Duration (s)",   0.0, 10.0, 0.1,  "Stall Pullout"),
    ("stall_pullout_steer",     "Steer Amount",         0.0, 1.0,  0.01, "Stall Pullout"),
    ("stall_pullout_throttle",  "Throttle",             0.0, 1.0,  0.01, "Stall Pullout"),
    # Straight
    ("straight_duration",       "Duration (s)",         0.0, 20.0, 0.1,  "Straight"),
    ("straight_throttle",       "Throttle",             0.0, 1.0,  0.01, "Straight"),
    # Turn
    ("turn_angle",              "Angle (degrees)",     -30.0, 30.0, 0.5, "Turn"),
    ("turn_duration",           "Duration (s)",         0.0, 10.0, 0.1,  "Turn"),
    ("turn_throttle",           "Throttle",             0.0, 1.0,  0.01, "Turn"),
    # Ramp
    ("ramp_duration",           "Duration (s)",         0.0,  5.0, 0.1,  "Ramp"),
    # Cruise
    ("cruise_until_lap_pct",    "Until Lap %",          0.0,  0.20, 0.005, "Cruise"),
    ("cruise_throttle",         "Throttle",             0.0,  1.0,  0.01,  "Cruise"),
    # Track position
    ("pit_exit_track_pos",      "Track Position",      -1.0,  1.0,  0.01, "Pit Exit Position"),
]


class PathPreviewCanvas(tk.Canvas):
    """Top-down bird's-eye preview of the pit exit path."""

    PHASE_COLORS = {
        "pullout_left":  "#FF8C00",  # orange
        "pullout_right": "#FF6347",  # tomato
        "straight":      "#4169E1",  # royal blue
        "turn":          "#DC143C",  # crimson
        "ramp":          "#FFD700",  # gold
        "cruise":        "#32CD32",  # lime green
    }

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("bg", "#1a1a2e")
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)

    def draw_path(self, config: dict):
        """Simulate and draw the pit exit path from config."""
        self.delete("all")

        w = self.winfo_width() or 400
        h = self.winfo_height() or 600

        # Draw title
        self.create_text(w // 2, 15, text="Pit Exit Path Preview",
                         fill="#888", font=("Helvetica", 10))

        # Simulate path with simple kinematics
        # Start at bottom-center, heading up
        x, y = w * 0.3, h - 40
        heading = -math.pi / 2  # pointing up
        speed = 0.0  # m/s
        dt = 0.05  # simulation timestep
        scale = 8.0  # pixels per meter

        segments = []  # list of (points_list, color, label)

        def sim_phase(duration, throttle, steer_deg, color, label, end_condition=None):
            nonlocal x, y, heading, speed
            points = [(x, y)]
            t = 0.0
            while t < duration or (end_condition and not end_condition(t)):
                # Simple acceleration model
                accel = throttle * 8.0 - 0.5  # rough approximation
                speed = max(0.0, speed + accel * dt)
                speed = min(speed, 20.0)  # cap at ~72 km/h

                # Steering: convert degrees to turning rate
                steer_rad = math.radians(steer_deg) * 0.15
                heading += steer_rad * dt * max(speed, 1.0) * 0.3

                dx = math.cos(heading) * speed * dt * scale
                dy = math.sin(heading) * speed * dt * scale
                x += dx
                y += dy
                points.append((x, y))
                t += dt

                if t > 30.0:  # safety cap
                    break
            if len(points) > 1:
                segments.append((points, color, label))

        # Phase 1: Stall pullout left
        left_dur = config.get("stall_pullout_left_dur", 0.0)
        if left_dur > 0:
            steer = -abs(config.get("stall_pullout_steer", 0.35)) * 30.0
            thr = config.get("stall_pullout_throttle", 0.2)
            sim_phase(left_dur, thr, steer, self.PHASE_COLORS["pullout_left"],
                      "Pullout L")

        # Phase 2: Stall pullout right
        right_dur = config.get("stall_pullout_right_dur", 0.0)
        if right_dur > 0:
            steer = abs(config.get("stall_pullout_steer", 0.35)) * 30.0
            thr = config.get("stall_pullout_throttle", 0.2)
            sim_phase(right_dur, thr, steer, self.PHASE_COLORS["pullout_right"],
                      "Pullout R")

        # Phase 3: Straight
        straight_dur = config.get("straight_duration", 5.0)
        straight_thr = config.get("straight_throttle", 0.2)
        sim_phase(straight_dur, straight_thr, 0.0,
                  self.PHASE_COLORS["straight"], "Straight")

        # Phase 4: Turn
        turn_dur = config.get("turn_duration", 1.2)
        turn_angle = config.get("turn_angle", -5.0)
        turn_thr = config.get("turn_throttle", 0.2)
        sim_phase(turn_dur, turn_thr, turn_angle,
                  self.PHASE_COLORS["turn"], "Turn")

        # Phase 5: Ramp (linearly reduce turn angle to 0)
        ramp_dur = config.get("ramp_duration", 0.5)
        if ramp_dur > 0:
            # Simulate ramp as half the turn angle on average
            sim_phase(ramp_dur, turn_thr, turn_angle * 0.5,
                      self.PHASE_COLORS["ramp"], "Ramp")

        # Phase 6: Cruise
        cruise_thr = config.get("cruise_throttle", 0.2)
        cruise_pct = config.get("cruise_until_lap_pct", 0.04)
        # Simulate cruise for a fixed visual duration
        cruise_visual_dur = max(1.0, cruise_pct * 50.0)
        sim_phase(cruise_visual_dur, cruise_thr, 0.0,
                  self.PHASE_COLORS["cruise"], "Cruise")

        # Draw segments
        for points, color, label in segments:
            if len(points) < 2:
                continue
            flat = []
            for px, py in points:
                flat.extend([px, py])
            self.create_line(*flat, fill=color, width=3, smooth=True)

            # Label at midpoint
            mid = len(points) // 2
            mx, my = points[mid]
            self.create_text(mx + 12, my, text=label, fill=color,
                             font=("Helvetica", 8), anchor="w")

        # Draw start marker
        if segments:
            sx, sy = segments[0][0][0]
            self.create_oval(sx - 5, sy - 5, sx + 5, sy + 5,
                             fill="#00FF00", outline="")
            self.create_text(sx, sy + 12, text="PIT", fill="#00FF00",
                             font=("Helvetica", 8, "bold"))

        # Draw end marker
        if segments:
            last_pts = segments[-1][0]
            ex, ey = last_pts[-1]
            self.create_oval(ex - 5, ey - 5, ex + 5, ey + 5,
                             fill="#FF4444", outline="")
            self.create_text(ex, ey + 12, text="HANDOFF", fill="#FF4444",
                             font=("Helvetica", 8, "bold"))

        # Legend
        ly = h - 20
        lx = 10
        for phase_name, color in self.PHASE_COLORS.items():
            self.create_rectangle(lx, ly - 4, lx + 12, ly + 4, fill=color, outline="")
            self.create_text(lx + 16, ly, text=phase_name.replace("_", " ").title(),
                             fill="#aaa", font=("Helvetica", 7), anchor="w")
            lx += 80


class PitExitGUI:
    """Main GUI application."""

    def __init__(self, initial_combo: str = None):
        self.root = tk.Tk()
        self.root.title("Pit Exit Configuration")
        self.root.geometry("900x700")
        self.root.configure(bg="#2d2d3d")

        CONFIG_DIR.mkdir(exist_ok=True)

        self.vars = {}  # key -> tk.DoubleVar
        self.current_combo = tk.StringVar()

        self._build_ui()

        # Load initial combo
        if initial_combo:
            self.current_combo.set(initial_combo)
            self._load_config()
        else:
            configs = self._list_configs()
            if configs:
                self.current_combo.set(configs[0])
                self._load_config()
            else:
                self._set_defaults()

        self._update_preview()

    def _list_configs(self) -> list:
        """List available config combos."""
        return sorted(
            p.stem for p in CONFIG_DIR.glob("*.json")
        )

    def _build_ui(self):
        # Top bar: combo selector + save/load
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=5)

        ttk.Label(top, text="Track/Car Combo:").pack(side="left", padx=(0, 5))

        self.combo_box = ttk.Combobox(
            top, textvariable=self.current_combo,
            values=self._list_configs(), width=40
        )
        self.combo_box.pack(side="left", padx=5)

        ttk.Button(top, text="Load", command=self._load_config).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self._save_config).pack(side="left", padx=2)
        ttk.Button(top, text="Save As...", command=self._save_as).pack(side="left", padx=2)
        ttk.Button(top, text="Defaults", command=self._set_defaults).pack(side="left", padx=2)

        # Main area: sliders left, preview right
        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=5)

        # Left: scrollable sliders
        left_frame = ttk.Frame(main)
        main.add(left_frame, weight=1)

        canvas_frame = tk.Canvas(left_frame, bg="#2d2d3d", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas_frame.yview)
        scrollable = ttk.Frame(canvas_frame)

        scrollable.bind(
            "<Configure>",
            lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        )
        canvas_frame.create_window((0, 0), window=scrollable, anchor="nw")
        canvas_frame.configure(yscrollcommand=scrollbar.set)

        canvas_frame.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build sliders grouped by section
        current_group = None
        for key, label, vmin, vmax, res, group in SLIDER_DEFS:
            if group != current_group:
                current_group = group
                grp_frame = ttk.LabelFrame(scrollable, text=group, padding=5)
                grp_frame.pack(fill="x", padx=5, pady=3)

            row = ttk.Frame(grp_frame)
            row.pack(fill="x", pady=1)

            var = tk.DoubleVar(value=DEFAULT_CONFIG.get(key, 0.0))
            self.vars[key] = var

            ttk.Label(row, text=label, width=20).pack(side="left")

            value_label = ttk.Label(row, text=f"{var.get():.3f}", width=8)
            value_label.pack(side="right")

            scale = ttk.Scale(
                row, from_=vmin, to=vmax, variable=var,
                orient="horizontal",
                command=lambda val, vl=value_label, v=var, r=res: (
                    self._snap_and_update(v, float(val), r, vl)
                ),
            )
            scale.pack(side="left", fill="x", expand=True, padx=5)

        # Right: path preview
        self.preview = PathPreviewCanvas(main, width=400, height=600)
        main.add(self.preview, weight=1)

        # Status bar
        self.status = ttk.Label(self.root, text="Ready", relief="sunken")
        self.status.pack(fill="x", padx=10, pady=(0, 5))

        # Keybindings
        self.root.bind("<Control-s>", lambda e: self._save_config())

    def _snap_and_update(self, var, val, resolution, value_label):
        """Snap value to resolution and update preview."""
        snapped = round(val / resolution) * resolution
        var.set(snapped)
        value_label.configure(text=f"{snapped:.3f}")
        self._update_preview()

    def _get_config(self) -> dict:
        """Get current config from slider values."""
        return {key: self.vars[key].get() for key in self.vars}

    def _set_config(self, config: dict):
        """Set slider values from config dict."""
        for key, var in self.vars.items():
            if key in config:
                var.set(config[key])
        self._update_preview()

    def _set_defaults(self):
        """Reset all sliders to defaults."""
        self._set_config(DEFAULT_CONFIG)
        self.status.configure(text="Reset to defaults")

    def _load_config(self):
        """Load config from JSON file."""
        combo = self.current_combo.get().strip()
        if not combo:
            messagebox.showwarning("No Combo", "Enter a track/car combo name.")
            return

        path = CONFIG_DIR / f"{combo}.json"
        if not path.exists():
            messagebox.showinfo("Not Found", f"No config for '{combo}'.\nStarting with defaults.")
            self._set_defaults()
            return

        with open(path) as f:
            config = json.load(f)

        # Merge with defaults for any missing keys
        merged = {**DEFAULT_CONFIG, **config}
        self._set_config(merged)
        self.status.configure(text=f"Loaded: {path.name}")

    def _save_config(self):
        """Save current config to JSON."""
        combo = self.current_combo.get().strip()
        if not combo:
            messagebox.showwarning("No Combo", "Enter a track/car combo name.")
            return

        path = CONFIG_DIR / f"{combo}.json"
        config = self._get_config()

        with open(path, "w") as f:
            json.dump(config, f, indent=2)

        # Refresh combo list
        self.combo_box.configure(values=self._list_configs())
        self.status.configure(text=f"Saved: {path.name}")

    def _save_as(self):
        """Save config with a new combo name."""
        name = tk.simpledialog.askstring(
            "Save As", "Enter combo name (e.g., summit_point_mx5_cup):",
            parent=self.root
        )
        if name:
            self.current_combo.set(name.strip())
            self._save_config()

    def _update_preview(self):
        """Redraw the path preview canvas."""
        config = self._get_config()
        self.preview.draw_path(config)

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pit Exit Configuration GUI")
    parser.add_argument("--combo", default=None,
                        help="Track/car combo to load (e.g., summit_point_mx5_cup)")
    args = parser.parse_args()

    gui = PitExitGUI(initial_combo=args.combo)
    gui.run()


if __name__ == "__main__":
    main()
