"""Probe actor-aimed head RGB-D coverage during the west-station range outage."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import run_layout2_adaptive_renewed_hold_dropout_v761 as dropout
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_exit_turn_v749 as turn
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
from screen_robocasa_owner_safe_stop_v494 import aim_camera


SOURCE = Path(__file__)
OUT = dropout.trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-dropout-head-depth-probe-v763"
CAPTURE_INTERVAL_S = .2


class DepthProbeMonitor(dropout.DropoutMonitor):
    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.renderer = None
        self.last_depth_capture_s = None
        self.head_target = None
        self.head_target_track = None
        tracks = grid.multi_history[-1]["tracks"] if grid.multi_history else []
        unknown = [track for track in tracks
                   if np.linalg.norm(np.asarray(track["center_xy_m"]) -
                                     full.FIRST_SURFACE) > .45]
        if unknown:
            selected = min(unknown, key=lambda track:
                           np.linalg.norm(np.asarray(track["center_xy_m"]) -
                                          estimate[:2]))
            self.head_target = np.asarray(selected["center_xy_m"])
            self.head_target_track = selected
            aim_camera(model, "head", estimate[:2], self.head_target, 20)

    def __call__(self, t):
        result = super().__call__(t)
        if self.head_target is None:
            result["head_depth_probe"] = {"status": "no_actor_unknown_track",
                                          "initial_tracks": self.grid.multi_history[-1]["tracks"]}
            return result
        if self.script.capture_time_s is None or t - self.script.capture_time_s > .81:
            return result
        if (self.last_depth_capture_s is not None and
                t - self.last_depth_capture_s < CAPTURE_INTERVAL_S - .001):
            return result
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        rgb, depth, wall = owner.capture(self.renderer, self.data, "head", True)
        self.last_depth_capture_s = t
        folder = OUT / "depth-packets"
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{len(list(folder.glob('*.npy'))):02d}"
        rgb_path = folder / f"{name}.png"
        depth_path = folder / f"{name}.npy"
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth)
        result["head_depth_probe"] = {
            "status": "captured", "capture_time_s": t,
            "last_actor_track": self.head_target_track,
            "target_xy_m": self.head_target.tolist(),
            "capture_wall_s": wall,
            "rgb_path": str(rgb_path), "rgb_sha256": route.sha256(rgb_path),
            "depth_path": str(depth_path), "depth_sha256": route.sha256(depth_path),
            "depth_valid_0p15_to_3m_pixels": int(np.sum(np.isfinite(depth) &
                                                    (depth > .15) & (depth < 3.))),
            "depth_min_m": float(np.nanmin(depth)),
            "depth_median_m": float(np.nanmedian(depth))}
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-dropout-head-depth-probe.v1",
        "control_report_sha256": route.sha256(dropout.OUT / "report.json"),
        "same_range_dropout_until_s": dropout.DROP_UNTIL_AFTER_CAPTURE_S,
        "head_view": "aim separate head RGB-D at nearest non-person actor range track available before item-image capture; retain owner_wide20 social camera",
        "capture_interval_s": CAPTURE_INTERVAL_S,
        "no_depth_based_action": True,
        "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=turn.ExitTurnProp, out=OUT,
                  profile="V761 range outage plus actor-directed head depth coverage probe",
                  monitor_factory=DepthProbeMonitor)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["depth_probe_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
