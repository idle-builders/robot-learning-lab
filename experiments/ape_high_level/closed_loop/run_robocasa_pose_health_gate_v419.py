"""Screen a conservative scan-innovation stop against V417 yaw faults.

This is a development gate chosen *after* observing V417. It must not be
presented as a frozen held-out result or a physical IMU-health estimator.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_localized_route_v408 as actor
from run_robocasa_heading_fault_v417 import biased_scan, original_scan
from run_robocasa_layout2_route_v407 import frame_with_map as layout2_frame
from run_robocasa_layout2_route_v407 import model_with_layout2, sha256
from run_robocasa_layout4_transfer_v412 import frame_layout4, layout4_scene


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-pose-health-gate-v419"
BaseLocalizer = actor.RangeLocalizer
BaseGrid = actor.SensorGrid
HEALTH = {"fault": False, "trigger": None}


class HealthLocalizer(BaseLocalizer):
    def __init__(self, max_range_m: float, *, grow: bool = False) -> None:
        super().__init__(max_range_m, grow=grow)
        self.frames = 0
        HEALTH["fault"] = False
        HEALTH["trigger"] = None

    def correct(self, scan, predicted):
        pose, status = super().correct(scan, predicted)
        if (self.frames >= 10 and status["status"] == "registered" and
                status["correction_m"] > .025 and
                status["median_residual_m"] > .040):
            HEALTH["fault"] = True
            HEALTH["trigger"] = {"frame": self.frames, **status}
            status = {**status, "status": "pose_health_fault"}
        self.frames += 1
        return pose, status


class HealthGrid(BaseGrid):
    def plan(self, origin_xy, goal_xy):
        if HEALTH["fault"]:
            return None, "localization_uncertain_stop", []
        return super().plan(origin_xy, goal_xy)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.RangeLocalizer = HealthLocalizer
    actor.SensorGrid = HealthGrid
    cases = []
    for layout in (2, 4):
        for mode in ("healthy", "fixed_plus5", "alternating_plusminus5"):
            actor.model_with_layout2 = model_with_layout2 if layout == 2 else layout4_scene
            actor.frame_with_map = layout2_frame if layout == 2 else frame_layout4
            actor.planar_scan = original_scan if mode == "healthy" else biased_scan(mode)
            actor.OUT = OUT / f"layout{layout}-{mode}"
            episode = actor.episode(5., .30, "range_growing")
            cases.append({"layout": layout, "mode": mode,
                          "health_trigger": HEALTH["trigger"], "episode": episode})
            print(layout, mode, episode["outcome"], episode["steps"],
                  round(episode["end_pose_error_m"], 3),
                  HEALTH["trigger"]["frame"] if HEALTH["trigger"] else None, flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_actor_source_sha256": sha256(Path(actor.__file__)),
              "gate": {"minimum_prior_frames": 10,
                       "translation_correction_gt_m": .025,
                       "median_scan_residual_gt_m": .040},
              "threshold_origin": "chosen after V417 fault and healthy layout2/4 trace inspection",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
