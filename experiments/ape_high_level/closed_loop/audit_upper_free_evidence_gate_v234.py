"""V234: necessary upper free-evidence gate on paired initial routes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
from audit_head_short_depth_v225 import FRAME_ROOT as IDEAL_FRAMES
from audit_crate_depth_holes_v232 import FRAME_ROOT as HOLE_FRAMES
from run import ROOT, sha
from run_rgb_floor_target_v213 import floor_entry
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v234"
V203 = ROOT / "artifacts/ape-closed-loop-cove50-v203/fourth-split-geometry-audit.json"
V204 = ROOT / "artifacts/ape-closed-loop-cove50-v204/fourth-split-origin-audit.json"
V225 = ROOT / "artifacts/ape-closed-loop-cove50-v225/head-short-depth-route-audit.json"
V232 = ROOT / "artifacts/ape-closed-loop-cove50-v232/crate-depth-hole-route-audit.json"


class UpperFreeEvidenceClearance(TwoHeightClearance):
    """A zero observed-free upper volume can never authorize an edge.

    Nonzero free samples are only a necessary condition; the usual point-gap
    check still applies and this class does not claim the swept body is clear.
    """

    checks = 0
    zero_free_rejections = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.voxel = args[5]
        self.time_s = args[6]
        self.coverage_cache: dict[tuple[int, int, int], bool] = {}

    def edge_gap(self, start_xy: np.ndarray, end_xy: np.ndarray,
                 yaw: float) -> float:
        measured_gap = super().edge_gap(start_xy, end_xy, yaw)
        if measured_gap < 0:
            return measured_gap
        delta = np.asarray(end_xy) - np.asarray(start_xy)
        for fraction in (.25, .5, .75, 1.):
            xy = np.asarray(start_xy) + fraction * delta
            key = (int(round(xy[0] / .04)), int(round(xy[1] / .04)),
                   int(round(yaw * 100)))
            if key not in self.coverage_cache:
                pose = np.array([xy[0], xy[1], yaw])
                coverage = self.voxel.buffered_upper_coverage(pose, self.time_s)
                self.coverage_cache[key] = coverage["free"] > 0
            type(self).checks += 1
            if not self.coverage_cache[key]:
                type(self).zero_free_rejections += 1
                return -math.inf
        return measured_gap


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(V203.read_text())["rows"]
    entries = json.loads(V204.read_text())["rows"]
    baseline_ideal = {r["seed"]: r["short_depth_route"]
                      for r in json.loads(V225.read_text())["rows"]}
    baseline_hole = {r["seed"]: r["crate_hole_route"]
                     for r in json.loads(V232.read_text())["rows"]}
    old_clearance = route.TwoHeightClearance
    route.TwoHeightClearance = UpperFreeEvidenceClearance
    rows = []
    try:
        for scene in scenes:
            if scene["label"] != "generous_kinematic":
                continue
            seed = scene["seed"]
            positions = scene["audits"]["0.065"]["positions_xy_m"]
            entry = next(r for r in entries if r["seed"] == seed and
                         not r["absent"] and r["light_multiplier"] == 1.)
            variant = {"variant": f"seed{seed}-upper-free-evidence",
                       "cream_cheese_xy_m": positions["CreamCheese"],
                       "parmesan_xy_m": positions["Parmesan"]}
            for label, frames, baseline in (
                    ("ideal_short", IDEAL_FRAMES, baseline_ideal[seed]),
                    ("crate_hole_short", HOLE_FRAMES, baseline_hole[seed])):
                UpperFreeEvidenceClearance.checks = 0
                UpperFreeEvidenceClearance.zero_free_rejections = 0
                selected = route.screen(seed, variant, floor_entry(entry),
                                        frame_root=frames, light_multiplier=1.)
                rows.append({"seed": seed, "observation": label,
                             "baseline_status": baseline["status"],
                             "baseline_kind": baseline.get("selected_kind"),
                             "baseline_min_full_scene_m": (
                                 baseline.get("evaluator_selected_path") or {}).get(
                                     "min_full_scene_m"),
                             "edge_coverage_checks": UpperFreeEvidenceClearance.checks,
                             "zero_free_edge_rejections":
                                 UpperFreeEvidenceClearance.zero_free_rejections,
                             "evidence_gated_route": selected})
                print(seed, label, selected["status"],
                      selected.get("selected_kind"),
                      (selected.get("evaluator_selected_path") or {}).get(
                          "min_full_scene_m"), flush=True)
    finally:
        route.TwoHeightClearance = old_clearance
    result = {"scope": "V234 revealed paired V225 ideal-short and V232 all-crate-depth-hole initial-route screens on V203 rooms, same RGB owner boxes/floor target, virtual 4/8/11cm scans, selected-body point gap and frozen planner. Candidate adds a necessary actor-map evidence condition: reject an edge if any swept sample pose has zero observed-free voxels in the existing 126-sample upper-body volume. Nonzero samples are not a safety proof; evaluator scores path afterward. No live controller, physical sensor or held-out qualification.",
              "source_sha256": sha(SOURCE), "v203_sha256": sha(V203),
              "v204_sha256": sha(V204), "v225_sha256": sha(V225),
              "v232_sha256": sha(V232), "rows": rows}
    (OUT / "upper-free-evidence-route-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
