"""V196p: predeclared new furniture/path challenge for the V194 cue.

Two table placements and a shifted faster prop path passed only evaluator
geometry preflight before this sensor run. Compare the old pooled derivative
and V194 static-reference cue without changing either detector.
"""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_static_reference_owner_v194p as v194p
import run_standoff_front_response_v150p as v150p
import run_standoff_scene_variants_v160p as v160p
import run_varied_furniture_owner_v191p as v191p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-heldout-furniture"
NEW_PATH = {"approach_start_s": 17.15,
            "approach_end_s": 17.75,
            "end_xy_m": (-.25, .12),
            "prop_half_size_xyz_m": (.045, .065, .04)}
LAYOUTS = {
    "left_mid": {"table_xyz_m": (-.30, .57, 0.),
                 "prop_variant": "fast_shifted"},
    "right_backward": {"table_xyz_m": (-.48, -.46, 0.),
                       "prop_variant": "fast_shifted"},
}


def run_condition(monitor: str, stationary: bool) -> dict:
    name = f"{monitor}-{'stationary' if stationary else 'moving'}"
    old = (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S,
           v150p.FrontMonitor.capture)
    try:
        v191p.OUT = OUT / name
        variant = "stationary_shifted" if stationary else "fast_shifted"
        v191p.LAYOUTS = {key: {**value, "prop_variant": variant}
                         for key, value in LAYOUTS.items()}
        v191p.RESPONSES_S = (None,) if stationary else (.2, None)
        if monitor == "static_reference":
            v150p.FrontMonitor.capture = v194p.static_reference_capture
        v191p.main()
        path = v191p.OUT / "report.json"
        report = json.loads(path.read_text())
        report.update({
            "schema": "robot-learning-lab.ape-heldout-furniture.v1",
            "heldout_source_sha256": sha(SOURCE),
            "monitor_rule": monitor,
            "stationary_prop_control": stationary,
            "path_variant": v160p.VARIANTS[variant],
            "scope": "Two new rigid-table layouts and a faster shifted low "
                     "prop path were chosen and collision-preflighted before "
                     "sensor outcomes. Matched frozen V174 pooled range or "
                     "V194 static-reference range warning plus the same head "
                     "camera and V153 owner task. Complete historical held "
                     "turns; no selected-body approach, motion response, "
                     "physical sensor or real-time qualification.",
        })
        path.write_text(json.dumps(report, indent=2) + "\n")
        return {"condition": name, "report_path": str(path.relative_to(ROOT)),
                "report_sha256": sha(path), "episodes": len(report["rows"])}
    finally:
        (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S,
         v150p.FrontMonitor.capture) = old


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v160p.VARIANTS["fast_shifted"] = NEW_PATH
    v160p.VARIANTS["stationary_shifted"] = {
        **NEW_PATH, "approach_start_s": 30., "approach_end_s": 31.}
    try:
        conditions = [
            run_condition("pooled_derivative", False),
            run_condition("pooled_derivative", True),
            run_condition("static_reference", False),
            run_condition("static_reference", True),
        ]
    finally:
        del v160p.VARIANTS["fast_shifted"]
        del v160p.VARIANTS["stationary_shifted"]
    report = {"schema": "robot-learning-lab.ape-heldout-furniture-index.v1",
              "source_sha256": sha(SOURCE), "layouts": LAYOUTS,
              "new_path": NEW_PATH, "conditions": conditions}
    (OUT / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "index.json")


if __name__ == "__main__":
    main()
