"""V198p: remove forward-left range returns in the new V196 rooms."""

from __future__ import annotations

import json
from pathlib import Path

from run import ROOT, sha
import run_continuous_owner_guard_v95 as v95
import run_furniture_sensor_ablation_v192p as v192p
import run_heldout_furniture_v196p as v196p
import run_static_reference_owner_v194p as v194p
import run_standoff_front_response_v150p as v150p
import run_standoff_scene_variants_v160p as v160p
import run_varied_furniture_owner_v191p as v191p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-heldout-furniture-occlusion"


def run_condition(stationary: bool) -> dict:
    name = "stationary" if stationary else "moving"
    old = (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S,
           v150p.FrontMonitor.capture, v95.front_scan)
    try:
        v191p.OUT = OUT / name
        variant = "stationary_shifted" if stationary else "fast_shifted"
        v191p.LAYOUTS = {key: {**value, "prop_variant": variant}
                         for key, value in v196p.LAYOUTS.items()}
        v191p.RESPONSES_S = (None,) if stationary else (.2, None)
        v150p.FrontMonitor.capture = v194p.static_reference_capture
        v95.front_scan = v192p.occluded_front_scan
        v191p.main()
        path = v191p.OUT / "report.json"
        report = json.loads(path.read_text())
        report.update({
            "schema": "robot-learning-lab.ape-heldout-occlusion.v1",
            "source_run_sha256": sha(SOURCE),
            "new_scene_source_sha256": sha(Path(v196p.__file__)),
            "static_rule_source_sha256": sha(Path(v194p.__file__)),
            "range_occlusion_source_sha256": sha(Path(v192p.__file__)),
            "stationary_prop_control": stationary,
            "injected_forward_pair_front_left_roi_occlusion": True,
            "scope": "Same new V196 rigid furniture and shifted prop path, "
                     "with the frozen V194 static-reference range cue, V174 "
                     "head-depth cue and V153 owner loop. Remove forward "
                     "pair front-left ROI returns from every 10Hz range "
                     "packet, retain angled returns. Historical held pose "
                     "not actor-admitted; scripted prop stop only.",
        })
        path.write_text(json.dumps(report, indent=2) + "\n")
        return {"condition": name, "path": str(path.relative_to(ROOT)),
                "sha256": sha(path), "episodes": len(report["rows"])}
    finally:
        (v191p.OUT, v191p.LAYOUTS, v191p.RESPONSES_S,
         v150p.FrontMonitor.capture, v95.front_scan) = old


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v160p.VARIANTS["fast_shifted"] = v196p.NEW_PATH
    v160p.VARIANTS["stationary_shifted"] = {
        **v196p.NEW_PATH, "approach_start_s": 30.,
        "approach_end_s": 31.}
    try:
        conditions = [run_condition(False), run_condition(True)]
    finally:
        del v160p.VARIANTS["fast_shifted"]
        del v160p.VARIANTS["stationary_shifted"]
    index = {"schema": "robot-learning-lab.ape-heldout-occlusion-index.v1",
             "source_sha256": sha(SOURCE),
             "layouts": v196p.LAYOUTS,
             "new_path": v196p.NEW_PATH,
             "conditions": conditions}
    (OUT / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(OUT / "index.json")


if __name__ == "__main__":
    main()
