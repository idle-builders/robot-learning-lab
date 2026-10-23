"""Independent 20 ms full-horizon geometry audit of owner-present route trials.

Reconstructs the furnished scene with the articulated person, then checks the
executed ape poses and the held post-stop pose against every geom as the prop
moves. This evaluator is separate from the range/localization route policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco

import audit_robocasa_dynamic_suite_v440 as common
import robocasa_owner_person as person
from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-person-route-audit-v493"
REPORTS = {
    "single": ROOT / "artifacts/ape-robocasa-person-route-v491/report.json",
    "multi": ROOT / "artifacts/ape-robocasa-person-multitrack-v492/report.json",
}


def evaluator_scene(obstacle_xy):
    model = person.scene_with_person(prop_xy=obstacle_xy)
    person.set_owner_pose(model, [1., -2.4], [1.10, -2.24, .43], show_can=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, model.body("audit_ape").id, model.geom("audit_ape_geom").id


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    common.evaluator_scene = evaluator_scene
    packages = []
    for name, path in REPORTS.items():
        report = json.loads(path.read_text())
        cases = []
        for row in report["cases"]:
            trial = dict(row)
            trial["package"] = name
            result = common.audit_case(trial)
            cases.append(result)
            print(name, result["scenario_id"], result["outcome"],
                  "ape gap", result["nearest_ape_any"],
                  "prop gap", result["nearest_ape_prop"],
                  "prop-fixture", result["prop_fixture_nearest"], flush=True)
        packages.append({"package": name,
                         "report_sha256": sha256(path),
                         "trial_source_sha_pass": report["source_sha256"] == sha256(
                             ROOT / "experiments/ape_high_level/closed_loop" /
                             ("run_robocasa_person_route_v491.py" if name == "single"
                              else "run_robocasa_person_multitrack_v492.py")),
                         "person_source_sha_pass": report["person_source_sha256"] == sha256(
                             Path(person.__file__)),
                         "videos_pass": all(c["video_sha_pass"] and c["video_frames_pass"]
                                            for c in cases),
                         "cases": cases})
    output = {"source_sha256": sha256(Path(__file__)),
              "person_source_sha256": sha256(Path(person.__file__)),
              "scope": "independent full-horizon MuJoCo signed distances for kinematic ape cylinder, scripted person proxy, furniture and moving prop; 20 ms samples plus route captures",
              "packages": packages}
    (OUT / "report.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
