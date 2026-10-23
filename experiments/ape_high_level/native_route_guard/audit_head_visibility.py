"""Check the evaluator-only camera coverage report and retained RGB crops."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

import head_visibility as current
import changed_speed_trial as holdout
import kitchen_native as kitchen
import kitchen_trial as trial


def audit() -> dict:
    report = json.loads((current.OUT / "report.json").read_text())
    challenge = json.loads((holdout.OUT / "report.json").read_text())
    checks = {
        "source": report["source_sha256"] == trial.sha(Path(current.__file__)),
        "scene": report["scene_sha256"] == trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace": report["native_trace_sha256"] == trial.sha(kitchen.TRACE),
        "holdout": report["holdout_report_sha256"] == trial.sha(holdout.OUT / "report.json"),
        "coverage_matrix": len(report["rows"]) == 48 and
            {(r["world"], r["time_s"], r["camera"]) for r in report["rows"]} ==
            {(w, t, c) for w in range(4) for t in current.TIMES_S
             for c in current.CAMERAS},
    }
    rows = {(r["world"], r["time_s"], r["camera"]): r for r in report["rows"]}
    triggers = {(r["world"], r["case"]): r["stop_command_s"]
                for r in challenge["results"] if r["mode"] == "mapped_low_guard"}
    for world in (1, 3):
        checks[f"false_stop:{world}"] = triggers[world, "fast_clear_held"] == .3
        for camera in current.CAMERAS:
            first = rows[world, .22, camera]
            last = rows[world, .30, camera]
            checks[f"visible_clipped:{world}:{camera}"] = (
                first["visible_pixels"] > 0 and first["touches_left_border"] and
                last["visible_pixels"] > 0 and
                first["valid_person_depth_fraction"] == 1.0 and
                last["valid_person_depth_fraction"] == 1.0)
            picture = Path(first["rgb_path"])
            rgb = cv2.imread(str(picture))
            checks[f"rgb:{world}:{camera}"] = (
                first["rgb_sha256"] == trial.sha(picture) and
                rgb is not None and rgb.shape == (240, 320, 3))
    outcome = {"schema": "robot-learning-lab.ape-robocasa-head-coverage-audit.v1",
               "report_sha256": trial.sha(current.OUT / "report.json"),
               "false_stop_camera_pairs_visible_and_left_cropped": 4,
               "checks": checks, "pass": all(checks.values())}
    (current.OUT / "audit.json").write_text(json.dumps(outcome, indent=2)+"\n")
    return outcome


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"pass": result["pass"],
                      "checked": len(result["checks"])}, indent=2))
    if not result["pass"]:
        print({k: v for k, v in result["checks"].items() if not v})
        raise SystemExit(1)
