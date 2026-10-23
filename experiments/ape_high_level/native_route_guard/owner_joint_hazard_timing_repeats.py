"""Retain repeat owner continuations to expose response-timing variation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import owner_joint_hazard_response as trial


OUT = trial.OUT / "timing-repeats"
CASES = ("early_entry", "holdout_closer_finish")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert 1 <= args.repeats <= 10
    source = trial.v40.v38.original()
    policy = json.loads(trial.v40.v38.joined.CONFIG.read_text())
    scenario = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    vision = trial.v40.v38.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(
        trial.v40.v38.owner.TEACH_CROP).convert("RGB")])[0]
    checkpoint_s = source["summary"]["range_packets"][-1]["time_s"]
    results = []
    for case in CASES:
        config = trial.PROP_CASES[case]
        obstacle_fn = lambda t, config=config: trial.v40.prop_at(
            t, checkpoint_s, config)
        for repeat in range(args.repeats):
            root = OUT / case / f"repeat-{repeat+1}"
            run = trial.v40.v38.run_case(
                "blind_return_flank_fast_response", source, policy, scenario,
                vision, exemplar,
                flank_probe_factory=lambda video: trial.LocalizedFlankCamera(
                    video),
                flank_response=trial.v40.v38.FLANK_RESPONSE[
                    "blind_return_flank_fast_response"],
                require_camera_pose=True,
                route_obstacle_fn=obstacle_fn,
                response_controller=trial.JointHazardResponse(),
                output_root=root)
            raw = root / "blind_return_flank_fast_response" / \
                "flank-rgbd-watch.npz"
            results.append({"prop_case": case, "repeat": repeat+1,
                            "run": run,
                            "camera_rgbd_evidence": {"path": str(raw),
                                                     "sha256": sha(raw)}})
            rows = json.loads(Path(run["trace"]["path"]).read_text())
            inference = next(row for row in rows if row["stage"] ==
                             "second_inference_result")
            print(case, repeat+1, inference["first_response_s"],
                  run["minimum_proxy_gap_m"],
                  run["minimum_prop_gap_m"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"schema": "robot-learning-lab.ape-owner-joint-hazard-timing-repeats.v1",
              "task": "T-9v3p3n", "source_sha256": sha(Path(__file__)),
              "v41_source_sha256": sha(trial.__file__),
              "v38_source_sha256": sha(trial.v40.v38.__file__),
              "scene_sha256": sha(trial.v40.v38.v35.moving.SCENE_XML),
              "v36_report_sha256": sha(trial.v40.v38.SOURCE),
              "current_design_pointer_sha256": sha(
                  trial.ROOT / "robot/ape-current-design.json"),
              "checkpoint_s": checkpoint_s,
              "cases": {case: trial.PROP_CASES[case] for case in CASES},
              "repeat_count": args.repeats,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
