"""Render evaluator plan views for retained V41 timing repeats."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import owner_joint_hazard_timing_repeats as repeats
import render_owner_joint_hazard_response as base


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report_path = repeats.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = repeats.trial.v40.v38.original()["summary"]
    videos = {}
    for item in report["results"]:
        key = f"{item['prop_case']}-repeat-{item['repeat']}"
        root = repeats.OUT / item["prop_case"] / f"repeat-{item['repeat']}"
        videos[key] = base.draw(item["prop_case"], item, report, summary,
                                output_root=root)
    result = {"schema": "robot-learning-lab.ape-owner-joint-hazard-repeat-videos.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(Path(__file__)),
              "base_renderer_sha256": sha(base.__file__),
              "overhead_videos": videos}
    (repeats.OUT / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
