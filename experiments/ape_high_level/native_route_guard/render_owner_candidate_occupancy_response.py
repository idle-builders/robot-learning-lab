"""Show V42 candidate-response motion and independently scored gaps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import owner_candidate_occupancy_response as trial
import render_owner_joint_hazard_response as view


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.v41.v40.v38.original()["summary"]
    videos = {item["case"]: view.draw(
        item["case"], item, report, summary, output_root=trial.OUT)
        for item in report["results"] if item["case"] != "parked_after_clear"}
    result = {"schema": "robot-learning-lab.ape-owner-candidate-occupancy-videos.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(Path(__file__)),
              "base_renderer_sha256": sha(view.__file__),
              "overhead_videos": videos}
    (trial.OUT / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
