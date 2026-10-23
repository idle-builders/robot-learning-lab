"""Show unchanged-scan V46 versus scan-history V48 on one 20 ms timeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import owner_fused_height_history as trial
import render_owner_low_prop_scan_planes as v46_render


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    earlier = json.loads((trial.v46.OUT / "report.json").read_text())
    current = json.loads((trial.OUT / "report.json").read_text())
    summary = trial.v38.original()["summary"]
    case = "parked_after_clear"
    old = next(row for row in earlier["results"] if row["profile"] ==
               "dual_05_30cm" and row["case"] == case)
    new = next(row for row in current["results"] if row["case"] == case)
    items = [{**old, "profile": "V46 current-scan choice"},
             {**new, "profile": "V48 scan-history choice"}]
    times = np.arange(min(x["run"]["resume_time_s"] for x in items),
                      max(x["run"]["last_time_s"] for x in items)+.0001, .1)
    video = trial.OUT / "comparisons" / "low-prop-current-vs-history.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            writer.append_data(np.concatenate([
                v46_render.panel(item, float(t), summary,
                                 current["prop_motion"]) for item in items],
                axis=1))
    result = {"schema": "robot-learning-lab.ape-owner-fused-history-video.v1",
              "v46_report_sha256": sha(trial.v46.OUT / "report.json"),
              "v48_report_sha256": sha(trial.OUT / "report.json"),
              "source_sha256": sha(__file__),
              "video": {"path": str(video), "sha256": sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
