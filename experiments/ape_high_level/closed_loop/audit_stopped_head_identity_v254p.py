"""Replay the unchanged actor recognizer on the stopped head RGB packets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_direct_structural_entry_v271 import V149, V150
from run import ROOT, Vision, sha
from run_color_grounded_entry_v151 import infer


OUT = ROOT / "artifacts/ape-articulated-depth-route-v252p"


def main() -> None:
    report_path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(report_path.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for episode in report["rows"]:
        trace = json.loads((ROOT / episode["trace_path"]).read_text())
        stopped = trace["recovery_inputs"][0]["continuation"]["stages"][1]
        assert stopped["actor_gate"]["status"] == "blocked"
        for entry in stopped["head_inputs"]:
            path = ROOT / entry["actor_frame_path"]
            assert sha(path) == entry["actor_frame_sha256"]
            with np.load(path) as packet:
                choice = infer(
                    vision, exemplar, owner_hist,
                    Image.fromarray(packet["rgb"]),
                    packet["measured_depth_m"],
                    (packet["camera_origin_estimated_m"],
                     packet["camera_rotation_estimated"]),
                    None, None)
            rows.append({"seed": episode["case"][0],
                         "yaw_rad": entry["yaw_rad"],
                         "actor_frame_path": entry["actor_frame_path"],
                         "actor_frame_sha256": entry["actor_frame_sha256"],
                         "proposals": choice["proposals"],
                         "accepted_box": choice["accepted_box"],
                         "accepted_score": choice["accepted_score"]})
    audit = {"schema": "robot-learning-lab.stopped-head-identity.v1",
             "episode_report_sha256": sha(report_path),
             "recognizer_source_sha256": sha(Path(infer.__code__.co_filename)),
             "exemplar_sha256": sha(V149),
             "owner_color_sha256": sha(V150),
             "rows": rows}
    (OUT / "stopped-head-identity.json").write_text(
        json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"views": len(rows),
                      "accepted": sum(row["accepted_box"] is not None
                                      for row in rows),
                      "proposals": [row["proposals"] for row in rows]},
                     indent=2))


if __name__ == "__main__":
    main()
