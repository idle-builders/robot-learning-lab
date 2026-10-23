"""Render source-matched opened automatic-versus-forced V151 routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v151_opened_live_route_ablation as v151,
)


ROOT = v151.ROOT
BASE = v151.BASE / "v151"
SUMMARY = BASE / "source-checked-opened-live-route-summary.json"
MANIFEST = BASE / "opened-live-route-videos.json"
RENDERER = ("experiments.compact_morphology.cove50_policy_transfer."
            "reach_image_goal.render_observable_video_poststep")


def main() -> None:
    plan = json.loads(v151.PLAN.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (summary["source_sha256"]["plan"] != v151.sha(v151.PLAN) or
            summary["source_sha256"]["matrix"] != v151.sha(v151.MATRIX) or
            summary["checked_forced_visit_count"] != 2 or
            len(summary["cases"]) != 2):
        raise ValueError("Complete source-checked V151 routes required")
    sources = [
        ("automatic-live", "v16_live", 0.0, False, False,
         ["finger contact shorter than 1 s"],
         plan["opened_baseline_sources"]["unchanged"]["report"],
         plan["opened_baseline_sources"]["unchanged"]["trace"]),
        *((row["id"].split("-")[-2] + "-" + row["id"].split("-")[-1],
           row["route"], row["contact_s"], row["full_gate_pass"],
           row["margin_1p2_s_met"], row["faults"], row["report"], row["trace"])
          for row in summary["cases"]),
    ]
    videos = []
    for label, route, contact, passed, margin, faults, report_rel, trace_rel in sources:
        report, trace = ROOT / report_rel, ROOT / trace_rel
        if not report.is_file() or not trace.is_file():
            raise ValueError(f"Missing V151 video source: {label}")
        video = report.with_name(report.stem + "-aligned.mp4")
        if not video.is_file():
            temporary = video.with_name(video.stem + ".tmp.mp4")
            temporary.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, "-m", RENDERER,
                                "--report", str(report), "--trace", str(trace),
                                "--video", str(temporary)], cwd=ROOT, check=True)
                os.replace(temporary, video)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        videos.append({"label": label, "route": route, "contact_s": contact,
                       "full_gate_pass": passed, "margin_1p2_s_met": margin,
                       "faults": faults,
                       "path": str(video.relative_to(ROOT)),
                       "video_sha256": v151.sha(video),
                       "report": report_rel, "report_sha256": v151.sha(report),
                       "trace": trace_rel, "trace_sha256": v151.sha(trace)})
    manifest = {
        "schema": "robot-learning-lab.cove50-v151-opened-route-videos.v1",
        "role": "posthoc_prescribed_route_diagnostic_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": v151.sha(v151.PLAN),
                          "summary": v151.sha(SUMMARY),
                          "renderer": v151.sha(Path(__file__).with_name(
                              "render_observable_video_poststep.py")),
                          "selection": v151.sha(Path(__file__))},
        "video_alignment": ("Saved trace qpos precedes each physics step; "
                            "frame i renders qpos[i+1] beside post-step contact."),
        "videos": videos,
    }
    encoded = json.dumps(manifest, indent=2) + "\n"
    if MANIFEST.exists() and MANIFEST.read_text() != encoded:
        raise ValueError("V151 video manifest changed")
    MANIFEST.write_text(encoded)
    print("V151 rendered three source-matched automatic/forced route videos")


if __name__ == "__main__":
    main()
