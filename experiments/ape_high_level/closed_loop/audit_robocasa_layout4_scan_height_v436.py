"""Independent layout-4 full-height proxy and video audit for V435."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout4_transfer_v413 import audit_episode
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-scan-height-v435"


def main() -> None:
    trial_path = OUT / "report.json"
    report = json.loads(trial_path.read_text())
    cases = []
    for case in report["cases"]:
        audit = audit_episode(case["episode"])
        cases.append({"case": case["case"],
                      "scan_heights_m": case["scan_heights_m"],
                      "outcome": audit["outcome"],
                      "steps": audit["steps"],
                      "final_pose_error_m": audit["final_pose_error_m"],
                      "clearance": audit["clearance"],
                      "proxy_5cm_margin_pass": audit["proxy_5cm_margin_pass"],
                      "video_sha_pass": audit["video_sha_pass"],
                      "video_frames_pass": audit["video_frames_pass"]})
        print(case["case"], audit["outcome"],
              round(audit["clearance"]["distance_m_capped_at_10cm"], 3),
              audit["clearance"]["geom"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(trial_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_scan_height_v435.py")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                    for case in cases),
              "cases": cases}
    (OUT / "audit-v436.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
