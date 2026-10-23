"""Independent post-stop audit of the frozen V465 late-crossing failure."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-late-policy-v469"
BASE = ROOT / "artifacts/ape-robocasa-layout4-crossing-v447/report.json"


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    base = json.loads(BASE.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    case = common_audit.audit_case(report["cases"][0])
    original = next(c for c in base["cases"] if c["scenario"]["id"] == "late_crossing")
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_late_policy_v469.py")),
              "policy_source_sha_pass": report["policy_source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_directional_wait_v465.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_crossing_v447.json")),
              "prior_map_sha_pass": report["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "same_initial_scene_pass": report["cases"][0]["episode"]["scene_fingerprint"] ==
                                         original["episode"]["scene_fingerprint"],
              "video_pass": case["video_sha_pass"] and case["video_frames_pass"],
              "prop_path_nonoverlap_pass": case["prop_fixture_nonoverlap_pass"],
              "case": case}
    (OUT / "audit-v470.json").write_text(json.dumps(output, indent=2) + "\n")
    print(case["outcome"], "gap", case["nearest_ape_any"]["gap_m_capped_at_10cm"],
          "phase", case["nearest_ape_any"]["phase"],
          "video", output["video_pass"], flush=True)


if __name__ == "__main__":
    main()
