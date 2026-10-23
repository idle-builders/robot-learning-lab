"""Full-horizon independent geometry replay of V520 changed hand poses."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_person_depth_standoff_v518 as audit
import robocasa_depth_standoff_turn_v517 as turn
import robocasa_owner_person as person
import run_robocasa_hand_pose_joined_v520 as trial
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "artifacts/ape-robocasa-hand-pose-joined-v520/report.json"
OUT = ROOT / "artifacts/ape-robocasa-hand-pose-joined-v520/audit-v521.json"


def main():
    report = json.loads(TRIAL.read_text())
    cases = []
    for ep in report["episodes"]:
        result = audit.audit_episode(ep)
        cases.append(result)
        print(result["response"], result["outcome"],
              result["nearest"]["ape_any"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(TRIAL),
              "trial_source_sha_pass": report["source_sha256"] == sha256(Path(trial.__file__)),
              "turn_source_sha_pass": report["depth_standoff_turn_source_sha256"] == sha256(Path(turn.__file__)),
              "depth_change_source_sha_pass": report["depth_change_source_sha256"] == sha256(
                  Path(turn.changed_points.__code__.co_filename)),
              "joined_source_sha_pass": report["joined_route_source_sha256"] == sha256(Path(joined.__file__)),
              "person_source_sha_pass": report["person_source_sha256"] == sha256(Path(person.__file__)),
              "all_videos_pass": all(row["video_sha_pass"] and row["video_frames_pass"]
                                     and row["depth_packet_sha_pass"] for row in cases),
              "scope": "independent V518 20ms full-horizon route/arm/standoff/owner/late-prop replay of the V520 changed-hand episodes; kinematic ape and owner surrogates",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
