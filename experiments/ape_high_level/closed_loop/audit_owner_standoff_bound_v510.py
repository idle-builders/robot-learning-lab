"""Independent full-horizon geometry replay of V509 bound challenges."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_person_standoff_v508 as audit
import robocasa_owner_person as person
import robocasa_standoff_turn_v507 as turn
import run_robocasa_person_joined_v497 as joined
import stress_owner_standoff_bound_v509 as trial
from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/ape-robocasa-person-standoff-bound-v509/report.json"
OUT = ROOT / "artifacts/ape-robocasa-person-standoff-bound-v509/audit-v510.json"


def main():
    report = json.loads(REPORT.read_text())
    cases = []
    for episode in report["episodes"]:
        result = audit.audit_episode(episode)
        cases.append(result)
        print(result["response"], result["outcome"],
              result["nearest"]["ape_any"], "clear", result["ape_5cm_margin_pass"],
              flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(REPORT),
              "trial_source_sha_pass": report["source_sha256"] == sha256(Path(trial.__file__)),
              "turn_source_sha_pass": report["standoff_turn_source_sha256"] == sha256(Path(turn.__file__)),
              "joined_source_sha_pass": report["joined_route_source_sha256"] == sha256(Path(joined.__file__)),
              "person_source_sha_pass": report["person_source_sha256"] == sha256(Path(person.__file__)),
              "all_videos_pass": all(row["video_sha_pass"] and row["video_frames_pass"]
                                     for row in cases),
              "scope": "V508 independent 20ms route/standoff/owner/prop geometry replay on V509 60/70cm challenge episodes; body/owner kinematic proxies",
              "cases": cases}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
