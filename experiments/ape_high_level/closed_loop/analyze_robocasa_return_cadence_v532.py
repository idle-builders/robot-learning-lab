"""Downsample saved return-range packets to expose scan cadence/phase cost."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import sha256


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-return-motion-v531/report.json"
OUT = ROOT / "artifacts/ape-robocasa-return-motion-v531/cadence-v532.json"
THRESHOLD_M = .10


def decision(observations, stride, phase):
    streak = 0
    for row in observations[phase::stride]:
        shifted = row["south_shift_from_first_packet_m"]
        streak = streak + 1 if shifted is not None and shifted >= THRESHOLD_M else 0
        if streak >= 2:
            return row["time_s"]
    return None


def main():
    source = json.loads(SOURCE.read_text())
    rows = []
    for group in source["groups"]:
        for case in group["cases"]:
            clocks = {}
            for stride, hz in ((1, 10), (2, 5), (5, 2)):
                clocks[str(hz)] = [decision(case["observations"], stride, phase)
                                   for phase in range(stride)]
            rows.append({"condition": group["condition"],
                         "response": case["response"], "alarm_s_by_rate_and_phase": clocks})
            print(group["condition"], case["response"], clocks, flush=True)
    OUT.write_text(json.dumps({"source_sha256": sha256(Path(__file__)),
                               "packet_report_sha256": sha256(SOURCE),
                               "scope": "same saved 10Hz ideal low/mid range packets, downsampled to 5/2Hz at all discrete scan phases; no new hardware exposure, dropouts or owner/ape motion",
                               "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
