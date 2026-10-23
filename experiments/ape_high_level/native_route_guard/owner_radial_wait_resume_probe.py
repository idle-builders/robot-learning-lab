"""V84: full owner task with bounded fresh-scan waits on a blocked move.

The evaluator-selected V81 radial destination remains a diagnostic oracle.
The new action behavior is actor-grounded: wait when a preceding range packet
blocks the next 2.5 cm step, rescan, and resume only when that step clears.
"""

from __future__ import annotations

import json

import owner_radial_oracle_pose_probe as v82
import owner_blocked_step_reobservation as v83
import owner_restage_wait_runner as wait_runner


OUT = v82.ROOT / "artifacts/ape-native-route-guard/v84-radial-wait-resume"
MAX_TICKS = 50


def main():
    old_runner = v82.v77.runner
    try:
        v82.v77.runner = wait_runner
        v82.run_suite(
            output_root=OUT, restage_wait_on_block=True,
            max_ticks=MAX_TICKS,
            schema="robot-learning-lab.ape-radial-wait-resume.v1")
    finally:
        v82.v77.runner = old_runner
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["v82_source_sha256"] = report["source_sha256"]
    report["source_sha256"] = v82.v77.v72.v52.sha(__file__)
    report["v83_report_sha256"] = v82.v77.v72.v52.sha(
        v83.OUT / "report.json")
    report["scope"] = (
        "Bounded fresh-scan wait/resume in two complete owner episodes at "
        "V81's evaluator-selected radial destination. The waiting action "
        "uses only prior packets, but the destination remains oracle-tuned. "
        "Historical kinematic proxy, ideal low/mid rays, no selected-body "
        "stop or physical safety qualification.")
    path.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
