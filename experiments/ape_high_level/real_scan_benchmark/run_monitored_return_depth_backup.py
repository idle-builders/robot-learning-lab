"""Run the fixed aimed-depth backup inside the live second-entrant worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_item_inference_response_19 as cohort
import run_monitored_return_stress as base
import run_shared_world_return as first


PRECOMMIT = ART / "monitored_return_depth_backup_precommit.json"
REPORT = ART / "monitored_return_depth_backup_report.json"
ARM = "range_outage_depth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    screen = json.loads(base.LANES.read_text())
    original = {c["id"]: c for c in cohort.case_rows()}
    source = json.loads(first.REPORT.read_text())
    first_rows = {r["case"]: r for r in source["rows"]}
    cases = []
    for lane in screen["rows"]:
        if lane["selected_relative_bearing_deg"] is None:
            continue
        case = {**original[lane["case"]],
            "entry_relative_deg": lane["selected_relative_bearing_deg"],
            "first_response_trace": first_rows[lane["case"]]["trace"]}
        cases.append(case)
    frozen = {"schema": "ape.monitored-return-depth-backup-precommit.v1",
        "lane_screen": relative(base.LANES),
        "lane_screen_sha256": sha(base.LANES),
        "comparison_report": relative(base.REPORT),
        "comparison_report_sha256": sha(base.REPORT),
        "runner": relative(Path(base.__file__)),
        "runner_sha256": sha(Path(base.__file__)),
        "cases": cases, "arm": ARM,
        "policy": "Withhold primary low range at 0.25-1.25s. The first healthy low-range packet can mark a changed object within 1.3m and select its bearing. On source loss, invalidate item evidence; if bearing fits ideal +/-90deg head yaw, wait 0.2s, acquire independent 160x90 RGB-D every 0.2s, and alarm after >=20 closer pixels change by >0.08m across consecutive aimed depth captures, current depth <0.9m. Respond opposite the prior measured range bearing with the same ideal 0.1s-delayed 0.25m/s disk. Continue live monitoring through fresh and fixed wide search after scripted departure. Rear bearings remain unsupported.",
        "limits": "14 pre-screened real-capture held poses; hypothetical ideal head yaw/tilt and clean depth, no measured mount, head motion, IMU/joint/contact or physical stop. Worker and monitor render separate instances of one timed scripted prop world."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Depth-backup continuation precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        path = base.OUT / case["id"] / ARM / "episode_trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"case": case["id"], "arm": ARM,
                "trace": relative(path), "outcome": trace["outcome"],
                "second_hazard_gap_m": trace["second_hazard_gap_m"],
                "owner_answer": trace["owner_answer"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = base.run_one(case, {}, ARM)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.monitored-return-depth-backup-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["id"], row["outcome"],
              row["second_hazard_gap_m"], flush=True)


if __name__ == "__main__":
    main()
