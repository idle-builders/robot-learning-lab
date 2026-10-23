"""Fixed wider-look recovery after a shared-world interruption returns uncertain."""

from __future__ import annotations

import argparse
import json

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_real_capture_conditional_search as conditional
import run_shared_world_return as first


OUT = ROOT / "assets/real_indoor_scenes/shared_world_wide_recovery_20261015"
PRECOMMIT = ART / "shared_world_wide_recovery_precommit.json"
REPORT = ART / "shared_world_wide_recovery_report.json"
OFFSETS = (90., -90., 135., -135., 180.)


def run_one(case: dict, first_row: dict) -> dict:
    earlier = json.loads((ROOT / first_row["trace"]).read_text())
    if earlier["outcome"] == "fresh_owner_confirmed":
        return {"case": case["id"], "first_outcome": earlier["outcome"],
            "recovery": "not_needed", "final_outcome": "owner_confirmed"}
    if earlier["outcome"] == "clearance_unavailable":
        return {"case": case["id"], "first_outcome": earlier["outcome"],
            "recovery": "clearance_gate", "final_outcome": "hold_unknown"}
    state = first.cohort.generalized_setup(case)
    endpoint = np.asarray(earlier["response_endpoint_xz_m"], float)
    start = state["actual_xy"]
    actor = np.asarray(state["stop"]["actor_pose_xy_m"], float) + endpoint - start
    # Copy the existing conditional-search escalation: a broader proposal
    # source, with the same taught exemplar and fixed acceptance thresholds.
    state["vision"].search_mode = "expanded"
    scene = state["worker_scene"]
    scene.set_props(state["base_props"])
    folder = OUT / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    original = conditional.baseline.SEARCH_YAW_OFFSETS_DEG
    try:
        conditional.baseline.SEARCH_YAW_OFFSETS_DEG = OFFSETS
        result = first.search(state, case, scene, endpoint, actor,
            folder / "wide_search",
            float(state["stop"]["decision_time_s"] + .4 +
                  earlier["worker_wall_s"] + 1.))
    finally:
        conditional.baseline.SEARCH_YAW_OFFSETS_DEG = original
    owner = ("yes" if result["action"] ==
             "candidate_to_owner_for_confirmation" and not
             result["wrong_confident_candidate"] else
             "no" if result["action"] ==
             "candidate_to_owner_for_confirmation" else None)
    outcome = "owner_confirmed" if owner == "yes" else "hold_unknown"
    trace = {"schema": "ape.shared-world-wide-recovery.v1",
        "case": case["id"], "source_episode": first_row["trace"],
        "first_outcome": earlier["outcome"],
        "recovery_offsets_deg": OFFSETS,
        "proposal_mode": "expanded",
        "post_response_endpoint_xz_m": endpoint.tolist(),
        "wide_search_trace": relative(folder / "wide_search/trace.json"),
        "wide_search_action": result["action"],
        "wide_target_visible": result["target_visible_in_any_view"],
        "wrong_confident_candidate": result["wrong_confident_candidate"],
        "scripted_owner_answer": owner, "final_outcome": outcome,
        "limits": "This is a deterministic continuation replay from the saved response endpoint after the hazard's scripted departure, not a second live hazard trial. Five fixed extra headings are tried for every failed/uncertain first return. No object truth chooses a heading or candidate; evaluator overlap generates only owner yes/no."}
    path = folder / "recovery_trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "first_outcome": earlier["outcome"],
        "recovery": relative(path), "final_outcome": outcome,
        "wide_target_visible": result["target_visible_in_any_view"],
        "scripted_owner_answer": owner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = first.cohort.case_rows()
    source = json.loads(first.REPORT.read_text())
    if source["processed"] != len(cases):
        raise ValueError("First-stage room report is incomplete")
    by_case = {x["case"]: x for x in source["rows"]}
    frozen = {"schema": "ape.shared-world-wide-recovery-precommit.v1",
        "source_report": relative(first.REPORT),
        "source_report_sha256": sha(first.REPORT),
        "cases": [x["id"] for x in cases],
        "trigger": ["owner_rejected_candidate", "fresh_item_unidentified"],
        "offsets_deg": OFFSETS,
        "policy": "After fresh search fails or owner says no, look at five fixed additional headings from the reached endpoint using the existing expanded proposal source and unchanged taught multi-cue thresholds; ask owner about a fresh candidate or keep item unknown."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Wide recovery precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        path = OUT / case["id"] / "recovery_trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"case": case["id"],
                "first_outcome": trace["first_outcome"],
                "recovery": relative(path),
                "final_outcome": trace["final_outcome"],
                "wide_target_visible": trace["wide_target_visible"],
                "scripted_owner_answer": trace["scripted_owner_answer"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case, by_case[case["id"]])
            if row["recovery"] not in ("not_needed", "clearance_gate"):
                launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.shared-world-wide-recovery-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["id"], row["first_outcome"], row["final_outcome"],
              flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
