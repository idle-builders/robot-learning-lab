"""Audit fixed recovery headings, rendered frames and owner-answer boundary."""

from __future__ import annotations

import json

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative
import audit_shared_world_return as base_audit
import run_shared_world_wide_recovery as trial


OUT = ART / "shared_world_wide_recovery_audit.json"


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    rows = []
    for row in report["rows"]:
        if row["recovery"] == "not_needed":
            if row["first_outcome"] != "fresh_owner_confirmed":
                raise ValueError("Skipped an unresolved return")
            rows.append({"case": row["case"], "first": row["first_outcome"],
                         "final": row["final_outcome"], "recovery": False})
            continue
        if row["recovery"] == "clearance_gate":
            if row["final_outcome"] != "hold_unknown":
                raise ValueError("Crossed an unavailable clearance gate")
            rows.append({"case": row["case"], "first": row["first_outcome"],
                         "final": row["final_outcome"], "recovery": False})
            continue
        trace = json.loads((ROOT / row["recovery"]).read_text())
        source = json.loads((ROOT / trace["source_episode"]).read_text())
        search = json.loads((ROOT / trace["wide_search_trace"]).read_text())
        if trace["first_outcome"] != source["outcome"]:
            raise ValueError("Continuation uses a different episode")
        if not np.allclose(trace["post_response_endpoint_xz_m"],
                           source["response_endpoint_xz_m"]):
            raise ValueError("Continuation moved before fresh observation")
        if tuple(trace["recovery_offsets_deg"]) != trial.OFFSETS:
            raise ValueError("Recovery headings changed")
        if trace["proposal_mode"] != "expanded":
            raise ValueError("Unexpected proposal mode")
        actual_heading = search["observations"][0]["camera_heading_deg"] - trial.OFFSETS[0]
        observed_offsets = tuple(round(x["camera_heading_deg"] - actual_heading, 5)
                                 for x in search["observations"])
        if observed_offsets != trial.OFFSETS:
            raise ValueError("Rendered headings do not match fixed policy")
        if base_audit.decoded_frames(ROOT / search["video"]) != search["frames"]:
            raise ValueError("Recovery video/trace frame mismatch")
        expected_answer = ("yes" if search["action"] ==
            "candidate_to_owner_for_confirmation" and not
            search["wrong_confident_candidate"] else
            "no" if search["action"] ==
            "candidate_to_owner_for_confirmation" else None)
        if trace["scripted_owner_answer"] != expected_answer:
            raise ValueError("Owner answer used wrong evidence")
        expected_outcome = "owner_confirmed" if expected_answer == "yes" else "hold_unknown"
        if trace["final_outcome"] != expected_outcome:
            raise ValueError("Incorrect final policy outcome")
        rows.append({"case": row["case"], "first": row["first_outcome"],
            "final": row["final_outcome"], "recovery": True,
            "target_visible": search["target_visible_in_any_view"],
            "wrong_candidate": search["wrong_confident_candidate"],
            "video_frames": search["frames"]})
    summary = {"schema": "ape.shared-world-wide-recovery-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "recovery_attempts": sum(r["recovery"] for r in rows),
        "confirmed_before_recovery": sum(r["first"] ==
            "fresh_owner_confirmed" for r in rows),
        "recovered_confirmed": sum(r["recovery"] and r["final"] ==
            "owner_confirmed" for r in rows),
        "final_owner_confirmed": sum(r["final"] == "owner_confirmed"
                                     for r in rows),
        "final_hold_unknown": sum(r["final"] == "hold_unknown" for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
