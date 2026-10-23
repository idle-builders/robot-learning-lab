"""V180: read-only existing visual-reanchor gate on information-route stops."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v180"
V178 = ROOT / "artifacts/ape-closed-loop-cove50-v178/heldout-origin-route-audit.json"


def score(seed: int, origin: dict) -> dict:
    path = ROOT / f"artifacts/ape-closed-loop-cove50-v179/scene{seed}/seed{seed}/trace.json"
    trace = json.loads(path.read_text())
    event = next(e for e in reversed(trace["events"])
                 if e.get("action") == "guarded_approach_step")
    selected = next((proposal for proposal in event["proposals"]
                     if proposal["box"] == event["accepted_box"]), None)
    anchor = np.asarray(origin["actor_target_xy_m"])
    drift = float(np.linalg.norm(np.asarray(event["actor_target_xy_m"]) - anchor))
    innovation = None if selected is None else selected["distance_to_memory_m"]
    route_kind = event["detail"]["planner_goal"]["goal_kind"]
    actor_qualifies = (event["detail"]["consecutive_scan_rejections"] >= 2 and
                       route_kind == "committed_information" and
                       innovation is not None and innovation <= .02 and drift <= .04)
    return {"seed": seed, "trace_sha256": sha(path), "route_kind": route_kind,
            "consecutive_scan_rejections": event["detail"]["consecutive_scan_rejections"],
            "fresh_accepted": event["accepted_box"] is not None,
            "target_innovation_m": innovation,
            "target_anchor_drift_m": round(drift, 5),
            "actor_existing_visual_reanchor_qualifies": actor_qualifies,
            "evaluator_accepted_correct": event["accepted_correct_evaluator_only"],
            "evaluator_pose_error_m": event["evaluator_pose_error_m"],
            "evaluator_body_clearance_m": event["evaluator_model_clearance_m"],
            "rejected_scan": event["detail"]["localizer"]}


def main() -> None:
    origins = {r["seed"]: r for r in json.loads(V178.read_text())["rows"]}
    rows = [score(seed, origins[seed]) for seed in (89, 93)]
    result = {"scope": "Read-only V179 information-route second scan rejection with unchanged actor visual-anchor innovation<=2cm and owner-origin drift<=4cm gates; evaluator correctness/clearance scored after qualification. No actor motion authorized.",
              "source_sha256": sha(SOURCE), "v178_sha256": sha(V178), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "information-reanchor-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "qualifies", row["actor_existing_visual_reanchor_qualifies"],
              "innovation", row["target_innovation_m"], "anchor drift",
              row["target_anchor_drift_m"], flush=True)


if __name__ == "__main__":
    main()
