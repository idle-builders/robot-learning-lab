"""One frozen from-start owner-task state machine across 30 captured rooms.

The first 20 rooms are the prior development cohort; the other ten are the
previously exposed transfer cohort. Each attempt starts fresh. A safe request
may continue with one scripted owner reshow in that same task episode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

from freeze_scan20_shared_policy_sources import relative, sha


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
DEVELOPMENT = ART / "scan20_shared_policy_sources.json"
TRANSFER = ART / "scan10_policy_transfer_sources.json"
PRECOMMIT = ART / "scan30_unified_policy_precommit.json"
REPORT = ART / "scan30_unified_policy_report.json"
OUT = ROOT / "assets/real_indoor_scenes/scan30_unified_policy_20261015"
BASE_OUT = OUT / "base"
SHOW_OUT = OUT / "show"


def cohort() -> list[dict]:
    development = json.loads(DEVELOPMENT.read_text())["sources"]
    transfer = json.loads(TRANSFER.read_text())["sources"]
    if len(development) != 20 or len(transfer) != 10:
        raise ValueError("Expected 20 development and ten transfer sources")
    cases = [{**c, "cohort": "development",
              "photometry": "cool_dim" if i % 2 else "none"}
             for i, c in enumerate(development)]
    cases += [{**c, "cohort": "prior_transfer"} for c in transfer]
    if len({c["capture_group"] for c in cases}) != 30 or len({c["scene_key"]
            for c in cases}) != 30:
        raise ValueError("Cohorts are not 30 distinct capture groups")
    return cases


def freeze(cases: list[dict]) -> dict:
    here = Path(__file__).resolve().parent
    code = [Path(__file__), here / "run_scan20_shared_owner_policy.py",
            here / "run_scan10_recovery_policy.py",
            here / "run_scan10_show_again_continuation.py",
            here / "run_alt_pair_early_reteach.py",
            here / "run_alt_pair_full_action_20.py",
            here / "capture_range_monitor.py",
            here / "run_real_capture_teach_find.py",
            here / "run_real_capture_conditional_search.py"]
    original = json.loads((ART / "scan20_shared_owner_policy_precommit.json").read_text())
    return {"schema": "ape.scan30-unified-policy-precommit.v1",
        "development_manifest": relative(DEVELOPMENT),
        "development_sha256": sha(DEVELOPMENT),
        "transfer_manifest": relative(TRANSFER),
        "transfer_sha256": sha(TRANSFER),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in code],
        "models": original["models"],
        "cases": [{"scene_key": c["scene_key"],
            "capture_group": c["capture_group"], "cohort": c["cohort"],
            "photometry": c["photometry"],
            "source_mesh_sha256": c["source_mesh_sha256"],
            "owner_screen_sha256": c["owner_screen_sha256"],
            "item_screen_sha256": c["item_screen_sha256"]}
            for c in cases],
        "policy": "One new full owner approach and teaching/search attempt per room. On a failed teaching click use the smallest generic proposal enclosing the owner point, otherwise request a reshow and keep identity unknown. On a fresh pre-move unidentified view, sweep five additional headings, then permit one owner point on an actor-visible proposal and a fresh relook. Only an owner-confirmed current RGB-D target associated within 10 cm with prior identity, healthy pose, and independent 10 Hz two-height range can authorize bounded motion; final claim requires a new owner-reviewed healthy view. If a healthy pose and visually tracked owner are safely encountered but the pre-screened teaching viewpoint cannot be reached, or a teaching crop is unavailable, make one scripted owner reshow at the current stop. The owner places the named prop on the first floor-supported visible point of the fixed near-field grid, points in the new image, and confirms or rejects the actor's crop. The same present/absent search, fresh metric association, independent range, bounded walk, and final review then continue. Unhealthy pose and unresolved identity hold. All cases use one rule, no room-specific thresholds.",
        "score": ["30 distinct start-to-finish task attempts", "owner/teach outcome",
                  "present and absent identity", "wrong-candidate questions",
                  "reshow burden", "fresh metric/range/pose authorization",
                  "observed standoff", "final owner response", "media and faults"],
        "limits": "Both cohorts were development-exposed before this freeze. Captured static room meshes, rigid scanned scripted owner and owner-prop reshow, staged HOPE objects, ideal aligned RGB-D/yaw, virtual independent two-height range and 17 cm kinematic proxy with historical rejected-body pose-shaped increments. No new blind holdout, current selected mechanics, trained high-level policy, live moving-person response or physical sensor qualification."}


def safe_owner_reshow(row: dict, base: Path) -> bool:
    if row.get("owner_status") != "owner_viewpoint_not_reached":
        return False
    trace = base / "initial/owner_approach/trace.json"
    if not trace.is_file():
        return False
    last = json.loads(trace.read_text())["trace"][-1]
    owner = last.get("owner_actor", {})
    safe = bool(last.get("pose_healthy") and owner.get("local_depth_m")
        is not None and float(owner["local_depth_m"]) <= 1.7 and
        abs(float(owner.get("owner_bearing_deg", 999.))) <= 20.)
    if safe:
        row.update(result="safe_unresolved",
                   final_action="ask_owner_show_again_at_current_view",
                   owner_recovery="visible_safe_far_viewpoint")
        (base / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return safe


def attempt(case: dict, policy, recovery, show) -> dict:
    key = case["scene_key"]
    photometry = case["photometry"]
    base = BASE_OUT / key / photometry / "healthy"
    base_summary = base / "summary.json"
    if base_summary.is_file():
        first = json.loads(base_summary.read_text())
    else:
        try:
            first = policy.run_case(case, photometry, "healthy")
        except Exception as error:
            first = recovery.explicit_stop(policy, case, photometry, error)
        safe_owner_reshow(first, base)
    continuation = None
    if (first.get("final_action") or "").startswith("ask_owner_show_again"):
        continuation_path = SHOW_OUT / key / "summary.json"
        continuation = (json.loads(continuation_path.read_text())
            if continuation_path.is_file() else show.run_one(case))
    result = continuation["result"] if continuation else first["result"]
    row = {"schema": "ape.scan30-unified-policy.v1", "scene_key": key,
           "capture_group": case["capture_group"], "cohort": case["cohort"],
           "photometry": photometry,
           "base_summary": relative(base_summary),
           "base_result": first["result"],
           "owner_status": first.get("owner_status"),
           "show_summary": relative(SHOW_OUT / key / "summary.json") if
               continuation else None,
           "show_result": continuation["result"] if continuation else None,
           "result": result,
           "motion_authorized": bool((continuation.get("action") or {}).get(
               "motion_authorized") if continuation else first.get(
               "motion_authorized")),
           "walk_result": ((continuation.get("action") or {}).get("walk_result")
               if continuation else first.get("walk_result")),
           "final_action": ((continuation.get("action") or {}).get("final_action")
               if continuation else first.get("final_action")),
           "wrong_candidate_questions": int(first.get(
               "wrong_candidate_questions", 0)) + (sum(int(((continuation.get(
                   "teaching") or {}).get(name) or {}).get(
                   "wrong_candidate_offered", False)) for name in
                   ("present", "absent")) if continuation else 0)}
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = cohort()
    definition = freeze(cases)
    if PRECOMMIT.is_file():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Unified policy or frozen source changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    policy.OUT = BASE_OUT
    recovery.OUT = BASE_OUT
    show.PRIOR_OUT = BASE_OUT
    show.OUT = SHOW_OUT
    show.SOURCE_OWNER = SHOW_OUT / "source_owner"
    recovery.install_recoveries(policy)
    rows = []
    new = 0
    for case in cases:
        if args.scene and args.scene != case["scene_key"]:
            continue
        if sha(ROOT / case["source_mesh_resolved"]) != case["source_mesh_sha256"]:
            raise ValueError(f"Changed captured room: {case['scene_key']}")
        for key, digest in (("item_screen", "item_screen_sha256"),
                            ("owner_screen", "owner_screen_sha256")):
            if case[key] is None:
                if case[digest] is not None:
                    raise ValueError(f"Unexpected missing {key}: {case['scene_key']}")
                continue
            if sha(ROOT / case[key]) != case[digest]:
                raise ValueError(f"Changed {key}: {case['scene_key']}")
        result_file = OUT / case["scene_key"] / "combined_summary.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text())
        elif args.max_new and new >= args.max_new:
            break
        else:
            try:
                row = attempt(case, policy, recovery, show)
            except Exception as error:
                row = {"schema": "ape.scan30-unified-policy.v1",
                    "scene_key": case["scene_key"],
                    "capture_group": case["capture_group"],
                    "cohort": case["cohort"], "photometry": case["photometry"],
                    "result": "runner_failed", "error": repr(error),
                    "error_traceback": traceback.format_exc()[-3000:]}
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(row, indent=2) + "\n")
            new += 1
        rows.append(row)
        REPORT.write_text(json.dumps({
            "schema": "ape.scan30-unified-policy-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["cohort"], case["scene_key"], row["result"], flush=True)


if __name__ == "__main__":
    main()
