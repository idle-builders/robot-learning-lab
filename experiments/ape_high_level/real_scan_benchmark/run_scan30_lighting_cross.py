"""Counterbalance camera lighting in the frozen 30-room whole owner task.

Each room keeps its source geometry, object pair, policy and sensor gates. The
only scenario change is the opposite camera photometry from Scan30 healthy.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from freeze_scan20_shared_policy_sources import relative, sha
import run_scan30_unified_policy as unified


ROOT = unified.ROOT
ART = unified.ART
HERE = Path(__file__).resolve().parent
BASELINE = ART / "scan30_unified_policy_precommit.json"
BASELINE_REPORT = ART / "scan30_unified_policy_report.json"
MOTION_TRACE = ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1/varied150-forward-seed144-trace.json"
PRECOMMIT = ART / "scan30_lighting_cross_precommit.json"
REPORT = ART / "scan30_lighting_cross_report.json"
OUT = ROOT / "assets/real_indoor_scenes/scan30_lighting_cross_20261015"


def cohort() -> list[dict]:
    original = unified.cohort()
    cases = [{**case, "photometry":
              "cool_dim" if case["photometry"] == "none" else "none",
              "matched_photometry": case["photometry"]} for case in original]
    if len(cases) != 30 or len({c["capture_group"] for c in cases}) != 30:
        raise ValueError("Expected 30 independent captured rooms")
    return cases


def freeze(cases: list[dict]) -> dict:
    baseline = json.loads(BASELINE.read_text())
    policy_sources = [Path(__file__)] + [ROOT / entry["path"]
        for entry in baseline["policy_sources"]]
    return {"schema": "ape.scan30-lighting-cross-precommit.v1",
        "baseline": relative(BASELINE), "baseline_sha256": sha(BASELINE),
        "baseline_report": relative(BASELINE_REPORT),
        "baseline_report_sha256": sha(BASELINE_REPORT),
        "historical_motion_trace": relative(MOTION_TRACE),
        "historical_motion_trace_sha256": sha(MOTION_TRACE),
        "development_manifest": baseline["development_manifest"],
        "development_sha256": baseline["development_sha256"],
        "transfer_manifest": baseline["transfer_manifest"],
        "transfer_sha256": baseline["transfer_sha256"],
        "policy_sources": [{"path": relative(p), "sha256": sha(p)}
                           for p in policy_sources],
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"],
                   "cohort": c["cohort"],
                   "baseline_photometry": c["matched_photometry"],
                   "cross_photometry": c["photometry"],
                   "photometry": c["photometry"],
                   "source_mesh_sha256": c["source_mesh_sha256"],
                   "owner_screen_sha256": c["owner_screen_sha256"],
                   "item_screen_sha256": c["item_screen_sha256"]}
                  for c in cases],
        "policy": "Rerun the entire frozen Scan30 owner approach, teach, present/absent find, fresh locate, guarded approach, reshow-if-requested and final review from a fresh start. Change only the camera photometry between normal and cool/dim in each matched room, including search and route views during a reshow continuation. Keep the same item pair, room geometry, owner behavior, vision model, search thresholds, pose source, independent range gate and route planner. Count every room, including safe stops and runner failures; no room-specific repair.",
        "score": ["whole-task completion by matched room",
                  "wrong candidate offers", "owner reshow burden",
                  "safe unresolved stage", "motion authorization",
                  "first-person video and RGB-D/range/pose trace"],
        "limits": "Previously exposed captured-room cohort; paired lighting robustness, not independent room or item transfer. Static rooms, scripted rigid owner and inserted props, ideal depth/yaw/range, provisional 17 cm proxy and rejected-body motion-shaped odometry. The comparison does not isolate physical camera exposure or qualify the selected 50 cm body."}


def verify_case(case: dict) -> None:
    for name, digest in (("source_mesh_resolved", "source_mesh_sha256"),
                         ("owner_screen", "owner_screen_sha256"),
                         ("item_screen", "item_screen_sha256")):
        path = case[name]
        if path is None:
            if case[digest] is not None:
                raise ValueError(f"Missing {name} digest: {case['scene_key']}")
        elif sha(ROOT / path) != case[digest]:
            raise ValueError(f"Changed {name}: {case['scene_key']}")


def install_cross_reshow(show) -> None:
    """Apply the scenario's search lighting during an owner reshow as well."""
    import run_alt_pair_full_action_20 as action
    import run_real_capture_conditional_search as conditional

    original_show = show.run_one
    original_search = conditional.conditional_run_search
    original_walk = action.motion.walk

    def run_one(case: dict) -> dict:
        photometry = case["photometry"]

        def search(*args, **kwargs):
            kwargs["search_photometry"] = photometry
            return original_search(*args, **kwargs)

        def walk(*args, **kwargs):
            kwargs["photometry"] = photometry
            return original_walk(*args, **kwargs)

        conditional.conditional_run_search = search
        action.motion.walk = walk
        try:
            return original_show(case)
        finally:
            conditional.conditional_run_search = original_search
            action.motion.walk = original_walk

    show.run_one = run_one


def require_inference_weights() -> None:
    root = ROOT / ".cache/ape-high-level/models"
    for name in ("grounding-dino-tiny", "dinov2-small",
                 "sam2-hiera-tiny"):
        folder = root / name
        if not folder.is_dir() or not any(folder.rglob("*.safetensors")) and \
                not any(folder.rglob("*.bin")):
            raise FileNotFoundError(
                f"Model weights missing in {folder}; stage the frozen Scan30 "
                "revisions before starting any room")


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
            raise ValueError("Frozen lighting comparison changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    if sha(MOTION_TRACE) != definition["historical_motion_trace_sha256"]:
        raise ValueError("Frozen motion-packet source changed")
    require_inference_weights()
    import run_scan20_shared_owner_policy as policy
    import run_scan10_recovery_policy as recovery
    import run_scan10_show_again_continuation as show
    unified.BASE_OUT = OUT / "base"
    unified.SHOW_OUT = OUT / "show"
    policy.OUT = unified.BASE_OUT
    recovery.OUT = unified.BASE_OUT
    show.PRIOR_OUT = unified.BASE_OUT
    show.OUT = unified.SHOW_OUT
    show.SOURCE_OWNER = unified.SHOW_OUT / "source_owner"
    recovery.install_recoveries(policy)
    install_cross_reshow(show)
    rows = []
    launched = 0
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        verify_case(case)
        result_file = OUT / case["scene_key"] / "combined_summary.json"
        if result_file.is_file():
            row = json.loads(result_file.read_text())
        elif args.max_new and launched >= args.max_new:
            break
        else:
            try:
                row = unified.attempt(case, policy, recovery, show)
            except Exception as error:
                row = {"schema": "ape.scan30-unified-policy.v1",
                       "scene_key": case["scene_key"],
                       "capture_group": case["capture_group"],
                       "cohort": case["cohort"],
                       "photometry": case["photometry"],
                       "result": "runner_failed", "error": repr(error),
                       "error_traceback": traceback.format_exc()[-3000:]}
            result_file.parent.mkdir(parents=True, exist_ok=True)
            result_file.write_text(json.dumps(row, indent=2) + "\n")
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.scan30-lighting-cross-report.v1", "precommit": relative(PRECOMMIT),
            "processed": len(rows), "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], case["photometry"], row["result"],
              flush=True)


if __name__ == "__main__":
    main()
