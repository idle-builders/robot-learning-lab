"""One shared recovery-state revision over the ten-room transfer cohort.

The original policy remains frozen. This runner adds only generic transitions:
an extra current-view sweep/owner point before an uncertain move, a proposal
fallback for a failed teaching click, and an explicit show-again/stop response.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from freeze_scan20_shared_policy_sources import relative, sha


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan10_policy_transfer_sources.json"
PRECOMMIT = ART / "scan10_recovery_policy_precommit.json"
REPORT = ART / "scan10_recovery_policy_report.json"
OUT = ROOT / "assets/real_indoor_scenes/scan10_recovery_policy_20261015"
BASE_POLICY = Path(__file__).with_name("run_scan20_shared_owner_policy.py")


def freeze(cases: list[dict]) -> dict:
    return {
        "schema": "ape.scan10-recovery-policy-precommit.v1",
        "source_manifest": relative(SOURCE), "source_sha256": sha(SOURCE),
        "base_policy_source": relative(BASE_POLICY),
        "base_policy_sha256": sha(BASE_POLICY),
        "recovery_source": relative(Path(__file__)),
        "recovery_sha256": sha(Path(__file__)),
        "cases": [{"scene_key": c["scene_key"],
                   "capture_group": c["capture_group"],
                   "item_screen_sha256": c["item_screen_sha256"],
                   "source_mesh_sha256": c["source_mesh_sha256"]}
                  for c in cases],
        "policy": "Start a new full episode with the frozen shared policy. If the teaching click fails, choose the smallest actor-detected generic proposal that contains it; the scripted owner still confirms or rejects the crop. If no crop exists, ask to show again and hold identity unknown. If a fresh pre-move relook is unidentified, perform one five-heading sweep at the stopped pose, then at most one owner point on an actor-seen proposal followed by a new relook. Keep the same metric association, independent range and pose gates before motion. If the owner is visible but the pre-screened teaching endpoint is not reached, ask for a new presentation at the current safe viewpoint; never walk from evaluator owner coordinates or claim a completed task. Pose-unhealthy stops remain stops.",
        "score": ["complete whole task", "teaching selection and owner confirmation",
                  "fresh metric motion authorization", "wrong-candidate questions",
                  "safe owner and pose stops", "visual/range/pose evidence"],
        "limits": "Development-exposed recorded captures with rigid scanned owner, scripted truthful reply, staged HOPE props, ideal RGB-D/yaw and virtual two-height range on a 17 cm kinematic proxy. A request to show again is an interaction decision, not a completed task. No selected mechanical body or trained high-level policy."}


def install_recoveries(policy) -> None:
    original_search = policy.search
    original_click = policy.baseline.Vision.select_click

    def select_click(self, image, click, proposals):
        box, info = original_click(self, image, click, proposals)
        if box is not None:
            return box, info
        x, y = click
        enclosing = [p for p in proposals if p["box"][0] <= x <= p["box"][2]
                     and p["box"][1] <= y <= p["box"][3]]
        if not enclosing:
            return None, {"source": "generic_fallback_after_click_failure",
                          "reason": "no_actor_proposal_enclosed_owner_point",
                          "initial_selector": info}
        chosen = min(enclosing, key=lambda p:
                     (p["box"][2]-p["box"][0]) *
                     (p["box"][3]-p["box"][1]))
        return [round(v) for v in chosen["box"]], {
            "source": "smallest_generic_proposal_at_owner_point",
            "initial_selector": info, "generic_label": chosen.get("generic_label")}

    def search(scene, case, state, start, memory, present, out,
               photometry, *, wide=False):
        result = original_search(scene, case, state, start, memory, present,
                                 out, photometry, wide=wide)
        if out.name != "fresh_before_move" or result["action"] == \
                "owner_confirmed_visual_identity":
            return result
        sources = [(out, result)]
        expanded = original_search(scene, case, state, start, memory, True,
                                   out / "wide", photometry, wide=True)
        sources.append((out / "wide", expanded))
        if expanded["action"] == "owner_confirmed_visual_identity":
            expanded["recovery"] = "fresh_wide_relook"
            return expanded
        initial = out.parent / "initial/summary.json"
        teach = json.loads(initial.read_text()).get("teaching") if initial.is_file() else None
        if teach is not None:
            new_memory = policy.owner_point_memory(sources, teach,
                                                    out / "owner_point")
            if new_memory is not None:
                memory.update(new_memory)
                helped = original_search(scene, case, state, start, memory,
                                         True, out / "help", photometry)
                helped["recovery"] = "fresh_owner_point_relook"
                return helped
        expanded["recovery"] = "fresh_relook_unresolved"
        return expanded

    policy.baseline.Vision.select_click = select_click
    policy.search = search


def explicit_stop(policy, case: dict, photometry: str, error: Exception) -> dict:
    out = OUT / case["scene_key"] / photometry / "healthy"
    initial = out / "initial"
    trace = initial / "owner_approach/trace.json"
    status = "teaching_no_visible_crop" if "Teaching click selected no visible proposal" in str(error) else "runner_failed"
    row = {"schema": "ape.scan10-recovery-policy.v1",
           "scene_key": case["scene_key"], "capture_group": case["capture_group"],
           "photometry": photometry, "arm": "healthy", "result":
           "safe_unresolved" if status != "runner_failed" else "runner_failed",
           "owner_status": status, "present_action": None,
           "absent_action": None, "fresh_action": None,
           "motion_authorized": False, "final_action": "ask_owner_show_again_hold_unknown"
           if status != "runner_failed" else None,
           "error": repr(error), "error_traceback": traceback.format_exc()[-3000:],
           "owner_approach_trace": relative(trace) if trace.is_file() else None,
           "teaching_rgb": relative(initial / "teach/rgb_000.png") if
           (initial / "teach/rgb_000.png").is_file() else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--scene")
    args = parser.parse_args()
    cases = json.loads(SOURCE.read_text())["sources"]
    if len(cases) != 10 or len({c["capture_group"] for c in cases}) != 10:
        raise ValueError("Expected ten distinct transfer captures")
    definition = freeze(cases)
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Recovery policy changed after freeze")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(cases))
        return
    import run_scan20_shared_owner_policy as policy
    policy.OUT = OUT
    install_recoveries(policy)
    rows = []
    new = 0
    for case in cases:
        if args.scene and case["scene_key"] != args.scene:
            continue
        if sha(ROOT / case["item_screen"]) != case["item_screen_sha256"]:
            raise ValueError(f"Changed item screen: {case['scene_key']}")
        if sha(ROOT / case["source_mesh_resolved"]) != case["source_mesh_sha256"]:
            raise ValueError(f"Changed source mesh: {case['scene_key']}")
        out = OUT / case["scene_key"] / case["photometry"] / "healthy"
        summary = out / "summary.json"
        if summary.is_file():
            row = json.loads(summary.read_text())
        elif args.max_new and new >= args.max_new:
            break
        else:
            try:
                row = policy.run_case(case, case["photometry"], "healthy")
            except Exception as error:
                row = explicit_stop(policy, case, case["photometry"], error)
            if row.get("owner_status") == "owner_viewpoint_not_reached":
                trace = out / "initial/owner_approach/trace.json"
                if trace.is_file():
                    last = json.loads(trace.read_text())["trace"][-1]
                    owner = last.get("owner_actor", {})
                    visible_safe = bool(last.get("pose_healthy") and
                        owner.get("local_depth_m") is not None and
                        float(owner["local_depth_m"]) <= 1.7 and
                        abs(float(owner.get("owner_bearing_deg", 999.))) <= 20.)
                    if visible_safe:
                        row.update(result="safe_unresolved",
                                   final_action="ask_owner_show_again_at_current_view",
                                   owner_recovery="visible_safe_far_viewpoint")
                        (out / "summary.json").write_text(
                            json.dumps(row, indent=2) + "\n")
            new += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.scan10-recovery-policy-report.v1",
            "source_manifest": relative(SOURCE), "source_sha256": sha(SOURCE),
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["result"], row.get("final_action"), flush=True)


if __name__ == "__main__":
    main()
