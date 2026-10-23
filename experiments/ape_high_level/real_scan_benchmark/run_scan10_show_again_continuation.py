"""Resume three safely stopped transfer episodes after a scripted owner reshow.

The owner moves the named prop into the currently visible floor area and
points at it. Only that scripted action uses evaluator geometry. The actor
receives a new RGB-D frame, an image point, and a yes/no crop confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from pathlib import Path

import numpy as np

from freeze_scan20_shared_policy_sources import relative, sha


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan10_policy_transfer_sources.json"
PRIOR = ART / "scan10_recovery_policy_report.json"
PRIOR_OUT = ROOT / "assets/real_indoor_scenes/scan10_recovery_policy_20261015"
PRECOMMIT = ART / "scan10_show_again_precommit.json"
REPORT = ART / "scan10_show_again_report.json"
OUT = ROOT / "assets/real_indoor_scenes/scan10_show_again_20261015"
SOURCE_OWNER = OUT / "source_owner"


def cases() -> list[dict]:
    source = {x["scene_key"]: x for x in json.loads(SOURCE.read_text())["sources"]}
    rows = json.loads(PRIOR.read_text())["rows"]
    selected = [source[r["scene_key"]] for r in rows if
                (r.get("final_action") or "").startswith("ask_owner_show_again")]
    if len(selected) != 3 or len({x["capture_group"] for x in selected}) != 3:
        raise ValueError("Expected three distinct owner-show-again stops")
    return selected


def freeze(selected: list[dict]) -> dict:
    here = Path(__file__).parent
    sources = [Path(__file__), here / "run_alt_pair_early_reteach.py",
               here / "run_alt_pair_full_action_20.py",
               here / "capture_range_monitor.py",
               here / "run_real_capture_conditional_search.py"]
    return {"schema": "ape.scan10-show-again-precommit.v1",
        "source": relative(SOURCE), "source_sha256": sha(SOURCE),
        "prior_report": relative(PRIOR), "prior_report_sha256": sha(PRIOR),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in sources],
        "cases": [{"scene_key": x["scene_key"],
                   "capture_group": x["capture_group"],
                   "item_screen_sha256": x["item_screen_sha256"]}
                  for x in selected],
        "policy": "For every visible, pose-healthy owner stop that requested another show, keep the same stopped actor state. Script one owner action that places the named item on the first supported, visibly rendered floor position from a fixed near-field grid, then points to its pixels. Choose the smallest actor-generated generic proposal containing the point; if absent try the existing SAM2 click. Accept only an owner-confirmed crop. From that memory, run unchanged present/absent search and, only after fresh RGB-D association plus healthy pose and independent two-height range, a bounded approach and final review. If any stage cannot be supported, hold unknown. No per-room threshold or true target endpoint enters actor decisions.",
        "limits": "Continuation of already retained full episodes, not an independent fresh start. The rigid owner does not animate the reshow; the evaluator stages the owner's prop movement and truthful point. The actor remains a 17cm proxy with ideal camera/yaw and virtual range and historical pose-shaped increments."}


def stage(case: dict, folder: Path) -> tuple[dict | None, dict]:
    import run_alt_pair_early_reteach as reteach
    from real_capture_dynamic_scene import TexturedFloorObject
    from run_real_capture_owner_loop import SCENES, owner_pose
    from run_real_capture_teach_find import camera

    key = case["scene_key"]
    prior = PRIOR_OUT / key / case["photometry"] / "healthy/initial/owner_approach/trace.json"
    owner = json.loads(prior.read_text())
    last = owner["trace"][-1]
    if not last["pose_healthy"]:
        return None, {"reason": "pose_unhealthy"}
    source_owner = SOURCE_OWNER / key / "owner_approach"
    source_owner.mkdir(parents=True, exist_ok=True)
    shutil.copy2(prior, source_owner / "trace.json")
    simple = {"scene_key": key, "screen": case["item_screen"],
              "target": case["item"]["target"],
              "distractor": case["item"]["distractor"]}
    scene = reteach.make_scene(simple, True)
    original_screen = json.loads((ROOT / case["item_screen"]).read_text())
    placement = next(x for x in original_screen["scenes"] if x.get("scene_key") == key)
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    target = TexturedFloorObject(props / f"{simple['target']}.obj", scale=.012)
    distractor = TexturedFloorObject(props / f"{simple['distractor']}.obj", scale=.012)
    actual = np.asarray(last["evaluator_actual_xz_m"], float)
    yaw = float(last["evaluator_actual_heading_deg"])
    heading = math.radians(yaw)
    forward = np.array([math.sin(heading), math.cos(heading)])
    right = np.array([-math.cos(heading), math.sin(heading)])
    owner_at_capture = owner_pose(float(last["decision_time_s"] + .2), SCENES[key])
    tested = []
    selected = None
    for radius in (.55, .7, .85, 1.0):
        if selected is not None:
            break
        for side in (0., -.25, .25):
            position = actual + radius * forward + side * right
            if not scene.floor_supported(*position):
                tested.append({"radius_m": radius, "side_m": side,
                               "floor_supported": False})
                continue
            scene.set_props({simple["target"]: (target, tuple(position), 0.),
                simple["distractor"]: (distractor,
                    tuple(placement["teach_distractor_xz"]), 0.)})
            _rgb, _depth, ids, _ranges = camera(scene, actual, yaw, owner_at_capture)
            visible = int((ids == simple["target"]).sum())
            tested.append({"radius_m": radius, "side_m": side,
                           "floor_supported": True, "target_visible_px": visible})
            if visible >= 400:
                selected = position.tolist()
                break
    if selected is None:
        return None, {"reason": "no_supported_owner_presentation", "tested": tested}
    staged = {**placement, "teach_target_xz": selected}
    screen = {**original_screen,
              "scenes": [staged if x.get("scene_key") == key else x
                         for x in original_screen["scenes"]]}
    screen_path = folder / "owner_reshow_screen.json"
    folder.mkdir(parents=True, exist_ok=True)
    screen_path.write_text(json.dumps(screen, indent=2) + "\n")
    return {**simple, "screen": relative(screen_path),
            "capture_group": case["capture_group"],
            "source_status": "owner_show_again_requested", "branch": "direct"}, {
            "reason": "staged_owner_presentation", "selected_xz_evaluator": selected,
            "tested": tested, "source_owner_trace": relative(prior),
            "screen": relative(screen_path)}


def run_one(case: dict) -> dict:
    import run_alt_pair_early_reteach as reteach
    import run_alt_pair_full_action_20 as action

    folder = OUT / case["scene_key"]
    staged_case, staging = stage(case, folder)
    row = {"schema": "ape.scan10-show-again.v1",
        "scene_key": case["scene_key"], "capture_group": case["capture_group"],
        "source_episode": relative(PRIOR_OUT / case["scene_key"] /
            case["photometry"] / "healthy/summary.json"),
        "staging": staging, "teaching": None, "action": None,
        "result": "safe_unresolved"}
    if staged_case is None:
        (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        return row
    old_source, old_out = reteach.SOURCE_ROOT, reteach.OUT
    old_action_source, old_action_out = action.SOURCE_ROOT, action.OUT
    try:
        reteach.SOURCE_ROOT = SOURCE_OWNER
        reteach.OUT = OUT
        teach = reteach.run_arm(staged_case, "accurate")
        row["teaching"] = teach
        if teach["teach_confirmed"] and teach.get("present") and \
                teach["present"]["action"] == "owner_confirmed_visual_identity":
            action.SOURCE_ROOT = OUT
            action.OUT = folder / "continuation_action"
            row["action"] = action.run_one(staged_case, "healthy")
            row["result"] = ("completed" if row["action"].get("final_action") ==
                "owner_confirmed_visual_identity" else "safe_unresolved")
    except Exception as error:
        row["error"] = repr(error)
        row["error_traceback"] = traceback.format_exc()[-3000:]
        row["result"] = "continuation_failed"
    finally:
        reteach.SOURCE_ROOT, reteach.OUT = old_source, old_out
        action.SOURCE_ROOT, action.OUT = old_action_source, old_action_out
    (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    args = parser.parse_args()
    selected = cases()
    definition = freeze(selected)
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Show-again policy changed after freeze")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(selected))
        return
    rows = []
    for case in selected:
        if args.scene and args.scene != case["scene_key"]:
            continue
        if sha(ROOT / case["item_screen"]) != case["item_screen_sha256"]:
            raise ValueError("Frozen item screen changed")
        path = OUT / case["scene_key"] / "summary.json"
        row = json.loads(path.read_text()) if path.is_file() else run_one(case)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.scan10-show-again-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows), "rows": rows},
            indent=2) + "\n")
        print(case["scene_key"], row["result"], flush=True)


if __name__ == "__main__":
    main()
