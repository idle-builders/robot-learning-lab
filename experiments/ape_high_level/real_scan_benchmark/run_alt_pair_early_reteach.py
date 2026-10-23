"""One show-again policy after incomplete teaching in three retained rooms.

The actor receives a newly rendered owner-pointed RGB-D view. The scripted
owner may point at the named item, point at the distractor, or stay silent.
The owner reviews the selected crop before any memory or find is created.
Accurate, confirmed teaching continues through present/absent search and the
same fresh-relook/range-guarded action policy used in the prior 15 rooms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import (SCENES, bbox, camera,
    inference_device, _iou)
from run_real_capture_owner_loop import owner_pose
from real_capture_dynamic_scene import TexturedFloorObject
from run_compound_owner_point_recovery import make_scene
import run_real_capture_conditional_search as conditional
import run_alt_pair_full_action_20 as full_action


HERE = Path(__file__).resolve().parent
SOURCE_PRECOMMIT = ART / "alt_pair_compound_20_precommit.json"
SOURCE_REPORT = ART / "alt_pair_compound_20_report.json"
PRECOMMIT = ART / "alt_pair_early_reteach_precommit.json"
REPORT = ART / "alt_pair_early_reteach_report.json"
OUT = ROOT / "assets/real_indoor_scenes/alt_pair_early_reteach_20261015"
ACTION_OUT = ROOT / "assets/real_indoor_scenes/alt_pair_early_reteach_action_20261015"
SOURCE_ROOT = ROOT / "assets/real_indoor_scenes/alt_pair_compound_20_20261015"
ARMS = ("accurate", "wrong", "silent")


def cases() -> list[dict]:
    original = {c["scene_key"]: c for c in json.loads(
        SOURCE_PRECOMMIT.read_text())["cases"]}
    rows = json.loads(SOURCE_REPORT.read_text())["rows"]
    needed = {"owner_rejected_teaching_crop", "owner_viewpoint_outside_frozen_item_placement"}
    selected = [{**original[r["scene_key"]], "source_status": r["status"]}
        for r in rows if r["status"] in needed]
    if len(selected) != 3 or len({c["capture_group"] for c in selected}) != 3:
        raise ValueError("Expected exactly three supported early task failures")
    return selected


def frozen() -> dict:
    policy = [Path(__file__), HERE / "run_alt_pair_full_action_20.py",
        HERE / "capture_range_monitor.py", HERE / "run_real_capture_conditional_search.py",
        HERE / "run_real_capture_teach_find.py",
        HERE / "run_compound_owner_point_recovery.py"]
    return {"schema": "ape.alt-pair-early-reteach-precommit.v1",
        "source_precommit": relative(SOURCE_PRECOMMIT),
        "source_precommit_sha256": sha(SOURCE_PRECOMMIT),
        "source_report": relative(SOURCE_REPORT),
        "source_report_sha256": sha(SOURCE_REPORT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in policy],
        "cases": cases(), "owner_reply_arms": list(ARMS),
        "action_arms": ["healthy", "range_gap"],
        "policy": "When a prior owner encounter has no accepted taught exemplar, request one new show and point in the current RGB-D view. Choose the smallest existing generic image proposal containing the point; if none exists, try the existing SAM2 click selector once. Show the crop and keep memory unknown until the scripted owner confirms it. Accurate, wrong and silent responses share this rule. After confirmation, run the unchanged generic-first/expanded multi-cue present and absent search, then the same fresh-relook, independent-range-guarded pose-shaped bounded action and final visual review as the 15 earlier rooms. The target need not be invisible in the first search view; that was a hidden-item scenario condition, not a general owner-task requirement. No per-room threshold change.",
        "score": ["three source failures retained", "owner-point acceptance",
            "wrong/silent no-claim behavior", "present and absent final identity",
            "fresh action outcome", "source-gap hold", "RGB/depth/video/packet evidence"],
        "limits": "Saved stopped-owner state and simulator-scripted point/yes/no, not a fresh uninterrupted owner route. Development-exposed static scans, inserted rigid person/HOPE props, ideal depth/range, historical rejected-body pose-shaped increments and 17 cm disk. Two unsupported source placements remain in the 20-room denominator."}


def save_depth(path: Path, depth: np.ndarray) -> None:
    Image.fromarray(np.where(np.isfinite(depth),
        np.clip(np.rint(depth * 1000.), 0, 65535), 0).astype(np.uint16)).save(path)


def source_state(case: dict) -> tuple[dict, dict]:
    key = case["scene_key"]
    owner = json.loads((SOURCE_ROOT / key / "owner_approach/trace.json").read_text())
    return owner, owner["trace"][-1]


def select_crop(vision, image: Image.Image, proposals: list[dict],
                click: tuple[int, int]) -> tuple[list[float] | None, dict]:
    x, y = click
    matching = [p for p in proposals if p["box"][0] <= x <= p["box"][2]
                and p["box"][1] <= y <= p["box"][3]
                and p["box"][0] >= 8 and p["box"][2] <= image.width - 8]
    if matching:
        selected = min(matching, key=lambda p: (p["box"][2] - p["box"][0]) *
                       (p["box"][3] - p["box"][1]))
        return selected["box"], {"source": "smallest_generic_proposal_at_owner_point",
                                  "selected_generic_label": selected.get("generic_label")}
    box, info = vision.select_click(image, list(click), proposals)
    return box, {"source": "existing_sam2_click_fallback", "detail": info}


def run_arm(case: dict, arm: str,
            abort_if: Callable[[], bool] | None = None) -> dict:
    key = case["scene_key"]
    out = OUT / key / arm
    out.mkdir(parents=True, exist_ok=True)
    if arm == "silent":
        row = {"scene_key": key, "capture_group": case["capture_group"],
            "source_status": case["source_status"], "arm": arm,
            "owner_signal": None, "teach_confirmed": False,
            "action": "ask_show_again_hold_unknown", "present": None,
            "absent": None, "motion_target": None}
        (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
        return row
    owner, last = source_state(case)
    cfg = SCENES[key]
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(p for p in screen["scenes"] if p.get("scene_key") == key)
    scene = make_scene(case, True)
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    scene.set_props({name: (TexturedFloorObject(props / f"{name}.obj", scale=.012),
        tuple(placement[position]), 0.) for name, position in (
            (case["target"], "teach_target_xz"),
            (case["distractor"], "teach_distractor_xz"))})
    actual = np.asarray(last["evaluator_actual_xz_m"], float)
    yaw = float(last["evaluator_actual_heading_deg"])
    actor = np.asarray(last["actor_pose_xy_m"], float)
    actor_yaw = float(last["actor_heading_deg"])
    captured_wall_s = time.perf_counter()
    rgb, depth, ids, ranges = camera(scene, actual, yaw,
        owner_pose(float(last["decision_time_s"] + .2), cfg))
    image = Image.fromarray(rgb)
    teaching = (OUT / key / "teach") if arm == "accurate" else (out / "teach")
    teaching.mkdir(exist_ok=True)
    image.save(teaching / "rgb_000.png")
    save_depth(teaching / "depth_000.png", depth)
    named = case["target"] if arm == "accurate" else case["distractor"]
    mask = ids == named
    ys, xs = np.where(mask)
    if len(xs) < 150:
        raise ValueError(f"Scripted owner cannot point to visible {named}: {key}")
    click = (int(np.median(xs)), int(np.median(ys)))
    vision = conditional.ConditionalVision("sam2-click",
                                          device=inference_device())
    proposals = vision.detect(image)
    if abort_if is not None and abort_if():
        raise InterruptedError("Owner show invalidated during image inference")
    selected, selection_info = select_crop(vision, image, proposals, click)
    if abort_if is not None and abort_if():
        raise InterruptedError("Owner show invalidated before crop confirmation")
    target_box = bbox(ids == case["target"])
    iou = _iou(selected, target_box) if selected is not None else 0.
    accepted = bool(selected is not None and iou >= .25)
    signal = {"kind": arm, "click_xy": click,
        "rgb": relative(teaching / "rgb_000.png"),
        "depth": relative(teaching / "depth_000.png"),
        "capture_wall_s": captured_wall_s,
        "source": "simulator-scripted owner point on current image"}
    teach = {"schema": "ape.alt-pair-reteach.v1", "scene_key": key,
        "owner_signal": signal, "name_input": case["target"],
        "selected_box_xyxy": selected, "selection_info": selection_info,
        "owner_confirmation": "confirmed" if accepted else "rejected",
        "source_rgb": signal["rgb"], "source_depth": signal["depth"],
        "capture_time_s": float(last["decision_time_s"] + .2),
        "proposals": proposals, "evaluator_target_box_xyxy": target_box,
        "evaluator_selected_target_iou": iou,
        "range_low_m": [float(x) if np.isfinite(x) else None for x in ranges[0]],
        "range_mid_m": [float(x) if np.isfinite(x) else None for x in ranges[1]]}
    row = {"scene_key": key, "capture_group": case["capture_group"],
        "source_status": case["source_status"], "arm": arm,
        "owner_signal": signal, "teach_confirmed": accepted,
        "teaching_selected_target_iou": iou, "present": None,
        "absent": None, "motion_target": None}
    if accepted:
        search_root = OUT / key if arm == "accurate" else out
        vision._teaching_frame_seen = True
        selected_image = image.crop(tuple(round(v) for v in selected))
        exemplar = vision.feature([selected_image])[0]
        negatives = [p for p in proposals if _iou(p["box"], selected) < .25]
        other_features = (vision.feature([image.crop(tuple(round(v) for v in p["box"]))
            for p in negatives]) if negatives else [])
        acceptance = max(.5, max((float(f @ exemplar) for f in other_features),
                                 default=0.) + .15)
        colour = conditional.baseline.colour_histogram(rgb, selected)
        colour_acceptance = max(.5, max((conditional.baseline.colour_similarity(
            colour, rgb, p["box"])
            for p in negatives), default=0.) + .1)
        teach.update({"acceptance_similarity": acceptance,
            "colour_acceptance_similarity": colour_acceptance})
        owner_source = SOURCE_ROOT / key / "owner_approach/trace.json"
        owner_out = search_root / "owner_approach"
        owner_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(owner_source, owner_out / "trace.json")
        for present, label in ((True, "present"), (False, "absent")):
            if abort_if is not None and abort_if():
                raise InterruptedError("Owner show invalidated before search")
            search_scene = make_scene(case, present)
            result = conditional.conditional_run_search(search_scene, vision,
                actual, yaw, actor, actor_yaw, exemplar, acceptance, present,
                search_root / f"find_{label}",
                start_time_s=float(last["decision_time_s"] + 10.),
                search_xz=tuple(placement["find_target_xz"]),
                search_distractor_xz=tuple(placement["find_distractor_xz"]),
                owner_start_xz=tuple(cfg["start_xz"]),
                target_name=case["target"], distractor_name=case["distractor"],
                identity_mode="multi_cue", colour_exemplar=colour,
                colour_acceptance=colour_acceptance,
                search_photometry="cool_dim")
            if abort_if is not None and abort_if():
                raise InterruptedError("Owner show invalidated during search")
            row[label] = {"action": result["action"],
                "wrong_candidate_offered": result["wrong_confident_candidate"],
                "owner_questions": result["owner_questions"],
                "trace": relative(search_root / f"find_{label}/episode_trace.json"),
                "video": result["video"]}
    row["action"] = ("teaching_confirmed" if accepted else
                      "ask_show_again_hold_unknown")
    (teaching / "trace.json").write_text(json.dumps(teach, indent=2) + "\n")
    (out / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def run_action(case: dict) -> list[dict]:
    key = case["scene_key"]
    recovery = json.loads((OUT / key / "accurate/summary.json").read_text())
    if not recovery["present"] or recovery["present"]["action"] != \
            "owner_confirmed_visual_identity":
        return []
    old_source, old_output = full_action.SOURCE_ROOT, full_action.OUT
    try:
        full_action.SOURCE_ROOT = OUT
        full_action.OUT = ACTION_OUT
        resumed_case = {**case, "branch": "direct"}
        return [full_action.run_one(resumed_case, arm)
                for arm in ("healthy", "range_gap")]
    finally:
        full_action.SOURCE_ROOT, full_action.OUT = old_source, old_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--scene")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    definition = frozen()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != definition:
            raise ValueError("Frozen early-reteach definition changed")
    else:
        PRECOMMIT.write_text(json.dumps(definition, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(definition["cases"]))
        return
    rows = []
    launched = 0
    for case in definition["cases"]:
        if args.scene and args.scene != case["scene_key"]:
            continue
        key = case["scene_key"]
        arms = []
        for arm in ARMS:
            path = OUT / key / arm / "summary.json"
            if path.exists():
                row = json.loads(path.read_text())
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_arm(case, arm)
                launched += 1
            arms.append(row)
            print(key, arm, row["action"],
                  (row["present"] or {}).get("action"), flush=True)
        if len(arms) == 3 and arms[0]["teach_confirmed"]:
            action_files = [ACTION_OUT / key / arm / "summary.json"
                for arm in ("healthy", "range_gap")]
            if all(path.exists() for path in action_files):
                actions = [json.loads(path.read_text()) for path in action_files]
            elif not args.max_new or launched < args.max_new:
                actions = run_action(case)
                launched += len(actions)
            else:
                actions = []
        else:
            actions = []
        rows.append({"scene_key": key, "capture_group": case["capture_group"],
            "source_status": case["source_status"], "arms": arms,
            "actions": actions})
        REPORT.write_text(json.dumps({"schema": "ape.alt-pair-early-reteach-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
