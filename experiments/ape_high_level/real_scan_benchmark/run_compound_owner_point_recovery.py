"""Continue compound-fault identity misses with fallible owner pointing.

The scripted owner points into a saved search RGB frame. The actor receives
only that frame reference and click, selects an existing depth-localized
proposal, and runs a newly rendered search before any identity claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import (SCENES, TexturedCaptureScene,
    TexturedFloorObject, ScannedSittingPerson)
from real_capture_dynamic_scene import scene_from_capture_source
import run_real_capture_colour_bank_search as colour_bank


SOURCES = ((ART / "integrated_owner_20_compound_stress_audit.json",
            ROOT / "assets/real_indoor_scenes/integrated_owner_20_compound_stress_20261015"),
           (ART / "compound_stress_fresh_three_audit.json",
            ROOT / "assets/real_indoor_scenes/compound_stress_fresh_three_20261015"))
PRECOMMIT = ART / "compound_owner_point_recovery_precommit.json"
REPORT = ART / "compound_owner_point_recovery_report.json"
OUT = ROOT / "assets/real_indoor_scenes/compound_owner_point_recovery_20261015"
ARMS = (("accurate_present", "find_present", "accurate"),
        ("wrong_present", "find_present", "wrong"),
        ("wrong_absent", "find_absent", "wrong"),
        ("silent_present", "find_present", "silent"))


def case_rows() -> list[dict]:
    result = []
    for audit_path, folder in SOURCES:
        audit = json.loads(audit_path.read_text())
        for row in audit["rows"]:
            if (row["status"] == "teach_find_completed_far_viewpoint" and
                    row["find_present"]["action"] !=
                    "owner_confirmed_visual_identity"):
                summary = json.loads((folder / row["case"] / "summary.json").read_text())
                result.append({"scene_key": row["case"],
                    "source_folder": relative(folder / row["case"]),
                    "source_audit": relative(audit_path),
                    "source_summary": relative(folder / row["case"] / "summary.json"),
                    "screen": summary["scenario_placement_screen"],
                    "target": summary["object_pair"]["target"],
                    "distractor": summary["object_pair"]["distractor"]})
    return result


def observations(folder: Path, branch: str) -> list[dict]:
    out = []
    for stage in ("generic", "expanded"):
        path = folder / branch / stage / "trace.json"
        trace = json.loads(path.read_text())
        out.extend({**o, "stage": stage, "stage_folder": path.parent}
                   for o in trace["observations"])
    return out


def owner_signal(folder: Path, branch: str, kind: str) -> dict | None:
    if kind == "silent":
        return None
    for o in observations(folder, branch):
        candidates = [p for p in o["proposals"] if
            p.get("actor_map_xy_m") is not None and not p.get("border_crop")]
        if kind == "accurate":
            box = o.get("evaluator_target_box_xyxy")
            if box is None:
                continue
            click = ((box[0] + box[2]) / 2., (box[1] + box[3]) / 2.)
            if not any(p["box"][0] <= click[0] <= p["box"][2] and
                       p["box"][1] <= click[1] <= p["box"][3]
                       for p in candidates):
                continue
        else:
            wrong = next((p for p in candidates if
                          float(p.get("evaluator_target_iou") or 0.) < .1), None)
            if wrong is None:
                continue
            box = wrong["box"]
            click = ((box[0] + box[2]) / 2., (box[1] + box[3]) / 2.)
        return {"kind": kind, "stage": o["stage"], "frame": o["frame"],
            "rgb": relative(o["stage_folder"] / o["rgb_file"]),
            "depth": relative(o["stage_folder"] / o["depth_file"]),
            "capture_time_s": o["capture_time_s"], "click_xy": click,
            "proposals": o["proposals"]}
    return None


def select_from_click(click: tuple[float, float], proposals: list[dict]) -> dict | None:
    x, y = click
    possible = [p for p in proposals if p.get("actor_map_xy_m") is not None
        and not p.get("border_crop") and p["box"][0] <= x <= p["box"][2]
        and p["box"][1] <= y <= p["box"][3]]
    return min(possible, key=lambda p: (p["box"][2] - p["box"][0]) *
               (p["box"][3] - p["box"][1]), default=None)


def make_scene(case: dict, present: bool, *,
               range_only: bool = False) -> TexturedCaptureScene:
    cfg = SCENES[case["scene_key"]]
    source = json.loads(cfg["trace"].read_text())
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    scene = scene_from_capture_source(ROOT, source, scanned_person=human,
        person_facing_deg=cfg["person_facing_deg"],
        range_only=range_only)
    screen = json.loads((ROOT / case["screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x.get("scene_key") ==
                     case["scene_key"])
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    scene.set_props({**({case["target"]: (TexturedFloorObject(
        props / f"{case['target']}.obj", scale=.012),
        tuple(placement["find_target_xz"]), 0.)} if present else {}),
        case["distractor"]: (TexturedFloorObject(
            props / f"{case['distractor']}.obj", scale=.012),
            tuple(placement["find_distractor_xz"]), 0.)})
    return scene


def run_one(case: dict, arm: tuple[str, str, str], *,
            output_root: Path = OUT, vision_device: str | None = None) -> dict:
    name, branch, kind = arm
    folder = ROOT / case["source_folder"]
    output = output_root / case["scene_key"] / name
    output.mkdir(parents=True, exist_ok=True)
    prior = json.loads((folder / branch / "episode_trace.json").read_text())
    if prior["action"] != "unidentified_ask_new_view_or_help":
        raise ValueError("Owner point is only admissible after unknown identity")
    signal = owner_signal(folder, branch, kind)
    if signal is None:
        trace = {"schema": "ape.compound-owner-point-recovery.v1",
            "scene_key": case["scene_key"], "arm": name,
            "source_episode": relative(folder / branch / "episode_trace.json"),
            "owner_signal": None, "action": "ask_show_again_hold_unknown",
            "motion_target": None, "fresh_search": None}
    else:
        rgb_path = ROOT / signal["rgb"]
        depth_path = ROOT / signal["depth"]
        selected = select_from_click(tuple(signal["click_xy"]),
                                     signal["proposals"])
        if selected is None:
            raise ValueError("Scripted click has no actor-observed proposal")
        owner = json.loads((folder / "owner_approach/trace.json").read_text())
        pose = owner["trace"][-1]
        teach = json.loads((folder / "teach/trace.json").read_text())
        vision = colour_bank.baseline.Vision("generic-proposal",
                                              device=vision_device)
        vision._teaching_frame_seen = True
        image = np.asarray(Image.open(rgb_path).convert("RGB"))
        box = selected["box"]
        crop = Image.fromarray(image).crop(tuple(round(v) for v in box))
        exemplar = vision.feature([crop])[0]
        colour = colour_bank.baseline.colour_histogram(image, box)
        screen = json.loads((ROOT / case["screen"]).read_text())
        placement = next(x for x in screen["scenes"] if x.get("scene_key") ==
                         case["scene_key"])
        scene = make_scene(case, branch == "find_present")
        result = colour_bank.baseline.run_search(scene, vision,
            np.asarray(pose["evaluator_actual_xz_m"], float),
            float(pose["evaluator_actual_heading_deg"]),
            np.asarray(pose["actor_pose_xy_m"], float),
            float(pose["actor_heading_deg"]), exemplar,
            float(teach["acceptance_similarity"]), branch == "find_present",
            output / "fresh_search", start_time_s=float(
                pose["decision_time_s"] + 30.),
            search_xz=tuple(placement["find_target_xz"]),
            search_distractor_xz=tuple(placement["find_distractor_xz"]),
            owner_start_xz=tuple(SCENES[case["scene_key"]]["start_xz"]),
            target_name=case["target"],
            distractor_name=case["distractor"], identity_mode="multi_cue",
            colour_exemplar=colour,
            colour_acceptance=float(teach["colour_acceptance_similarity"]),
            search_photometry="cool_dim")
        signal = {k: v for k, v in signal.items() if k != "proposals"}
        signal.update(rgb_sha256=hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
            depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest())
        trace = {"schema": "ape.compound-owner-point-recovery.v1",
            "scene_key": case["scene_key"], "arm": name,
            "source_episode": relative(folder / branch / "episode_trace.json"),
            "owner_signal": signal,
            "selected_proposal": selected,
            "fresh_search": relative(output / "fresh_search/episode_trace.json"),
            "fresh_video": result["video"],
            "action": result["action"],
            "motion_target": result["motion_target"]}
        trace["vision_device"] = vision.device
    trace["limits"] = ("Owner click is simulator-scripted using object identity only"
        " to construct the reply; the actor sees the saved pixel and existing"
        " depth-localized proposals. New identity evidence is a separately"
        " rendered RGB-D search in the same static room. Owner yes/no is"
        " scripted. No physical person, moving hazard or articulated motion.")
    path = output / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": case["scene_key"], "arm": name,
        "trace": relative(path), "action": trace["action"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = case_rows()
    if len(cases) != 6 or len({c["scene_key"] for c in cases}) != 6:
        raise ValueError("Expected six whole-episode identity misses")
    frozen = {"schema": "ape.compound-owner-point-recovery-precommit.v1",
        "source_audits": [{"path": relative(p), "sha256": sha(p)} for p, _ in SOURCES],
        "cases": cases, "arms": [a[0] for a in ARMS],
        "policy": "For every unknown present-item episode, compare an accurate owner point, wrong point, and silence; for the matching absent item, compare a wrong point. Points reference a saved search RGB/depth pair. The actor chooses the smallest existing non-border depth-localized proposal containing the pixel, refreshes its exemplar, then re-renders and searches with unchanged multi-cue thresholds and scripted owner yes/no. No point leaves an unknown/no-motion state.",
        "limits": "Six development-observed failures across 20 exposed and three previously inspected captures; scripted owner inputs, static room and ideal sensor/body geometry."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Owner-point precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm[0] / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"], "arm": arm[0],
                    "trace": relative(path), "action": trace["action"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_one(case, arm)
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.compound-owner-point-recovery-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(row["scene_key"], row["arm"], row["action"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
