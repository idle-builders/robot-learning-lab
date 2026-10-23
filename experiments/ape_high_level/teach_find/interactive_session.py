"""Teach once, correct a displayed RGB frame, then search fresh scenes.

This is a scripted human-in-the-loop *simulation* using the same released image
models as active_scan. The correction is a supplied pixel click on a frame the
ape actually captured. It replaces the prior appearance example. New searches
use that memory without showing the teaching scene again. They turn toward and
reobserve their best candidate, then ask for confirmation because identity
scores are not calibrated for a present/absent declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .active_scan import (Vision, _ground_point, _iou,
                              _search_positions, run_episode)
    from .scene import HeadScene, HEIGHT, OBJECTS, WIDTH
except ImportError:
    from active_scan import (Vision, _ground_point, _iou,
                             _search_positions, run_episode)
    from scene import HeadScene, HEIGHT, OBJECTS, WIDTH


def _captured_frame(scene: HeadScene, path: Path) -> Image.Image:
    image = Image.fromarray(scene.image())
    image.save(path)
    return image


def _proposals(image: Image.Image, vision: Vision, exemplar: np.ndarray,
               pose: tuple[np.ndarray, np.ndarray]) -> list[dict]:
    proposals = vision.detect(image)
    if not proposals:
        return []
    crops = [image.crop(tuple(round(v) for v in p["box"])) for p in proposals]
    features = vision.feature(crops)
    for proposal, feature in zip(proposals, features):
        box = proposal["box"]
        proposal["similarity"] = float(feature @ exemplar)
        proposal["floor_xy_m"] = _ground_point(box, pose)
        proposal["border_crop"] = box[0] < 8 or box[2] > WIDTH - 8
    return proposals


def scan_after_correction(scene: HeadScene, vision: Vision, target: str,
                          absent: bool, seed: int, exemplar: np.ndarray,
                          out: Path) -> dict:
    """Image-driven yaw scan and candidate-centric reobservation, no reteach."""
    out.mkdir(parents=True, exist_ok=True)
    positions = _search_positions(target, absent, seed)
    rng = np.random.default_rng(seed + OBJECTS.index(target) * 41)
    headings = {name: float(rng.uniform(-.3, .3)) for name in OBJECTS}
    scene.place(positions, headings)
    pending_yaws = [0.0]
    seen_yaws = set()
    observations = []
    while pending_yaws and len(seen_yaws) < 4:
        yaw = pending_yaws.pop(0)
        if round(yaw, 2) in seen_yaws:
            continue
        seen_yaws.add(round(yaw, 2))
        scene.look(yaw, .75)
        path = out / f"yaw{yaw:+.2f}.png"
        image = _captured_frame(scene, path)
        proposals = _proposals(image, vision, exemplar, scene.camera_pose())
        observations.append({"requested_action": "look_and_capture",
                             "yaw_rad": yaw, "pitch_rad": .75,
                             "rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
                             "camera_xyz_m": scene.camera_pose()[0].tolist(),
                             "proposals": proposals})
        if len(seen_yaws) == 1:
            borders = [p for p in proposals if p["border_crop"]]
            if borders:
                edge = max(borders, key=lambda p: p["detector_score"])
                center = (edge["box"][0] + edge["box"][2]) / 2
                pending_yaws.append(.7 if center < WIDTH / 2 else -.7)
            pending_yaws.extend([-.7, .7])

    candidates = [(o["yaw_rad"], p) for o in observations for p in o["proposals"]
                  if not p["border_crop"] and p["floor_xy_m"] is not None]
    candidates.sort(key=lambda pair: pair[1]["similarity"], reverse=True)
    best = candidates[0] if candidates else None
    verification = None
    action = "ask_to_show_again" if best is None else "ask_owner_to_confirm_candidate"
    if best is not None:
        yaw, proposal = best
        center = (proposal["box"][0] + proposal["box"][2]) / 2
        focal = HEIGHT / (2 * math.tan(math.radians(67) / 2))
        centered_yaw = float(np.clip(yaw - math.atan((center - WIDTH / 2) / focal),
                                     -1.5, 1.5))
        scene.look(centered_yaw, .75)
        image = _captured_frame(scene, out / "candidate-reobserve.png")
        found = _proposals(image, vision, exemplar, scene.camera_pose())
        # The confirmation view must remain tied to the same floor candidate;
        # a higher-scoring lookalike elsewhere in the new image is not a match.
        nearby = [p for p in found if p["floor_xy_m"] is not None and
                  np.linalg.norm(np.subtract(p["floor_xy_m"], proposal["floor_xy_m"])) < .12]
        if nearby:
            chosen = min(nearby, key=lambda p: np.linalg.norm(
                np.subtract(p["floor_xy_m"], proposal["floor_xy_m"])))
            verification = {"requested_action": "reobserve_candidate",
                            "yaw_rad": centered_yaw, "box": chosen["box"],
                            "floor_xy_m": chosen["floor_xy_m"],
                            "similarity": chosen["similarity"],
                            "rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest()}
        else:
            action = "ask_to_show_again"

    # Only the scorer sees simulator segmentation, after actions are fixed.
    for observation in observations:
        scene.look(observation["yaw_rad"], observation["pitch_rad"])
        truth = scene.evaluator_box(target)
        for proposal in observation["proposals"]:
            proposal["evaluator_target_iou"] = _iou(proposal["box"], truth)
    if verification is not None:
        scene.look(verification["yaw_rad"], .75)
        verification["evaluator_target_iou"] = _iou(
            verification["box"], scene.evaluator_box(target))
    return {"target": target, "seed": seed, "target_absent": absent,
            "memory_status": "corrected_view_unvalidated",
            "observations": observations, "candidate_count": len(candidates),
            "best_candidate": {"yaw_rad": best[0], "box": best[1]["box"],
                               "similarity": best[1]["similarity"],
                               "floor_xy_m": best[1]["floor_xy_m"]} if best else None,
            "verification": verification, "action": action,
            "provisional_candidate_correct": bool(verification and
                                                  verification["evaluator_target_iou"] >= .25),
            "declared_found": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--teach-clicks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[5, 7, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    correction = json.loads(args.correction.read_text())
    target = correction["target"]
    click = correction["click_px"]
    yaw = float(correction["yaw_rad"])
    source_dir = Path(correction["source"])
    if not source_dir.name.startswith("seed") or not source_dir.name[4:].isdigit():
        raise ValueError("Correction source directory must end in seed<number>")
    source_seed = int(source_dir.name[4:])
    clicks = json.loads(args.teach_clicks.read_text())["clicks_px"]
    if source_seed in args.seeds:
        raise ValueError("New search seeds must differ from correction source seed")
    scene, vision = HeadScene(), Vision("generic-proposal")
    try:
        source_out = args.output / "initial-search"
        source_out.mkdir(exist_ok=True)
        initial = run_episode(scene, vision, target, False, source_seed, source_out, clicks)
        observation = next(o for o in initial["observations"]
                           if abs(o["yaw_rad"] - yaw) < 1e-6 and o["camera_healthy"])
        frame = Image.open(source_out / f"{target}-moved-yaw{yaw:+.2f}.png").convert("RGB")
        selected, selection_info = vision.select_click(frame, click, observation["proposals"])
        if selected is None:
            raise ValueError(f"Correction click missed all proposals: {selection_info}")
        exemplar = vision.feature([frame.crop(selected)])[0]
        correction_event = {"event": "scripted_visual_correction", "name": target,
                            "source_yaw_rad": yaw, "click_px": click,
                            "click_source": correction.get("source_of_click"),
                            "selected_box": selected, "selection": selection_info,
                            "rgb_sha256": hashlib.sha256(np.asarray(frame).tobytes()).hexdigest(),
                            "old_view_status": "contradicted",
                            "new_view_status": "requires_present_absent_validation"}
        episodes = [scan_after_correction(scene, vision, target, absent, seed,
                                          exemplar, args.output / f"seed{seed}-"
                                          f"{'absent' if absent else 'moved'}")
                    for seed in args.seeds for absent in (False, True)]
    finally:
        scene.close()

    result = {"scope": "Scripted text name and correction click; synthetic fixed-base 50 cm "
                       "head rig with released RGB models; same HOPE assets across new scenes; "
                       "exact simulated camera pose; no person, physical sensor, moving base, "
                       "grasp or calibrated identity threshold.",
              "interaction": [
                  {"event": "teach_name_and_select_crop", "input": "text_and_click_fixtures",
                   "name": target, "click_px": clicks[target],
                   "action": "search_using_selected_crop"},
                  {"event": "initial_search", "seed": source_seed, "action": initial["action"],
                   "healthy_views": sum(o["camera_healthy"] for o in initial["observations"])},
                  correction_event,
                  *[{"event": "search_new_scene", "seed": e["seed"],
                     "target_absent": e["target_absent"], "action": e["action"],
                     "candidate_correct": e["provisional_candidate_correct"]}
                    for e in episodes]],
              "initial_search": initial, "searches_after_correction": episodes}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    for event in result["interaction"]:
        print(event)


if __name__ == "__main__":
    main()
