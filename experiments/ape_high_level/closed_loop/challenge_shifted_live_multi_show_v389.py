"""V389: frozen V384 owner-help tracker under shifted/clothed/lit people.

Two previously defined V346 person variants are crossed with two task seeds.
Only present-can episodes run; the V384 candidate/motion/freshness rules stay
unchanged. Evaluator scores item source and person/scene gaps afterward.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from articulated_person_scene import (ITEM_SIDE_OFFSET_M,
                                      START_HAND_XYZ_M, add_person)
from challenge_independent_hand_show_v374 import hand_offset
from challenge_person_belief_v346 import POSITIVE_VARIANTS
from challenge_raised_replica_show_v381 import REPLICA_XYZ_M
import run_blocked_owner_help_v317 as owner_module
import run_live_moving_distant_help_v373 as base
import run_live_multi_candidate_show_v384 as live
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v389"
VARIANT_NAMES = ("left_dim", "right_bright")


def evaluate(report: dict, variant: dict) -> list[dict]:
    offset = np.asarray(variant["offset_xyz_m"], dtype=float)
    rows = []
    for entry in report["owner_show_rows"]:
        response = entry["response"]
        samples = response["actor_samples"]
        start = float(samples[0]["capture_time_s"])
        classes = []
        for sample in samples:
            point = sample["actor_choice"]["accepted_world_xyz_m"]
            if point is None:
                classes.append("none")
                continue
            elapsed = float(sample["capture_time_s"]) - start
            hand = START_HAND_XYZ_M + ITEM_SIDE_OFFSET_M + offset
            if elapsed >= .3:
                hand = hand + hand_offset("hand_faster", elapsed - .3)
            position = np.asarray(point)
            classes.append("moving_held_item" if np.linalg.norm(
                position - hand) < np.linalg.norm(
                    position - REPLICA_XYZ_M)
                else "static_raised_replica")
        rows.append({
            "variant": variant["name"], "seed": entry["seed"],
            "status": response["status"],
            "actor_track_fresh_at_answer": response[
                "actor_track_fresh_at_answer"],
            "evaluator_selected_held_frames": classes.count(
                "moving_held_item"),
            "evaluator_selected_replica_frames": classes.count(
                "static_raised_replica"),
            "evaluator_final_selected_class": classes[-1],
            "evaluator_min_person_gap_m": response[
                "evaluator_min_person_gap_m"],
            "evaluator_min_scene_gap_m": response[
                "evaluator_min_scene_gap_m"],
        })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sources = []
    for name in VARIANT_NAMES:
        variant = next(row for row in POSITIVE_VARIANTS
                       if row["name"] == name)
        offset = np.asarray(variant["offset_xyz_m"], dtype=float)
        original_scene = live.staged_scene
        original_out = live.OUT
        original_add_person = base.add_person
        original_cases = owner_module.CASES

        def shifted_scene(room, start_s: float, elapsed_s: float,
                          carried_item: str | None) -> None:
            original_scene(room, start_s, elapsed_s, carried_item)
            room.model.body_pos[room.model.body("loop_person").id] = offset
            if carried_item is not None:
                room.model.body_pos[room.model.body(carried_item).id] += offset
            if not hasattr(room, "_v389_base_light"):
                room._v389_base_light = room.model.light_diffuse.copy()
            room.model.light_diffuse[:] = (
                room._v389_base_light * variant["light_scale"])
            room._forward()

        try:
            live.staged_scene = shifted_scene
            live.OUT = OUT / name
            base.add_person = lambda model: add_person(
                model, shirt_rgba=variant["shirt_rgba"])
            owner_module.CASES = tuple(case for case in original_cases
                                      if case[1] == "present")
            live.main()
        finally:
            live.staged_scene = original_scene
            live.OUT = original_out
            base.add_person = original_add_person
            owner_module.CASES = original_cases
        source_path = OUT / name / "blocked-owner-help-audit.json"
        report = json.loads(source_path.read_text())
        sources.append({"variant": name,
                        "report_path": str(source_path.relative_to(ROOT)),
                        "report_sha256": sha(source_path)})
        scored = evaluate(report, variant)
        rows.extend(scored)
        for row in scored:
            print(name, row["seed"], row["status"], "held", row[
                "evaluator_selected_held_frames"], "replica", row[
                "evaluator_selected_replica_frames"], "person", row[
                "evaluator_min_person_gap_m"], flush=True)
    result = {
        "scope": "Frozen V384 all-candidate DINO+hue+noisy-long-depth moving-item tracker and V373 guarded blocked-route owner-help on the V346 predeclared left/dim and right/bright articulated-person variants crossed with seeds170/172. Owner shirt, root shift and light change, while the held can follows the shifted hand and an identical raised replica stays at the same box. No thresholds or appearance weights change. Actor-selected item source and 20ms person/scene gaps are evaluator-only. The box/replica still appear at help onset; native head/audio/hand dynamics, realistic optics and broader room layouts remain absent.",
        "source_sha256": sha(SOURCE),
        "v384_source_sha256": sha(Path(live.__file__)),
        "v346_variant_source_sha256": sha(Path(
            ROOT / "experiments/ape_high_level/closed_loop/challenge_person_belief_v346.py")),
        "variant_reports": sources,
        "rows": rows,
    }
    (OUT / "shifted-live-multi-show.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
