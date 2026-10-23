"""V315: isolate side-look time and fresh shin recheck from depth veto.

This uses the same actuated +1.2rad look and post-look shin scan as V314 but
does not use the lower-depth measurement to block or authorize movement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import run_lateral_clutter_shin_search_v284 as search
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, Vision, sha
from run_active_side_probe_v314 import CASES, MAX_RANGE_AGE_S, MAX_WAITS
from run_heldout_moving_low_v305 import HeldoutMovingRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v315"
V314 = ROOT / "artifacts/ape-closed-loop-cove50-v314/active-side-probe-audit.json"


def main() -> None:
    paired = json.loads(V314.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        branch = OUT / f"seed{seed}-{mode}" / f"seed{seed}-{mode}"
        branch.mkdir(parents=True, exist_ok=True)
        rooms = []
        probes = []

        def factory(requested_seed: int, requested_mode: str):
            if (requested_seed, requested_mode) != (seed, mode):
                raise ValueError("Unexpected room request")
            room = HeldoutMovingRoom(case=case, pose_source="command",
                                     branch=branch)
            room.layout(seed, absent=mode != "present", furnished=True)
            rooms.append(room)
            return room

        def timing_probe(room, robot, target_branch: Path, video,
                         direction: int, step: int):
            del robot, direction
            room.look(1.2, "lower_rgb")
            rgb, _ = room.capture("lower_rgb", depth=False)
            source = target_branch / f"timing-side-look-{len(probes):02d}.png"
            rgb.save(source)
            panel = Image.fromarray(np.concatenate((np.asarray(rgb),
                np.asarray(room.capture_external())), axis=1))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, panel.width, 30), fill=(0, 0, 0))
            draw.text((8, 7),
                f"timing-only side look | seed {seed} step {step} | "
                f"t{room.time_s:.1f} | fresh shin recheck next",
                fill=(255, 255, 255))
            video.append_data(np.asarray(panel))
            entry = {"allowed": not room.motion_interrupted,
                     "capture_time_s": round(room.time_s, 4),
                     "actor_rgb_path": str(source.relative_to(ROOT)),
                     "actor_rgb_sha256": sha(source)}
            probes.append(entry)
            return entry

        result = search.run_case(
            seed, mode, vision, exemplar, owner_hist,
            room_factory=factory,
            output_root=branch.parent,
            # Match the active-depth branch's deterministic shin noise for
            # each corresponding scan serial; only the depth veto differs.
            sensor_case_prefix=f"v314-{seed}-active_side_depth",
            max_clear_side_waits=MAX_WAITS,
            max_range_age_s=MAX_RANGE_AGE_S,
            pre_step_probe=timing_probe)
        room = rooms[0]
        trace = branch / "episode-trace.json"
        trace.write_text(json.dumps({
            "case": case, "probe_inputs": probes,
            "controller_events": result["events"],
            "slips_evaluator_only": room.applied_slips,
            "evaluator_path_min_clearance": room.path_min_clearance,
            "evaluator_motion_interrupted": room.motion_interrupted,
            "final_time_s": room.time_s}, indent=2) + "\n")
        row = {"case": case, "condition": "timing_only_side_look",
               "status": result["status"],
               "owner_questions": result["owner_questions"],
               "motion_steps_requested": len(room.applied_slips),
               "probe_count": len(probes),
               "wait_rechecks": sum(e["action"] == "lateral_sector_recheck"
                                    for e in result["events"]),
               "evaluator_path_min_clearance_m": round(
                   room.path_min_clearance[0], 5),
               "evaluator_motion_interrupted": room.motion_interrupted,
               "trace_path": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace),
               "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
               "video_sha256": result["video_sha256"]}
        rows.append(row)
        (OUT / "timing-control-audit.json").write_text(
            json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
        print(seed, row["status"], row["evaluator_path_min_clearance_m"],
              row["motion_steps_requested"], row["probe_count"], flush=True)
    report = {"scope": "V315 matched live control for V314 four revealed approaching-prop Direct Frame owner scenes. A +1.2rad lower RGB side look consumes the same 0.4s simulated head action, then the controller takes a new noisy four-shin scan. The side image is saved but no depth points or other side-camera pixels enter the route decision. Any safety change shared with V314 is therefore attributable to look timing/fresh reobservation, not the depth veto. The room still uses a kinematic base, synthetic moving prop, ideal head action and no qualified native stop envelope.",
              "source_sha256": sha(SOURCE),
              "controller_sha256": sha(Path(search.__file__)),
              "selected_visual_sha256": sha(selected_visual()),
              "v314_sha256": sha(V314),
              "rows": rows}
    (OUT / "timing-control-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
