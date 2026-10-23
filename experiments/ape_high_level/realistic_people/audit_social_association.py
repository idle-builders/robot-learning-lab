"""Score obvious wrong-person choices after an Isaac social run.

This is evaluator-only: runtime character roots and camera poses never enter
the actor. A horizontal projection can expose a definite switch when the
people are well separated; overlapping/occluded boxes stay ambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def projected_x(person_xyz: list[float], eye_xyz: list[float],
                heading_deg: float, fov_deg: float, width: int = 640) -> float | None:
    heading = math.radians(heading_deg)
    dx, dy = person_xyz[0] - eye_xyz[0], person_xyz[1] - eye_xyz[1]
    depth = dx * math.sin(heading) + dy * math.cos(heading)
    if depth <= 0.05:
        return None
    right = dx * math.cos(heading) - dy * math.sin(heading)
    focal = width / (2 * math.tan(math.radians(fov_deg / 2)))
    return width / 2 + focal * right / depth


def audit(run_dir: Path, replay: Path | None = None) -> dict:
    evidence = run_dir / "evidence"
    inputs = json.loads((run_dir / "inputs.json").read_text())
    scene = json.loads((evidence / "scene-info.json").read_text())
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    replay_rows = json.loads(replay.read_text())["trace"] if replay else None
    motion = json.loads((evidence / "character-motion-trace.json").read_text())
    if len(actions) != len(motion):
        raise ValueError("Action and character traces differ in frame count")
    if replay_rows is not None and len(replay_rows) != len(actions):
        raise ValueError("Replay and executed trace differ in frame count")
    fov = inputs["control_fov_deg"]
    eye = scene["camera_eye_xyz_m"]
    heading = 0.0
    rows = []
    for index, (action, truth) in enumerate(zip(actions, motion)):
        frame = action["frame"]
        if frame != truth["frame"]:
            raise ValueError("Action and character frames are not aligned")
        actor = replay_rows[index] if replay_rows is not None else action["actor"]
        if replay_rows is not None and actor["frame"] != frame:
            raise ValueError("Replay and executed frames are not aligned")
        people = truth["characters"]
        owner_x = projected_x(people[0]["xyz_m"], eye, heading, fov)
        bystander_x = projected_x(people[1]["xyz_m"], eye, heading, fov)
        # A diagnostic best box in a hold/retry is not an accepted owner
        # association and must not be scored as a wrong-person binding.
        actionable = actor.get("intent") in (
            "look_to_owner", "approach_bounded", "hold_near_person")
        choice = actor.get("chosen_proposal") if actionable else None
        proposals = actor.get("proposals", [])
        box = proposals[choice]["box_xyxy"] if choice is not None and choice < len(proposals) else None
        if box is None and actionable and actor.get("observation_method") == "rgbd_flow":
            box = actor.get("track", {}).get("box_xyxy")
        separate_proposals = False
        if (owner_x is not None and bystander_x is not None and
                0 <= owner_x < 640 and 0 <= bystander_x < 640):
            owner_boxes = [index for index, proposal in enumerate(proposals)
                           if proposal["box_xyxy"][0] <= owner_x <= proposal["box_xyxy"][2]
                           and not proposal["box_xyxy"][0] <= bystander_x <= proposal["box_xyxy"][2]]
            bystander_boxes = [index for index, proposal in enumerate(proposals)
                               if proposal["box_xyxy"][0] <= bystander_x <= proposal["box_xyxy"][2]
                               and not proposal["box_xyxy"][0] <= owner_x <= proposal["box_xyxy"][2]]
            separate_proposals = any(a != b for a in owner_boxes for b in bystander_boxes)
        label = "no_choice"
        # A person behind the camera has no forward projection and cannot
        # explain a visible box; still score the visible person's root.
        if box is not None and owner_x is not None and bystander_x is None:
            x0, _, x1, _ = box
            if x0 <= owner_x <= x1:
                label = "owner"
        elif box is not None and bystander_x is not None and owner_x is None:
            x0, _, x1, _ = box
            if x0 <= bystander_x <= x1:
                label = "bystander"
        if box is not None and owner_x is not None and bystander_x is not None:
            x0, _, x1, _ = box
            center = (x0 + x1) / 2
            separation = abs(owner_x - bystander_x)
            if separation < 80:
                label = "projection_overlap"
            elif x0 <= owner_x <= x1 and x0 <= bystander_x <= x1:
                label = "mixed_box"
            elif abs(center - owner_x) + 45 < abs(center - bystander_x):
                label = "owner"
            elif abs(center - bystander_x) + 45 < abs(center - owner_x):
                label = "bystander"
            else:
                label = "ambiguous_projection"
        row = {"frame": frame, "actor_intent": actor["intent"],
               "owner_root_x_px_evaluator_only": owner_x,
               "bystander_root_x_px_evaluator_only": bystander_x,
               "chosen_box_xyxy": box, "association_evaluator_only": label,
               "both_projected_in_fov": owner_x is not None and bystander_x is not None
               and 0 <= owner_x < 640 and 0 <= bystander_x < 640,
               "two_separate_person_proposals_evaluator_only": separate_proposals}
        rows.append(row)
        eye = action["proxy_eye_xyz_m_evaluator_only"]
        heading = action["proxy_heading_deg_evaluator_only"]
    counts = {label: sum(row["association_evaluator_only"] == label for row in rows)
              for label in ("owner", "bystander", "mixed_box", "projection_overlap",
                            "ambiguous_projection", "no_choice")}
    return {"scope": "horizontal root-projection and separate detector-box audit only; no person mask or identity truth fed to actor",
            "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "run": str(run_dir), "replay": str(replay) if replay else None,
            "fov_deg": fov, "frames": len(rows),
            "both_projected_in_fov_frames": sum(row["both_projected_in_fov"] for row in rows),
            "two_separate_person_proposal_frames": sum(
                row["two_separate_person_proposals_evaluator_only"] for row in rows),
            "counts": counts, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir, args.replay)
    out = args.run_dir / ("ssdlite-association-audit.json" if args.replay else
                          "social-association-audit.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("frames", "both_projected_in_fov_frames",
                                                    "two_separate_person_proposal_frames", "counts")},
                     indent=2))


if __name__ == "__main__":
    main()
