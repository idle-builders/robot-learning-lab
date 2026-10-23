"""Bounded RGB-D head-view sweep after a moving-bystander reset.

This continues the V586 joined episode at its actual stopped pose. The actor
aims from its saved anonymous range track, tries finite pitch/FOV views, and
still withholds owner identity. Evaluator person labels are absent from its
choice and used only later for audit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import owner_depth_change_component_v516 as depth_change
import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_person_interaction as interaction
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_two_moving_people_v586 as moving
from screen_robocasa_owner_safe_stop_v494 import aim_camera
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-post-retreat-view-sweep-v587"
VIEWS = (("owner_wide20", 15), ("owner_wide20", 20),
         ("owner_wide20", 25), ("owner_up25", 15),
         ("owner_up25", 20), ("owner_up25", 25))
SCAN = height_scan((.30,))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    policy = json.loads(joined.CONFIG.read_text())
    case = next(x for x in policy["cases"] if x["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    original = turn.owner_turn
    original_out = moving.OUT
    original_margin = route.MARGIN_M
    original_hand = interaction.SHOW_HAND.copy()
    original_first_m = turn.FIRST_RETURN_OWNER_M
    original_first_s = turn.FIRST_RETURN_OWNER_S
    original_late = turn.LATE_OWNER_APPROACH_M
    original_second = turn.SECOND_RETURN_OWNER_M
    original_owner = turn.OWNER_APPROACH_M
    captured = []

    def sweep_turn(response, model, data, renderer, actual, estimate,
                   time_s, use_depth, vision, exemplar):
        outcome, rows, frames, wall, error, t = original(
            response, model, data, renderer, actual, estimate,
            time_s, use_depth, vision, exemplar)
        if outcome != "owner_interaction_reset_after_reacquire_motion":
            return outcome, rows, frames, wall, error, t
        tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
        target = np.asarray(max(tracks, key=lambda row: row["hits"])["center_xy_m"])
        healthy_count = 0
        for index, (camera, pitch) in enumerate(VIEWS):
            t += .2
            turn._prop(model, t)
            aim_camera(model, camera, estimate[:2], target, pitch)
            model.cam_pos[model.camera(camera).id, :2] = actual
            mujoco.mj_forward(model, data)
            rgb, depth, age = owner.capture(renderer, data, camera, True)
            t += max(0., age - .2)
            healthy, fractions = depth_change.packet_health(depth, depth)
            healthy_count += int(healthy)
            packet = SCAN(model, data, actual, 5.,
                          np.random.default_rng(22000 + index))
            raw = OUT / f"view-{index:02d}.npz"
            np.savez_compressed(raw, rgb=rgb, depth=depth)
            image = OUT / f"view-{index:02d}.png"
            Image.fromarray(rgb).save(image)
            record = {"stage": "post_retreat_active_view", "time_s": round(t, 4),
                      "index": index, "camera": camera,
                      "fovy_degrees": float(model.cam_fovy[model.camera(camera).id]),
                      "pitch_degrees": pitch,
                      "aimed_at_anonymous_track_xy_est_m": target.tolist(),
                      "ape_xy_m": actual.tolist(),
                      "capture_wall_s": age,
                      "depth_central_valid_fraction": fractions[0],
                      "depth_healthy": healthy,
                      "mid_height_range_returns": sum(x is not None for x in packet),
                      "raw_rgbd": {"path": str(raw), "sha256": sha(raw)},
                      "image": {"path": str(image), "sha256": sha(image)},
                      "identity_state": "unknown_no_show"}
            captured.append(record)
            rows.append(record)
            frames.append(owner.owner_frame(rgb,
                          f"fresh view {camera} up{pitch}: {'depth OK' if healthy else 'depth weak'}",
                          None, None))
        return (("owner_identity_unknown_healthy_view" if healthy_count
                 else "owner_identity_unknown_no_healthy_view"),
                rows, frames, wall, error, t)

    try:
        moving.OUT = OUT
        turn.owner_turn = sweep_turn
        route.MARGIN_M = .15
        interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        turn.OWNER_APPROACH_M = 0.
        turn.FIRST_RETURN_OWNER_M = .4
        turn.FIRST_RETURN_OWNER_S = 1.
        turn.LATE_OWNER_APPROACH_M = 0.
        turn.SECOND_RETURN_OWNER_M = 0.
        result = moving.run_trial(case, prior, policy, vision, exemplar,
                                  bystander=True)
    finally:
        turn.owner_turn = original
        moving.OUT = original_out
        route.MARGIN_M = original_margin
        interaction.SHOW_HAND = original_hand
        turn.OWNER_APPROACH_M = original_owner
        turn.FIRST_RETURN_OWNER_M = original_first_m
        turn.FIRST_RETURN_OWNER_S = original_first_s
        turn.LATE_OWNER_APPROACH_M = original_late
        turn.SECOND_RETURN_OWNER_M = original_second
    report = {"scope": "V586 full two-moving-person joined route continued after actual owner-association reset with six bounded exploratory RGB-D pitch/FOV views at held pose; no identity or show claim; scripted motion and kinematic ape",
              "source_sha256": sha(Path(__file__)),
              "base_moving_runner_sha256": sha(Path(moving.__file__)),
              "turn_source_sha256": sha(Path(original.__code__.co_filename)),
              "scene_xml_sha256": sha(moving.SCENE_XML),
              "views": captured, "result": result}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("outcome", result["episode"]["outcome"],
          "healthy", sum(x["depth_healthy"] for x in captured),
          "video", result["episode"]["video"]["path"], flush=True)


if __name__ == "__main__":
    main()
