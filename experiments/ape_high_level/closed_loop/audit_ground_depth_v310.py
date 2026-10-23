"""V310: floor-height separation on the same noisy Direct Frame side frames.

Diagnostic only: compare fixed world-Z cuts after a 4cm self-depth mask with
the moving-prop action labels. The fixed ground plane is an idealization and
cannot certify a route on a tilted body, uneven floor, or unobserved space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_localization_v297 import depth_variant
from audit_lateral_clutter_depth_v285 import project
from audit_lateral_clutter_sensors_v283 import actor_gap
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from run_heldout_moving_low_v305 import CASES, HeldoutMovingRoom
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v310"
V308 = ROOT / "artifacts/ape-closed-loop-cove50-v308/moving-depth-action-audit.json"
V309 = ROOT / "artifacts/ape-closed-loop-cove50-v309/self-depth-noise-audit.json"
HEIGHT_CUTS_M = (.02, .04, .05, .06, .07)
MARGIN_M = .065


def main() -> None:
    truth = {r["case"][0]: r for r in json.loads(V308.read_text())["rows"]}
    labels = {r["case"][0]: r for r in json.loads(V309.read_text())["rows"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        seed, mode, *_ = case
        room = HeldoutMovingRoom(case=case, pose_source="command", branch=OUT)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H,
                             agreement_m=.04)
        try:
            room.layout(seed, absent=mode != "present", furnished=True)
            room.time_s = 7.6
            room.actual[:] = (0., .0464, 0.)
            room.estimated[:] = (0., .08, 0.)
            room.head = 1.2
            room._update_moving_obstacle()
            room._forward()
            _, ideal = room.capture("lower_rgb", depth=True)
            if ideal is None:
                raise AssertionError("Rendered depth unavailable")
            measured = depth_variant(ideal, "short", seed, 22)
            rng = np.random.default_rng(seed * 1009 + 41 + 12)
            measured += rng.normal(0., .01, measured.shape).astype(measured.dtype)
            measured[rng.random(measured.shape) < .15] = np.nan
            camera_pose = room.camera_pose_estimate("lower_rgb")
            actor_qpos = room.pose_data.qpos.copy()
            filtered, _ = mask.apply(measured, actor_qpos, max_depth_m=2.5)
            fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
            points = project(filtered, camera_pose, fovy)
            # The actor would restore head yaw to zero before the side step.
            actor_qpos[room.head_addr] = 0.
            variants = {}
            for height in HEIGHT_CUTS_M:
                cloud = points[points[:, 2] >= height]
                gap = float(actor_gap(robot, actor_qpos, room.estimated.copy(),
                                      cloud, 1))
                variants[str(height)] = {
                    "point_count": len(cloud),
                    "actor_gap_m": None if not np.isfinite(gap) else round(gap, 5),
                    "actor_pass": bool(gap >= MARGIN_M)}
            old_gap = labels[seed]["alternatives"]["0.04"]["actor_point_gap_m"]
            if abs(variants["0.02"]["actor_gap_m"] - old_gap) > 1e-5:
                raise AssertionError("Frozen baseline depth points changed")
            rows.append({"case": case,
                         "evaluator_safe_5cm": truth[seed]["evaluator_safe_5cm"],
                         "evaluator_min_dynamic_clearance_m": truth[seed][
                             "evaluator_min_dynamic_clearance_m"],
                         "variants": variants})
            print(seed, mode, "safe", truth[seed]["evaluator_safe_5cm"],
                  {k: (v["actor_pass"], v["point_count"])
                   for k, v in variants.items()}, flush=True)
        finally:
            mask.close()
            room.close()
    scores = {}
    for height in HEIGHT_CUTS_M:
        key = str(height)
        scores[key] = {
            "actions": len(rows),
            "unsafe_actions": sum(not r["evaluator_safe_5cm"] for r in rows),
            "false_safe": sum(r["variants"][key]["actor_pass"] and
                              not r["evaluator_safe_5cm"] for r in rows),
            "false_block": sum(not r["variants"][key]["actor_pass"] and
                               r["evaluator_safe_5cm"] for r in rows),
            "passed": sum(r["variants"][key]["actor_pass"] for r in rows)}
    report = {"scope": "V310 same six frozen V307 t7.6s side-looking +1.2rad lower short-depth captures with identical 1cm Gaussian/15% holes, actor estimated pose and 4cm self mask; compare fixed world-Z cuts 2/4/5/6/7cm before the same 6.5cm point-distance gate against V308 independent moving full-body 5cm action labels. Actor does not receive prop masks or truth. Fixed world-Z ground and head actuation-free snapshot are diagnostics only; sensor mounting, ground estimation under tilt, occlusion/free-space coverage, moving-obstacle prediction and native braking remain unqualified.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "v308_sha256": sha(V308), "v309_sha256": sha(V309),
              "height_cuts_m": HEIGHT_CUTS_M,
              "rows": rows, "scores": scores}
    (OUT / "ground-depth-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
