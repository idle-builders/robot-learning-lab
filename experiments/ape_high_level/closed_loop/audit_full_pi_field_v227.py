"""V227: replay V211 optics with the Pi Wide's full horizontal field."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import audit_deployable_head_depth_v211 as v211
from run import H, ROOT, sha
from run_moved_props_v134 import MovedPropRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v227"
WIDTH = 896  # With 480 rows and 67 deg vertical, horizontal FOV is 102.03 deg.
BASELINE = v211.OUT / "head-depth-optical-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v211.W = WIDTH
    scenes = json.loads(v211.V203.read_text())["rows"]
    baseline = json.loads(BASELINE.read_text())
    old_rows = {(r["seed"], r["absent"], r["head_yaw_rad"]): r
                for r in baseline["rows"]}
    rows = []
    for scene in scenes:
        if scene["label"] != "generous_kinematic":
            continue
        seed = scene["seed"]
        positions = scene["audits"]["0.065"]["positions_xy_m"]
        for absent in (False, True):
            layout = {"CreamCheese": tuple(positions["CreamCheese"]),
                      "Parmesan": tuple(scene["absent_challenge"]["parmesan_xy_m"]
                                         if absent else positions["Parmesan"])}
            room = MovedPropRoom(layout)
            try:
                room.reset()
                room.layout(seed, absent=absent, furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40, (1 if seed % 2 else -1) * .46, 0.)
                room.set_table_height(.37)
                room._forward()
                room.renderer.close()
                room.renderer = mujoco.Renderer(room.model, width=WIDTH, height=H)
                branch = v211.V204 / f"seed{seed}-light1.00-{'absent' if absent else 'present'}"
                paths = sorted(branch.glob("lower-angle-*.npz"))
                assert len(paths) == 5
                for path in paths:
                    with np.load(path) as z:
                        qpos = z["actor_qpos"].copy()
                    room.data.qpos[:] = qpos
                    room.actual[:] = 0.
                    room.head = float(qpos[room.head_addr])
                    room._forward()
                    down_rgb, down_depth = room.capture("lower_rgb", depth=True)
                    _, up_depth = room.capture("upper_rgb", depth=True)
                    assert down_depth is not None and up_depth is not None
                    target = v211.segment(room, "TomatoSauce")
                    pair = v211.stereo(room, down_depth, up_depth, target)
                    target_depth = down_depth[target]
                    valid_target = np.isfinite(target_depth) & (target_depth > .02)
                    metric = target_depth[valid_target]
                    yaw = round(room.head, 3)
                    old = old_rows[(seed, absent, yaw)]
                    image_path = OUT / f"seed{seed}-{'absent' if absent else 'present'}-yaw{yaw:+.1f}-lower-full-field.png"
                    down_rgb.save(image_path)
                    ys, xs = np.where(target)
                    truth_box = ([int(xs.min()), int(ys.min()), int(xs.max() + 1),
                                  int(ys.max() + 1)] if len(xs) else None)
                    row = {"seed": seed, "absent": absent, "head_yaw_rad": yaw,
                           "source_frame_sha256": sha(path),
                           "full_field_rgb_file": image_path.name,
                           "full_field_rgb_sha256": sha(image_path),
                           "full_field_true_can_box_evaluator_only": truth_box,
                           "full_field_stereo": pair,
                           "cropped_field_stereo": old["stereo"],
                           "full_field_can_visible_pixels_evaluator_only": int(valid_target.sum()),
                           "cropped_field_can_visible_pixels_evaluator_only":
                               old["true_can_visible_pixels_evaluator_only"],
                           "full_field_can_short_depth_pixels_evaluator_only": int(
                               ((metric >= .07) & (metric <= .5)).sum()),
                           "cropped_field_can_short_depth_pixels_evaluator_only":
                               old["true_can_pixels_d405_ideal_7_to_50cm_evaluator_only"]}
                    rows.append(row)
                    print(seed, absent, yaw,
                          pair["geometric_shared_pixels"],
                          row["full_field_can_visible_pixels_evaluator_only"], flush=True)
            finally:
                room.close()
    present = [r for r in rows if not r["absent"]]
    summary = {
        "paired_views": len(rows),
        "full_field_lower_valid_sampled_pixels": sum(
            r["full_field_stereo"]["lower_valid_sampled_pixels"] for r in rows),
        "full_field_stereo_geometric_shared_pixels": sum(
            r["full_field_stereo"]["geometric_shared_pixels"] for r in rows),
        "full_field_stereo_depth_consistent_shared_pixels": sum(
            r["full_field_stereo"]["upper_depth_consistent_shared_pixels"] for r in rows),
        "full_field_present_target_visible_views": sum(
            r["full_field_can_visible_pixels_evaluator_only"] > 0 for r in present),
        "full_field_present_short_depth_target_views": sum(
            r["full_field_can_short_depth_pixels_evaluator_only"] > 0 for r in present),
        "cropped_field_present_target_visible_views": sum(
            r["cropped_field_can_visible_pixels_evaluator_only"] > 0 for r in present),
        "cropped_field_present_short_depth_target_views": sum(
            r["cropped_field_can_short_depth_pixels_evaluator_only"] > 0 for r in present),
    }
    result = {"scope": "V227 same V211 saved selected-body room/head poses rerendered at 896x480 with fovy=67: simulated 102.03x67 degree pinhole field approximates the Pi Wide published full field. RGB render and ideal evaluator depth/segmentation only; no recalibrated physical lens distortion, detector rerun, actor loop, real stereo depth, physical sensor, or safety qualification.",
              "source_sha256": sha(SOURCE), "v211_sha256": sha(BASELINE),
              "resolution_width_height": [WIDTH, H],
              "pinhole_fov_degrees_hv": [2 * math.degrees(math.atan(
                  WIDTH / H * math.tan(math.radians(67) / 2))), 67],
              "rows": rows, "summary": summary}
    (OUT / "full-pi-field-optical-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
