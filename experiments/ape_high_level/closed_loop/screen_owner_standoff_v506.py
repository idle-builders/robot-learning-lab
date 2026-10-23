"""Can the taught-item view work from a farther owner interaction standoff?

This is a static camera/geometry screen with the V492 range-track bearing.
The 30 cm earlier standoff is a candidate high-level goal, not an executed
or native-qualified backward route. It tests whether planning the meeting
place earlier could avoid reliance on V502's narrow emergency timing.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import robocasa_owner_person as person
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_person_joined_v497 as joined
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from screen_robocasa_owner_person_v489 import evaluator_box
from screen_robocasa_owner_safe_stop_v494 import aim_camera, depth_point
from stress_robocasa_owner_approach_v500 import place_owner


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-owner-standoff-v506"
ROUTE = ROOT / "artifacts/ape-robocasa-person-multitrack-v492/report.json"
CAMERA = "owner_wide20"


def scene(ape_xy, estimated_xy, track_xy):
    model = person.scene_with_person(prop_xy=[3., -1.05])
    model.body_pos[model.body("audit_ape").id, :2] = ape_xy
    aim_camera(model, CAMERA, estimated_xy, track_xy, 20)
    model.cam_pos[model.camera(CAMERA).id, :2] = ape_xy
    return model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    route = json.loads(ROUTE.read_text())["cases"][2]
    actual = np.asarray(route["episode"]["end_true_xy_m"])
    estimate = np.asarray(route["episode"]["end_estimated_xy_m"])
    track_state = route["multi_history"][-1]["tracks"][0]
    track = track_state["center_xy_m"]
    policy = json.loads(joined.CONFIG.read_text())
    prior = np.load(joined.PRIOR)
    envelope = (track_state["radius_m"] + policy["actor_body_radius_m"] +
                policy["required_gap_m"] + policy["object_shape_allowance_m"] +
                policy["assumed_object_max_total_displacement_m"])
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    results = []
    for label, offset in (("range_stop", 0.), ("farther_30cm", -.3)):
        ape_xy = actual + [0., offset]
        est_xy = estimate + [0., offset]
        model = scene(ape_xy, est_xy, track)
        data = mujoco.MjData(model)
        ape_body = model.body("audit_ape").id
        ape_geom = model.geom("audit_ape_geom").id
        closest = {"gap_m_capped_at_10cm": .1, "time_s": None, "geom": None}
        for t in np.arange(4., 6.001, .02):
            south = .4 * min(1., max(0., (t - 4.) / 2.))
            place_owner(model, south)
            gap, geom = clearance(model, data, ape_body, ape_geom, ape_xy)
            if gap < closest["gap_m_capped_at_10cm"]:
                closest = {"gap_m_capped_at_10cm": gap,
                           "time_s": round(float(t), 4), "geom": geom}
        renderer = mujoco.Renderer(model, height=480, width=640)
        views = []
        for suffix, south in (("presented", 0.), ("after_40cm_approach", .4)):
            place_owner(model, south)
            mujoco.mj_forward(model, data)
            rgb, depth, age = owner.capture(renderer, data, CAMERA, True)
            image = Image.fromarray(rgb)
            proposals = vision.detect(image)
            scores = (vision.feature([image.crop(item["box"]) for item in proposals]) @ exemplar
                      if proposals else np.empty(0))
            ranked = sorted(zip(proposals, scores), key=lambda item: float(item[1]), reverse=True)
            selected = ranked[0] if ranked and float(ranked[0][1]) >= owner.THRESHOLD else None
            can_box, can_pixels = evaluator_box(renderer, data, model, CAMERA,
                                                 "shown_can_geom")
            head_box, head_pixels = evaluator_box(renderer, data, model, CAMERA,
                                                   "owner_head_geom")
            point = (depth_point(selected[0]["box"], depth, data, model, CAMERA)
                     if selected else None)
            if point is not None:
                point = np.asarray(point)
                point[:2] += est_xy - ape_xy
                point = point.tolist()
            error = (float(np.linalg.norm(np.asarray(point) -
                                          model.body_pos[model.body("shown_can").id]))
                     if point is not None else None)
            frame = owner.owner_frame(rgb, f"{label}: {suffix}",
                                      selected[0]["box"] if selected else None, can_box)
            path = OUT / f"{label}-{suffix}.png"
            Image.fromarray(frame).save(path)
            views.append({"owner_motion": suffix, "south_m": south,
                          "capture_wall_s": age, "can_pixels_evaluator": can_pixels,
                          "head_pixels_evaluator": head_pixels,
                          "can_box_evaluator": can_box,
                          "head_box_evaluator": head_box,
                          "top_similarity": float(ranked[0][1]) if ranked else None,
                          "selected_box": selected[0]["box"] if selected else None,
                          "selected_iou_evaluator": owner.iou(
                              selected[0]["box"] if selected else None, can_box),
                          "depth_point_error_evaluator_m": error,
                          "image": {"path": str(path), "sha256": sha256(path)}})
            print(label, suffix, "can", can_pixels, "head", head_pixels,
                  "sim", views[-1]["top_similarity"],
                  "iou", views[-1]["selected_iou_evaluator"],
                  "gap", closest["gap_m_capped_at_10cm"], flush=True)
        renderer.close()
        reserve = float(np.linalg.norm(estimate + [0., offset] - track) - envelope)
        sampled = [actual + [0., value] for value in np.linspace(0., offset, 7)]
        path_cells = []
        for xy in sampled:
            xi = int(np.argmin(np.abs(prior["xs"] - xy[0])))
            yi = int(np.argmin(np.abs(prior["ys"] - xy[1])))
            path_cells.append({"xy_m": xy.tolist(),
                               "prior_free": bool(prior["free"][yi, xi]),
                               "prior_occupied": bool(prior["occupied"][yi, xi])})
        results.append({"standoff": label, "ape_xy_m": ape_xy.tolist(),
                        "estimated_xy_m": est_xy.tolist(),
                        "actor_assumed_motion_envelope_m": envelope,
                        "actor_standoff_reserve_m": reserve,
                        "actor_standoff_reserve_pass": reserve >= policy["minimum_staging_reserve_m"],
                        "actor_prior_backtrack_cells": path_cells,
                        "actor_prior_path_known_free": all(row["prior_free"] and
                                                           not row["prior_occupied"]
                                                           for row in path_cells),
                        "minimum_gap_through_40cm_owner_approach": closest,
                        "views": views})
    report = {"source_sha256": sha256(Path(__file__)),
              "route_report_sha256": sha256(ROUTE),
              "prior_map_sha256": sha256(joined.PRIOR),
              "policy_sha256": sha256(joined.CONFIG),
              "person_source_sha256": sha256(Path(person.__file__)),
              "taught_crop_sha256": sha256(owner.TEACH_CROP),
              "scope": "static actor-track-aimed ideal 90deg RGB-D views and independent 20ms clearance sweep at route stop vs 30cm farther south; not an executed route or skill, scripted owner approach, single taught can",
              "range_track_center_xy_m": track,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
