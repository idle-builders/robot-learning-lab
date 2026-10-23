"""Ablate captured owner-presentation packets with spatial depth support.

Every actor choice uses only transformed saved depth, the confirmed range
track and saved prior-free map. A separate evaluator screens the chosen fixed
pose through the scripted owner trajectory; it never feeds the actor.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import robocasa_depth_standoff_turn_v512 as actor
import owner_depth_change_robust_v515 as robust
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_route_v407 as route
from audit_robocasa_layout2_routes_v409 import clearance, sha256
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
TRIAL = ROOT / "artifacts/ape-robocasa-person-depth-standoff-v512/report.json"
OUT = ROOT / "artifacts/ape-robocasa-person-depth-standoff-v512/depth-ablation-v515.json"


def camera_geometry(ep):
    row = next(row for row in ep["trace"] if row["stage"] == "owner_depth_change")
    track = max(ep["multi_history"][-1]["tracks"], key=lambda x: x["hits"])
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    estimate = np.asarray(row["capture_pose_est_xy_m"])
    aim_camera(model, actor.base.CAMERA, estimate, track["center_xy_m"], 20)
    model.cam_pos[model.camera(actor.base.CAMERA).id, :2] = estimate
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cam = model.camera(actor.base.CAMERA).id
    return estimate, track, data.cam_xmat[cam].reshape(3, 3).copy(), data.cam_xpos[cam].copy()


def evaluator_gap(ep, retreat_m, approach_m):
    if retreat_m is None:
        return None
    actual = np.asarray(ep["route_episode"]["end_true_xy_m"]) + [0., -retreat_m]
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    ape_body = model.body("audit_ape").id
    ape_geom = model.geom("audit_ape_geom").id
    best = {"gap_m_capped_at_10cm": .1, "south_m": None, "geom": None}
    for south in np.arange(0., approach_m + .001, .02):
        actor._place_owner(model, float(south))
        gap, geom = clearance(model, data, ape_body, ape_geom, actual)
        if gap < best["gap_m_capped_at_10cm"]:
            best = {"gap_m_capped_at_10cm": float(gap),
                    "south_m": round(float(south), 3), "geom": geom}
    return best


def main():
    report = json.loads(TRIAL.read_text())
    ep = next(ep for ep in report["episodes"]
              if ep["response"] == "approach_60cm_depth")
    depth_row = next(row for row in ep["trace"] if row["stage"] == "owner_depth_change")
    packet_path = Path(depth_row["depth_packets"]["path"])
    raw = np.load(packet_path)
    before = raw["before_depth"]
    after = raw["after_depth"]
    estimate, track, rotation, camera_xyz = camera_geometry(ep)
    # Freeze only the corridor cells the live actor recorded as free. The
    # offline replay must not grant unobserved cells from the survey map.
    selected = next(row["chosen"] for row in ep["trace"]
                    if row["stage"] == "choose_owner_standoff")
    grid = route.SensorGrid()
    for cell in selected["cells"]:
        if cell["free"] and not cell["occupied"]:
            grid.free[grid.cell(np.asarray(cell["xy_m"]))] = True
    variations = [("ideal", 0, before, after),
                  ("stale_after_packet", 0, before, before),
                  ("missing_before_packet", 0, np.full_like(before, np.nan), after),
                  ("right_40pct_occluded", 0, before,
                   np.where(np.indices(after.shape)[1] >= 384, np.nan, after))]
    for fraction in (.1, .3, .5, .8):
        for seed in range(10):
            rng = np.random.default_rng(51400 + seed + round(fraction*100)*100)
            damaged = after.copy()
            damaged[rng.random(after.shape) < fraction] = np.nan
            variations.append((f"dropout_{fraction:.1f}", seed, before, damaged))
    for sigma in (.01, .03):
        for seed in range(10):
            rng = np.random.default_rng(51500 + seed + round(sigma*1000)*100)
            noisy_before = before + rng.normal(0., sigma, before.shape)
            noisy_after = after + rng.normal(0., sigma, after.shape)
            variations.append((f"independent_noise_{sigma:.2f}m", seed,
                               noisy_before, noisy_after))
    for bias in (.03, .06):
        variations.append((f"after_depth_bias_{bias:.2f}m", 0,
                           before, after + bias))
    rows = []
    gap_cache = {}
    for name, seed, base_depth, changed_depth in variations:
        mask, points, _ = robust.changed_points(base_depth, changed_depth,
                                                rotation, camera_xyz,
                                                np.asarray(track["center_xy_m"]))
        chosen = None
        if len(points) >= 50:
            chosen, _ = actor._choose_standoff(grid, points, estimate)
        retreat = chosen["distance_m"] if chosen else None
        if retreat not in gap_cache:
            gap_cache[retreat] = evaluator_gap(ep, retreat, .6)
        gap = gap_cache[retreat]
        rows.append({"variation": name, "seed": seed,
                     "raw_changed_pixels": int(mask.sum()),
                     "filtered_points": int(len(points)),
                     "chosen_retreat_m": retreat,
                     "chosen_reserve_m": chosen["reserve_m"] if chosen else None,
                     "actor_outcome": ("standoff_supported" if chosen else
                                       "abstain_unknown_or_no_free_candidate"),
                     "evaluator_60cm_owner_gap": gap,
                     "evaluator_5cm_margin_pass": (gap is not None and
                         gap["gap_m_capped_at_10cm"] >= .05)})
    grouped = {}
    for name in sorted({row["variation"] for row in rows}):
        subset = [row for row in rows if row["variation"] == name]
        grouped[name] = {"trials": len(subset),
                         "chosen_distances_m": sorted({row["chosen_retreat_m"]
                                                       for row in subset if row["chosen_retreat_m"] is not None}),
                         "abstentions": sum(row["chosen_retreat_m"] is None for row in subset),
                         "evaluator_margin_passes": sum(row["evaluator_5cm_margin_pass"] for row in subset),
                         "filtered_point_range": [min(row["filtered_points"] for row in subset),
                                                  max(row["filtered_points"] for row in subset)]}
        print(name, grouped[name], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(TRIAL),
              "raw_depth_packet_sha_pass": sha256(packet_path) == depth_row["depth_packets"]["sha256"],
              "depth_change_source_sha256": sha256(Path(robust.__file__)),
              "actor_standoff_source_sha256": sha256(Path(actor.__file__)),
              "scope": "offline same-packet depth dropout, per-frame noise, bias, occlusion and stale/missing packet ablation with 3x3 morphological support and >=50px connected changes on one revealed owner-presentation frame; choices use actor calibration/range track and only corridor cells recorded free by live route, static evaluator sweeps chosen pose through 60cm owner approach; does not simulate exposure, scan timing or gait",
              "variations": grouped, "rows": rows}
    OUT.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
