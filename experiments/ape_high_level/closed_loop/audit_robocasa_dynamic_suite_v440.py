"""Independent full-horizon MuJoCo geometry and video audit for V439."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from audit_robocasa_layout2_routes_v409 import clearance, evaluator_scene, sha256
from run_robocasa_layout2_route_v407 import GOAL


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-dynamic-suite-v439"
SUBSTEP_S = .02


def prop_at(case: dict, time_s: float) -> np.ndarray:
    start = np.asarray(case["from_xy_m"], dtype=float)
    end = np.asarray(case["to_xy_m"], dtype=float)
    fraction = min(1., max(0., (time_s - case["begin_s"]) /
                            (case["end_s"] - case["begin_s"])))
    return start + fraction * (end - start)


def prop_fixture_gap(model, data, prop_geom: int, ape_geom: int,
                     xy: np.ndarray) -> tuple[float, str | None]:
    best = .3
    label = None
    for other in range(model.ngeom):
        if other in (prop_geom, ape_geom) or model.geom(other).name.startswith("floor_"):
            continue
        if np.linalg.norm(data.geom_xpos[other, :2] - xy) > (
                model.geom_rbound[other] + model.geom_rbound[prop_geom] + .3):
            continue
        gap = float(mujoco.mj_geomDistance(model, data, prop_geom, other, .3, None))
        if gap < best:
            best, label = gap, model.geom(other).name
    return best, label


def audit_case(case: dict) -> dict:
    scenario, episode = case["scenario"], case["episode"]
    model, data, ape_body, ape_geom = evaluator_scene(scenario["from_xy_m"])
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    nearest = {"gap_m_capped_at_10cm": .1, "geom": None, "time_s": None,
               "phase": None}
    prop_nearest = {"signed_gap_m": .5, "time_s": None, "phase": None}
    total_samples = 0

    def sample(time_s: float, pose: np.ndarray, phase: str) -> None:
        nonlocal nearest, prop_nearest, total_samples
        prop = prop_at(scenario, time_s)
        model.body_pos[prop_body, :2] = prop
        mujoco.mj_forward(model, data)
        gap, geom = clearance(model, data, ape_body, ape_geom, pose)
        prop_gap = float(mujoco.mj_geomDistance(model, data, ape_geom,
                                                prop_geom, .5, None))
        if gap < nearest["gap_m_capped_at_10cm"]:
            nearest = {"gap_m_capped_at_10cm": gap, "geom": geom,
                       "time_s": round(time_s, 4), "phase": phase}
        if prop_gap < prop_nearest["signed_gap_m"]:
            prop_nearest = {"signed_gap_m": prop_gap,
                            "time_s": round(time_s, 4), "phase": phase}
        total_samples += 1

    for row in episode["trace"]:
        start = np.asarray(row["actual_xy_m"], dtype=float)
        sample(float(row["time_s"]), start, "route_capture")
        requested = row.get("requested_xy_m")
        if requested is None:
            continue
        delta = ((np.asarray(requested) - np.asarray(row["estimated_xy_m"]))
                 * (1. + .30 / 5.89))
        for index, sub in enumerate(row["motion_substeps"]):
            fraction = (index if sub["stopped_before_move"] else index + 1) / 20
            sample(float(sub["time_s"]), start + fraction * delta, "executed_move")
    if episode["end_time_s"] < scenario["end_s"]:
        held = np.asarray(episode["end_true_xy_m"], dtype=float)
        times = np.arange(episode["end_time_s"] + SUBSTEP_S,
                          scenario["end_s"] + SUBSTEP_S / 2, SUBSTEP_S)
        for time_s in np.unique(np.r_[times, scenario["end_s"]]):
            sample(float(time_s), held, "post_stop_hold")

    prop_fixture_min = {"gap_m_capped_at_30cm": .3, "geom": None,
                        "time_s": None}
    for time_s in np.arange(0., scenario["end_s"] + SUBSTEP_S / 2, SUBSTEP_S):
        prop = prop_at(scenario, float(time_s))
        model.body_pos[prop_body, :2] = prop
        mujoco.mj_forward(model, data)
        gap, geom = prop_fixture_gap(model, data, prop_geom, ape_geom, prop)
        if gap < prop_fixture_min["gap_m_capped_at_30cm"]:
            prop_fixture_min = {"gap_m_capped_at_30cm": gap,
                                "geom": geom, "time_s": round(float(time_s), 4)}
    video = Path(episode["video"]["path"])
    reader = imageio.get_reader(video)
    frames = reader.count_frames()
    reader.close()
    actual_goal_gap = float(np.linalg.norm(np.asarray(episode["end_true_xy_m"]) - GOAL))
    return {"scenario_id": scenario["id"], "package": case["package"],
            "outcome": episode["outcome"],
            "actual_goal_gap_m": actual_goal_gap,
            "end_pose_error_m": float(np.linalg.norm(
                np.asarray(episode["end_true_xy_m"]) -
                np.asarray(episode["end_estimated_xy_m"]))),
            "nearest_ape_any": nearest, "nearest_ape_prop": prop_nearest,
            "ape_5cm_margin_pass": nearest["gap_m_capped_at_10cm"] >= .05,
            "prop_fixture_nearest": prop_fixture_min,
            "prop_fixture_nonoverlap_pass": prop_fixture_min["gap_m_capped_at_30cm"] >= 0.,
            "video_sha_pass": sha256(video) == episode["video"]["sha256"],
            "video_frames_pass": frames == episode["video"]["frames"] == len(episode["trace"]) + 1,
            "geometry_samples": total_samples}


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    cases = [audit_case(case) for case in report["cases"]]
    pairs = {}
    for row in report["cases"]:
        pairs.setdefault(row["scenario"]["id"], set()).add(row["episode"]["scene_fingerprint"])
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_dynamic_suite_v439.py")),
              "suite_source_sha_pass": report["suite_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_dynamic_suite_v439.json")),
              "scene_pair_pass": all(len(fingerprints) == 1 for fingerprints in pairs.values()),
              "all_videos_pass": all(row["video_sha_pass"] and row["video_frames_pass"]
                                     for row in cases),
              "all_prop_paths_nonoverlap": all(row["prop_fixture_nonoverlap_pass"]
                                               for row in cases),
              "cases": cases}
    (OUT / "audit-v440.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in cases:
        print(row["scenario_id"], row["package"], row["outcome"],
              "ape_gap", round(row["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_fixture", round(row["prop_fixture_nearest"]["gap_m_capped_at_30cm"], 3),
              "goal", round(row["actual_goal_gap_m"], 3), flush=True)
    print("all_videos_pass", output["all_videos_pass"],
          "all_prop_paths_nonoverlap", output["all_prop_paths_nonoverlap"], flush=True)


if __name__ == "__main__":
    main()
