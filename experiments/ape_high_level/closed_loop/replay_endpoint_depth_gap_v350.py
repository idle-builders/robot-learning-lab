"""V350: arm-box RGB-D points versus selected robot body at V349 views.

All inputs are saved actor RGB-D, DINO boxes, camera pose and joint/pose
estimates. The evaluator's person gap is attached only after actor scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_lateral_clutter_depth_v285 import project
from direct_structural_rooms_v270 import selected_visual
from robot_geometry import RobotGeometry
from run import ROOT, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v350"
V349 = ROOT / "artifacts/ape-closed-loop-cove50-v349/endpoint-view-audit.json"


def box_depth_points(depth: np.ndarray, box: list[float] | None,
                     camera_pose: tuple[np.ndarray, np.ndarray],
                     fovy_deg: float) -> np.ndarray:
    if box is None:
        return np.empty((0, 3))
    x0, y0, x1, y1 = box
    selected = depth.copy()
    selected[:max(0, int(y0))] = np.nan
    selected[min(depth.shape[0], int(np.ceil(y1))):] = np.nan
    selected[:, :max(0, int(x0))] = np.nan
    selected[:, min(depth.shape[1], int(np.ceil(x1))):] = np.nan
    return project(selected, camera_pose, fovy_deg)


def score_points(points: np.ndarray, robot: RobotGeometry) -> dict:
    if not len(points):
        return {"points": 0, "min_body_gap_m": None,
                "q10_body_gap_m": None, "median_body_gap_m": None}
    gaps = RobotGeometry.solid_distances(points, robot.primitives())
    return {"points": len(points),
            "min_body_gap_m": round(float(np.min(gaps)), 5),
            "q10_body_gap_m": round(float(np.quantile(gaps, .1)), 5),
            "median_body_gap_m": round(float(np.median(gaps)), 5)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = json.loads(V349.read_text())
    model = mujoco.MjModel.from_xml_path(str(selected_visual()))
    robot = RobotGeometry(model)
    selfmask = SelfDepthMask(model, camera="lower_rgb",
                             width=640, height=480, agreement_m=.04)
    fovy = float(model.cam_fovy[model.camera("lower_rgb").id])
    rows = []
    try:
        for case in saved["rows"]:
            views = []
            for view in case["results"]:
                with np.load(ROOT / view["actor_packet_path"]) as packet:
                    qpos = packet["actor_qpos"].copy()
                    camera_pose = (packet["camera_origin_estimated_m"].copy(),
                                   packet["camera_rotation_estimated"].copy())
                    depths = {mode: packet[f"lower_{mode}_depth_m"].copy()
                              for mode in ("short", "noisy_long")}
                if len(qpos) != model.nq:
                    raise AssertionError("Saved actor qpos differs from selected model")
                robot.set_state(qpos)
                box = (max(view["actor_lower_arm_detections"],
                           key=lambda d: d["score"])["box_xyxy"]
                       if view["actor_lower_arm_detections"] else None)
                scored = {}
                for mode, measured in depths.items():
                    filtered, mask_info = selfmask.apply(
                        measured, qpos, max_depth_m=2.5)
                    points = box_depth_points(filtered, box, camera_pose, fovy)
                    scored[mode] = {**score_points(points, robot),
                                    "actor_self_mask": mask_info}
                views.append({"head_yaw_rad": view["head_yaw_rad"],
                              "actor_arm_box_xyxy": box,
                              "actor_arm_localized_evaluator": view[
                                  "evaluator_lower_arm_localized"],
                              "actor_gap_by_depth_mode": scored,
                              "evaluator_person_gap_m": view[
                                  "evaluator_person_gap_m"]})
            rows.append({"case": case["case"], "reach": case["reach"],
                         "seed": case["seed"], "views": views})
    finally:
        selfmask.close()
    report = {"scope": "Actor-only replay of selected Cove-50 visual robot model, estimated lower-camera pose, saved joint positions, DINO top arm box, self-masked short/noisy-long RGB-D and full-body collision primitives on V349 static endpoint views. Evaluator reach and true person gap are attached after scoring. This is point-to-body clearance from an imperfect box, not a tracked person surface, swept reach forecast, or physical stopping proof.",
              "source_sha256": sha(SOURCE),
              "v349_report_sha256": sha(V349),
              "selected_visual_sha256": sha(selected_visual()),
              "rows": rows}
    (OUT / "endpoint-depth-gap.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], [(v["head_yaw_rad"],
                            v["actor_gap_by_depth_mode"]["short"]["min_body_gap_m"],
                            v["actor_gap_by_depth_mode"]["noisy_long"]["min_body_gap_m"])
                           for v in row["views"]], flush=True)


if __name__ == "__main__":
    main()
