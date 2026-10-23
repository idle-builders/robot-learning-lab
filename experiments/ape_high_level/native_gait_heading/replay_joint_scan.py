"""Test bounded XY+yaw scan correction on saved native ape gait packets.

This reuses the exact range packets from the translation-only native gait
comparison. Simulator truth is evaluator-only after the initial pose anchor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from experiments.ape_high_level.closed_loop.localization import ScanLocalizer
from experiments.ape_high_level.native_gait_range import replay_range_fusion as prior


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-native-gait-heading/v1"
PRIOR = ROOT / "artifacts/ape-native-gait-range/v1"
CASES = prior.CASES
SCAN_ARMS = ("range_360", "range_front_270", "range_360_drop_2to5s")
MAX_XY_UPDATE_M = 0.13
MAX_YAW_UPDATE_RAD = math.radians(5)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrapped(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class JointScanLocalizer:
    def __init__(self) -> None:
        self.points: np.ndarray | None = None

    def correct(self, scan: list[float | None], predicted: np.ndarray
                ) -> tuple[np.ndarray, dict]:
        local = ScanLocalizer.endpoints(scan, np.zeros(3))
        if self.points is None:
            self.points = ScanLocalizer.endpoints(scan, predicted)
            return predicted, {"status": "mapped" if len(self.points) >= 20
                               else "weak_initial_scan",
                               "scan_points": len(local), "map_points": len(self.points)}
        if len(self.points) < 20:
            return predicted, {"status": "weak_map", "scan_points": len(local)}
        if len(local) < 20:
            return predicted, {"status": "weak_scan", "scan_points": len(local)}
        tree = cKDTree(self.points)

        def transformed(delta: np.ndarray) -> np.ndarray:
            yaw = predicted[2] + delta[2]
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])
            return local @ rotation.T + predicted[:2] + delta[:2]

        def residual(delta: np.ndarray) -> float:
            if (np.linalg.norm(delta[:2]) > MAX_XY_UPDATE_M or
                    abs(delta[2]) > MAX_YAW_UPDATE_RAD):
                return 1.0 + float(np.dot(delta, delta))
            distances, _ = tree.query(transformed(delta))
            nearest = np.sort(distances)[:max(16, int(.7 * len(distances)))]
            return float(np.mean(np.minimum(nearest, .12) ** 2) +
                         .08 * np.sum(delta[:2] ** 2) +
                         .015 * delta[2] ** 2)

        initial = np.zeros(3)
        simplex = np.array([[0, 0, 0], [.015, 0, 0],
                            [0, .015, 0], [0, 0, .012]])
        answer = minimize(residual, initial, method="Nelder-Mead",
                          options={"initial_simplex": simplex, "xatol": 1e-4,
                                   "fatol": 1e-8, "maxiter": 180})
        offset = np.asarray(answer.x, dtype=float)
        distances, _ = tree.query(transformed(offset))
        inliers = distances < .10
        quality = {"scan_points": len(local),
                   "inlier_fraction": round(float(inliers.mean()), 3),
                   "median_residual_m": round(float(np.median(distances)), 4),
                   "xy_correction_m": round(float(np.linalg.norm(offset[:2])), 4),
                   "yaw_correction_deg": round(float(np.degrees(offset[2])), 3)}
        if (np.linalg.norm(offset[:2]) > MAX_XY_UPDATE_M or
                abs(offset[2]) > MAX_YAW_UPDATE_RAD or inliers.mean() < .48):
            return predicted, {"status": "registration_rejected", **quality}
        corrected = predicted + offset
        if quality["inlier_fraction"] >= .65 and quality["median_residual_m"] <= .06:
            new_points = ScanLocalizer.endpoints(scan, corrected)
            self.points = np.vstack([self.points, new_points])[-5000:]
        return corrected, {"status": "registered", **quality}


def run(case: str, arm: str) -> tuple[dict, list[dict]]:
    packet_path = PRIOR / f"{case}-{arm}-trace.json"
    packets = json.loads(packet_path.read_text())
    if len(packets) != 400:
        raise ValueError("Expected 400 native frames")
    localizer = JointScanLocalizer()
    current = np.asarray(packets[0]["proprio_xy_m"], dtype=float)
    yaw_offset = 0.0
    last_packet_step: int | None = None
    last_registration_step: int | None = None
    status = "uninitialized"
    counts: dict[str, int] = {}
    rows = []
    for step, packet in enumerate(packets):
        if step:
            current[:2] += (np.asarray(packet["proprio_xy_m"]) -
                            np.asarray(packets[step - 1]["proprio_xy_m"]))
        pose = np.array([current[0], current[1],
                         packet["measured_yaw_rad"] + yaw_offset])
        scan = packet["scan_m"]
        quality = None
        if scan is not None:
            last_packet_step = step
            corrected, quality = localizer.correct(scan, pose)
            current[:] = corrected[:2]
            yaw_offset = wrapped(float(corrected[2] - packet["measured_yaw_rad"]))
            pose = corrected
            status = quality["status"]
            counts[status] = counts.get(status, 0) + 1
            last_registration_step = step if status == "registered" else last_registration_step
        elif packet["scan_missing_declared"]:
            status = "declared_packet_loss"
        age_packet = ((step - last_packet_step) * .02
                      if last_packet_step is not None else None)
        age_registered = ((step - last_registration_step) * .02
                          if last_registration_step is not None else None)
        healthy = (age_packet is not None and age_packet <= .3 and
                   age_registered is not None and age_registered <= .3 and
                   status != "registration_rejected")
        actual_xy = np.asarray(packet["native_xy_evaluator_only_m"])
        xy_error = float(np.linalg.norm(current - actual_xy))
        yaw_error = abs(math.degrees(wrapped(
            float(pose[2] - packet["native_yaw_evaluator_only_rad"]))))
        rows.append({
            "frame": step, "t_s": packet["t_s"],
            "source_packet_frame": step,
            "scan_status": status, "registration_quality": quality,
            "scan_m": scan,
            "measured_yaw_rad": packet["measured_yaw_rad"],
            "estimated_xy_m": current.round(5).tolist(),
            "estimated_yaw_rad": round(float(pose[2]), 6),
            "actor_belief_healthy": healthy,
            "packet_age_s": round(age_packet, 3) if age_packet is not None else None,
            "registered_age_s": round(age_registered, 3) if age_registered is not None else None,
            "native_xy_evaluator_only_m": packet["native_xy_evaluator_only_m"],
            "native_yaw_evaluator_only_rad": packet["native_yaw_evaluator_only_rad"],
            "xy_error_evaluator_only_m": round(xy_error, 5),
            "yaw_error_evaluator_only_deg": round(yaw_error, 4),
        })
    errors = np.asarray([row["xy_error_evaluator_only_m"] for row in rows])
    yaw_errors = np.asarray([row["yaw_error_evaluator_only_deg"] for row in rows])
    healthy_rows = [row for row in rows if row["actor_belief_healthy"]]
    summary = {
        "case": case, "arm": arm,
        "source_packet_trace": str(packet_path.relative_to(ROOT)),
        "source_packet_trace_sha256": sha(packet_path),
        "frames": len(rows), "received_scans": sum(row["scan_m"] is not None for row in rows),
        "status_counts": counts,
        "final_xy_error_m": round(float(errors[-1]), 5),
        "p95_xy_error_m": round(float(np.quantile(errors, .95)), 5),
        "final_yaw_error_deg": round(float(yaw_errors[-1]), 4),
        "p95_yaw_error_deg": round(float(np.quantile(yaw_errors, .95)), 4),
        "healthy_frames": len(healthy_rows),
        "healthy_over_10cm_xy_frames_evaluator_only": sum(
            row["xy_error_evaluator_only_m"] > .1 for row in healthy_rows),
        "healthy_over_5deg_yaw_frames_evaluator_only": sum(
            row["yaw_error_evaluator_only_deg"] > 5 for row in healthy_rows),
    }
    return summary, rows


def render_video(rows_by_arm: dict[str, list[dict]], output: Path) -> int:
    source = prior.SOURCE / "varied150-forward-seed144.mp4"
    reader = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             reader.get(cv2.CAP_PROP_FPS), (1660, 720))
    if not reader.isOpened() or not writer.isOpened():
        raise RuntimeError("Cannot read or write video")
    colors = {"range_360": (70, 210, 70),
              "range_front_270": (70, 150, 230),
              "range_360_drop_2to5s": (210, 70, 220)}
    count = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        step = min(399, count * 2)
        panel = np.zeros((720, 700, 3), dtype=np.uint8)
        scale = min((700 - 80) / (prior.ROOM[1] - prior.ROOM[0]),
                    (720 - 280) / (prior.ROOM[3] - prior.ROOM[2]))
        def pixel(xy: list[float] | np.ndarray | tuple[float, float]) -> tuple[int, int]:
            return (round(40 + (xy[0] - prior.ROOM[0]) * scale),
                    round(720 - 230 - (xy[1] - prior.ROOM[2]) * scale))
        def box(b: tuple[float, float, float, float], color: tuple[int, int, int],
                thickness: int) -> None:
            x0, x1, y0, y1 = b
            cv2.rectangle(panel, pixel((x0, y1)), pixel((x1, y0)), color, thickness)
        box(prior.ROOM, (110, 110, 110), 2)
        for furniture in prior.FURNITURE.values():
            box(furniture, (70, 70, 70), -1)
        cv2.putText(panel, "Joint XY + heading scan correction", (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
        truth = [row["native_xy_evaluator_only_m"]
                 for row in rows_by_arm["range_360"][:step + 1]]
        if len(truth) > 1:
            cv2.polylines(panel, [np.asarray([pixel(p) for p in truth], np.int32)],
                          False, (235, 235, 235), 2)
        for index, (arm, rows) in enumerate(rows_by_arm.items()):
            points = [row["estimated_xy_m"] for row in rows[:step + 1]]
            if len(points) > 1:
                cv2.polylines(panel, [np.asarray([pixel(p) for p in points], np.int32)],
                              False, colors[arm], 2)
            current = rows[step]
            false_healthy = (current["actor_belief_healthy"] and
                             current["xy_error_evaluator_only_m"] > .1)
            cv2.putText(panel, f"{arm}: {current['xy_error_evaluator_only_m']*100:.1f}cm "
                        f"{current['yaw_error_evaluator_only_deg']:.1f}deg "
                        f"[{current['scan_status'][:10]}]"
                        f"{' FALSE HEALTH' if false_healthy else ''}",
                        (18, 560 + 34 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, colors[arm], 1, cv2.LINE_AA)
        writer.write(np.hstack([frame, panel]))
        count += 1
    reader.release()
    writer.release()
    return count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior_report = PRIOR / "report.json"
    if not prior_report.exists():
        raise FileNotFoundError("Run native range packet experiment first")
    summaries = []
    varied = {}
    for case in CASES:
        for arm in SCAN_ARMS:
            summary, rows = run(case, arm)
            trace = OUT / f"{case}-{arm}-joint-trace.json"
            trace.write_text(json.dumps(rows, indent=2) + "\n")
            summary["trace"] = str(trace.relative_to(ROOT))
            summary["trace_sha256"] = sha(trace)
            summaries.append(summary)
            if case == "varied150-forward-seed144":
                varied[arm] = rows
    video = OUT / "varied150-joint-scan.mp4"
    frames = render_video(varied, video)
    result = {
        "schema": "robot-learning-lab.native-gait-joint-scan.v1",
        "task": "T-mftnhd",
        "prior_report": str(prior_report.relative_to(ROOT)),
        "prior_report_sha256": sha(prior_report),
        "joint_scan_code_sha256": sha(Path(__file__)),
        "packet_generator_code_sha256": sha(Path(prior.__file__)),
        "source_video_sha256": sha(prior.SOURCE / "varied150-forward-seed144.mp4"),
        "algorithm": {"bounded_xy_update_m": MAX_XY_UPDATE_M,
                      "bounded_yaw_update_deg": math.degrees(MAX_YAW_UPDATE_RAD),
                      "trimmed_nearest_fraction": .7,
                      "min_inlier_fraction": .48,
                      "registered_age_gate_s": .3},
        "video": str(video.relative_to(ROOT)), "video_sha256": sha(video),
        "video_frames": frames,
        "scope": "Saved 50 cm native gait and synthetic 2D room range packets on rejected Direct Frame; no action feedback, 3D ray physics, physical IMU/range or owner scene. Initial true pose anchor only; subsequent native truth evaluator-only.",
        "cases": summaries,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in summaries:
        print(row["case"], row["arm"], row["final_xy_error_m"],
              row["final_yaw_error_deg"], row["healthy_frames"],
              row["healthy_over_10cm_xy_frames_evaluator_only"], row["status_counts"])


if __name__ == "__main__":
    main()
