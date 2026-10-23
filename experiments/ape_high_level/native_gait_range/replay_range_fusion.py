"""Replay range-scan correction on retained native 50 cm walking traces.

The walking policy is not rerun in the room. Simulator truth generates range
packets and scores the estimator, but never enters its update after the first
pose anchor. The room and all packet faults are declared synthetic probes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from experiments.ape_high_level.closed_loop.localization import ScanLocalizer


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ODOMETRY = ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1"
SOURCE = ROOT / "artifacts/ape-cove50-transfer-20261008"
OUT = ROOT / "artifacts/ape-native-gait-range/v1"
CASES = ("planar200v3-forward", "varied150-forward-seed144",
         "robust10v200-forward-seed144")
ROOM = (0.0, 5.0, -3.2, 1.6)
FURNITURE = {
    "table": (1.8, 2.9, 0.65, 1.25),
    "sofa": (1.65, 2.8, -3.05, -2.58),
    "cabinet": (3.75, 4.5, -2.95, -2.45),
    "stool": (4.05, 4.48, -0.15, 0.28),
    "storage": (0.2, 0.72, -0.25, 0.28),
}
SCAN_RAYS = 72
SCAN_PERIOD_FRAMES = 5  # 10 Hz for a 50 Hz gait trace.
MAX_RANGE_M = 2.2
RANGE_NOISE_STD_M = 0.01
RANGE_DROPOUT_FRACTION = 0.02
RANGE_FAULT_SEED = 76012
_fault_rng = np.random.default_rng(RANGE_FAULT_SEED)
RANGE_DROPS = _fault_rng.random((400, SCAN_RAYS))
RANGE_NOISE = _fault_rng.normal(0, RANGE_NOISE_STD_M, (400, SCAN_RAYS))
PACKET_STALE_S = 0.3
REMEMBERED_LOCATION_ERROR_GATE_M = 0.1  # Evaluator audit only.
CONFIGS = {
    "proprio_only": {"fov_deg": None, "fault_interval_s": None},
    "range_360": {"fov_deg": 360, "fault_interval_s": None},
    "range_front_180": {"fov_deg": 180, "fault_interval_s": None},
    "range_front_270": {"fov_deg": 270, "fault_interval_s": None},
    "range_360_ideal_heading": {"fov_deg": 360, "fault_interval_s": None,
                                "ideal_heading": True},
    "range_360_drop_2to5s": {"fov_deg": 360, "fault_interval_s": (2.0, 5.0)},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rectangle_segments(box: tuple[float, float, float, float]
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    x0, x1, y0, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return [(np.array(corners[i], dtype=float),
             np.array(corners[(i + 1) % 4], dtype=float)) for i in range(4)]


SEGMENTS = (rectangle_segments((ROOM[0], ROOM[1], ROOM[2], ROOM[3])) +
            [segment for box in FURNITURE.values()
             for segment in rectangle_segments(box)])


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def ray_segment_distance(origin: np.ndarray, direction: np.ndarray,
                         start: np.ndarray, end: np.ndarray) -> float | None:
    span = end - start
    denominator = cross(direction, span)
    if abs(denominator) < 1e-9:
        return None
    difference = start - origin
    distance = cross(difference, span) / denominator
    fraction = cross(difference, direction) / denominator
    return distance if distance > 0 and 0 <= fraction <= 1 else None


def sensor_scan(actual_xy: np.ndarray, actual_yaw: float, fov_deg: int,
                step: int) -> list[float | None]:
    scan: list[float | None] = []
    for ray_index, relative_yaw in enumerate(
            np.linspace(-math.pi, math.pi, SCAN_RAYS, endpoint=False)):
        if fov_deg < 360 and abs(relative_yaw) > math.radians(fov_deg / 2):
            scan.append(None)
            continue
        direction = np.array([math.cos(actual_yaw + relative_yaw),
                              math.sin(actual_yaw + relative_yaw)])
        distances = [distance for start, end in SEGMENTS
                     if (distance := ray_segment_distance(
                         actual_xy, direction, start, end)) is not None]
        nearest = min(distances) if distances else math.inf
        if (nearest >= MAX_RANGE_M or
                RANGE_DROPS[step, ray_index] < RANGE_DROPOUT_FRACTION):
            scan.append(None)
        else:
            scan.append(round(max(.01, nearest + RANGE_NOISE[step, ray_index]), 4))
    return scan


def native_yaw(qpos: np.ndarray) -> np.ndarray:
    w, x, y, z = qpos[:, 3:7].T
    return np.unwrap(np.arctan2(2 * (w * z + x * y),
                                1 - 2 * (y * y + z * z)))


def load_case(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source = SOURCE / f"{case}.npz"
    with np.load(source, allow_pickle=False) as saved:
        qpos = saved["qpos"].astype(float)
        dt = float(saved["dt"])
    odometry_path = ODOMETRY / f"{case}-trace.json"
    rows = json.loads(odometry_path.read_text())
    if len(rows) != len(qpos) or abs(dt - .02) > 1e-9:
        raise ValueError(f"Native trace mismatch: {case}")
    for index, row in enumerate(rows):
        if not np.allclose(row["actual_xy_m_evaluator_only"], qpos[index, :2], atol=1e-5):
            raise ValueError(f"Evaluator alignment mismatch: {case}, frame {index}")
    raw = np.asarray([row["estimated_xy_m"]["combined_illustrative_impairments"]
                      for row in rows])
    return qpos[:, :2].copy(), native_yaw(qpos), raw, {
        "source_qpos": str(source.relative_to(ROOT)),
        "source_qpos_sha256": sha(source),
        "source_odometry": str(odometry_path.relative_to(ROOT)),
        "source_odometry_sha256": sha(odometry_path),
        "dt_s": dt,
    }


def run_case(case: str, config_name: str, config: dict) -> tuple[dict, list[dict]]:
    actual, actual_yaw, raw, provenance = load_case(case)
    dt = provenance["dt_s"]
    measured_yaw = actual_yaw.copy()
    if not config.get("ideal_heading", False):
        measured_yaw += np.deg2rad(np.arange(len(raw)) * dt)
    estimate = raw[0].copy()
    localizer = ScanLocalizer()
    last_packet_step: int | None = None
    last_registered_step: int | None = None
    status = "no_range_sensor" if config["fov_deg"] is None else "uninitialized"
    rows = []
    status_counts: dict[str, int] = {}
    for step in range(len(raw)):
        if step:
            estimate += raw[step] - raw[step - 1]
        now = step * dt
        packet: list[float | None] | None = None
        registration_quality: dict | None = None
        due = step % SCAN_PERIOD_FRAMES == 0 and config["fov_deg"] is not None
        fault = config["fault_interval_s"]
        missing = bool(due and fault and fault[0] <= now < fault[1])
        if due and not missing:
            packet = sensor_scan(actual[step], actual_yaw[step],
                                 config["fov_deg"], step)
            last_packet_step = step
            predicted = np.array([estimate[0], estimate[1], measured_yaw[step]])
            corrected, outcome = localizer.correct(packet, predicted)
            registration_quality = outcome
            estimate[:] = corrected[:2]
            status = outcome["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "registered":
                last_registered_step = step
                if (outcome["inlier_fraction"] >= .65 and
                        outcome["median_residual_m"] <= .06):
                    new_points = ScanLocalizer.endpoints(packet, corrected)
                    localizer.points = np.vstack([localizer.points, new_points])[-5000:]
        elif missing:
            status = "declared_packet_loss"
        packet_age = ((step - last_packet_step) * dt
                      if last_packet_step is not None else None)
        registered_age = ((step - last_registered_step) * dt
                          if last_registered_step is not None else None)
        actor_healthy = (registered_age is not None and
                         registered_age <= PACKET_STALE_S and
                         packet_age is not None and packet_age <= PACKET_STALE_S and
                         status != "registration_rejected")
        error = float(np.linalg.norm(estimate - actual[step]))
        rows.append({
            "frame": step, "t_s": round(now, 3),
            "native_xy_evaluator_only_m": actual[step].round(5).tolist(),
            "native_yaw_evaluator_only_rad": round(float(actual_yaw[step]), 6),
            "measured_yaw_rad": round(float(measured_yaw[step]), 6),
            "proprio_xy_m": raw[step].round(5).tolist(),
            "estimated_xy_m": estimate.round(5).tolist(),
            "scan_due": due, "scan_missing_declared": missing,
            "scan_m": packet, "scan_status": status,
            "registration_quality": registration_quality,
            "scan_valid_rays": sum(v is not None for v in packet) if packet else None,
            "packet_age_s": round(packet_age, 3) if packet_age is not None else None,
            "registered_age_s": round(registered_age, 3) if registered_age is not None else None,
            "belief_health_allows_remembered_location": actor_healthy,
            "xy_error_evaluator_only_m": round(error, 5),
        })
    errors = np.asarray([r["xy_error_evaluator_only_m"] for r in rows])
    permitted = [r for r in rows if r["belief_health_allows_remembered_location"]]
    summary = {
        "case": case, "config": config_name, "config_values": config,
        **provenance,
        "frames": len(rows),
        "received_scans": sum(r["scan_m"] is not None for r in rows),
        "scan_status_counts": status_counts,
        "final_xy_error_m": round(float(errors[-1]), 5),
        "mean_xy_error_m": round(float(errors.mean()), 5),
        "p95_xy_error_m": round(float(np.quantile(errors, .95)), 5),
        "max_xy_error_m": round(float(errors.max()), 5),
        "actor_healthy_frames": len(permitted),
        "actor_healthy_but_over_10cm_error_frames_evaluator_only": sum(
            r["xy_error_evaluator_only_m"] > REMEMBERED_LOCATION_ERROR_GATE_M
            for r in permitted),
        "first_actor_healthy_but_over_10cm_time_s_evaluator_only": next(
            (r["t_s"] for r in permitted
             if r["xy_error_evaluator_only_m"] > REMEMBERED_LOCATION_ERROR_GATE_M),
            None),
    }
    return summary, rows


def draw_map(panel: np.ndarray, rows_by_config: dict[str, list[dict]],
             step: int) -> None:
    height, width = panel.shape[:2]
    scale = min((width - 80) / (ROOM[1] - ROOM[0]),
                (height - 280) / (ROOM[3] - ROOM[2]))
    def pixel(xy: tuple[float, float] | np.ndarray) -> tuple[int, int]:
        return (round(40 + (xy[0] - ROOM[0]) * scale),
                round(height - 230 - (xy[1] - ROOM[2]) * scale))
    def rectangle(box: tuple[float, float, float, float], color: tuple[int, int, int],
                  thickness: int = 2) -> None:
        x0, x1, y0, y1 = box
        cv2.rectangle(panel, pixel((x0, y1)), pixel((x1, y0)), color, thickness)
    rectangle(ROOM, (120, 120, 120))
    for box in FURNITURE.values():
        rectangle(box, (80, 80, 80), -1)
    colors = {"proprio_only": (30, 190, 240),
              "range_360": (70, 210, 70),
              "range_front_180": (230, 190, 50),
              "range_front_270": (70, 150, 230),
              "range_360_ideal_heading": (160, 230, 160),
              "range_360_drop_2to5s": (210, 70, 220)}
    cv2.putText(panel, "Range room + estimated paths (truth for audit)",
                (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .58, (240, 240, 240), 1, cv2.LINE_AA)
    truth = np.asarray([r["native_xy_evaluator_only_m"]
                        for r in rows_by_config["proprio_only"][:step + 1]])
    if len(truth) > 1:
        cv2.polylines(panel, [np.asarray([pixel(p) for p in truth], np.int32)],
                      False, (235, 235, 235), 2)
    cv2.circle(panel, pixel(truth[-1]), 5, (255, 255, 255), -1)
    for index, (name, rows) in enumerate(rows_by_config.items()):
        path = np.asarray([r["estimated_xy_m"] for r in rows[:step + 1]])
        if len(path) > 1:
            cv2.polylines(panel, [np.asarray([pixel(p) for p in path], np.int32)],
                          False, colors[name], 2)
        cv2.circle(panel, pixel(path[-1]), 5, colors[name], -1)
        current = rows[step]
        label = name.replace("range_", "").replace("_2to5s", " loss")
        false_healthy = (current["belief_health_allows_remembered_location"] and
                         current["xy_error_evaluator_only_m"] >
                         REMEMBERED_LOCATION_ERROR_GATE_M)
        cv2.putText(panel, f"{label}: {current['xy_error_evaluator_only_m']*100:.1f}cm "
                            f"[{current['scan_status'][:12]}]"
                            f"{' FALSE HEALTH' if false_healthy else ''}",
                    (18, height - 185 + 27 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, colors[name], 1, cv2.LINE_AA)


def render_video(rows_by_config: dict[str, list[dict]], output: Path) -> int:
    source = SOURCE / "varied150-forward-seed144.mp4"
    reader = cv2.VideoCapture(str(source))
    if not reader.isOpened():
        raise RuntimeError("Cannot read native source video")
    fps = reader.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (1660, 720))
    if not writer.isOpened():
        raise RuntimeError("Cannot write native range comparison video")
    count = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        step = min(399, round(count / fps / .02))
        panel = np.zeros((720, 700, 3), dtype=np.uint8)
        draw_map(panel, rows_by_config, step)
        writer.write(np.hstack([frame, panel]))
        count += 1
    reader.release()
    writer.release()
    return count


def main() -> None:
    if json.loads((ROOT / "robot/ape-current-design.json").read_text())[
            "status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Recheck body status before replaying historical evidence")
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    varied_rows: dict[str, list[dict]] = {}
    for case in CASES:
        for name, config in CONFIGS.items():
            summary, rows = run_case(case, name, config)
            path = OUT / f"{case}-{name}-trace.json"
            path.write_text(json.dumps(rows, indent=2) + "\n")
            summary["derived_trace"] = str(path.relative_to(ROOT))
            summary["derived_trace_sha256"] = sha(path)
            summaries.append(summary)
            if case == "varied150-forward-seed144":
                varied_rows[name] = rows
    video = OUT / "varied150-range-fusion.mp4"
    frames = render_video(varied_rows, video)
    report = {
        "schema": "robot-learning-lab.native-gait-range-fusion.v1",
        "body_status": "historical_rejected_direct_frame",
        "native_body": str(BODY.relative_to(ROOT)),
        "native_body_sha256": sha(BODY),
        "native_odometry_report_sha256": sha(ODOMETRY / "report.json"),
        "analysis_code_sha256": sha(Path(__file__)),
        "scan_localizer_code_sha256": sha(ROOT / "experiments/ape_high_level/closed_loop/localization.py"),
        "room_bounds_xy_m": ROOM,
        "furniture_boxes_xy_m": FURNITURE,
        "scan_model": {"rays": SCAN_RAYS, "max_range_m": MAX_RANGE_M,
                       "noise_std_m": RANGE_NOISE_STD_M,
                       "random_ray_dropout_fraction": RANGE_DROPOUT_FRACTION,
                       "fault_seed": RANGE_FAULT_SEED,
                       "period_frames": SCAN_PERIOD_FRAMES,
                       "packet_stale_s": PACKET_STALE_S},
        "video": str(video.relative_to(ROOT)),
        "video_sha256": sha(video), "video_frames": frames,
        "scope": "Open-loop replay of retained native 50 cm gait in a declared synthetic 2D room. Simulator pose only renders idealized range and scores the result; no walking-policy response, self-occlusion, 3D material returns, person, RGB-D correction or physical sensor timing.",
        "cases": summaries,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for summary in summaries:
        print(summary["case"], summary["config"],
              summary["final_xy_error_m"], summary["scan_status_counts"],
              summary["actor_healthy_but_over_10cm_error_frames_evaluator_only"])


if __name__ == "__main__":
    main()
