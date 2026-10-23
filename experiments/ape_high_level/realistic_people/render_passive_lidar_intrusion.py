"""Render the fixed-eye lidar blind-zone challenge beside the RGB simulation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


RUN = Path("artifacts/ape-realistic-people/isaac61-midroom-held-probe-passive-os1-v216")
FRAMES = range(7, 17)
SCALE = 300
OX, OY = 240, 300
FLOOR = -.8154


def xy_pixel(point_xy, eye_xy):
    return (round(OX + (point_xy[0] - eye_xy[0]) * SCALE),
            round(OY - (point_xy[1] - eye_xy[1]) * SCALE))


def main() -> None:
    evidence = RUN / "evidence"
    lidar = np.load(evidence / "rtx-lidar-three-scans.npz")
    actions = json.loads((evidence / "online-action-trace.json").read_text())["actions"]
    audit = json.loads((RUN / "passive-rtx-low-intrusion-audit.json").read_text())["rows"]
    video = cv2.VideoCapture(str(evidence / "ape_social_90deg.mp4"))
    camera_frames = []
    while True:
        good, image = video.read()
        if not good:
            break
        camera_frames.append(cv2.resize(image, (480, 360)))
    video.release()
    assert len(camera_frames) == 36
    output = RUN / "passive-lidar-intrusion-explainer-v216.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (960, 360))
    for frame in FRAMES:
        action = actions[frame]
        eye = np.asarray(action["proxy_eye_xyz_m_evaluator_only"])
        cube = np.asarray(action["route_cube_center_xy_m_evaluator_only"])
        az = np.deg2rad(lidar[f"frame_{frame:02d}_azimuth_deg"])
        el = np.deg2rad(lidar[f"frame_{frame:02d}_elevation_deg"])
        radius = lidar[f"frame_{frame:02d}_range_m"]
        points = np.column_stack((radius * np.cos(el) * np.cos(az),
                                  radius * np.cos(el) * np.sin(az),
                                  radius * np.sin(el))) + eye
        low = np.isfinite(points).all(axis=1) & (points[:, 2] >= FLOOR + .03) & (
            points[:, 2] <= FLOOR + .32) & (np.linalg.norm(
                points[:, :2] - eye[:2], axis=1) <= 1.2)
        panel = np.full((360, 480, 3), (22, 25, 29), dtype=np.uint8)
        cv2.circle(panel, (OX, OY), round(.4 * SCALE), (65, 65, 65), 1)
        cv2.circle(panel, (OX, OY), round(.2 * SCALE), (45, 45, 165), 2)
        for point in points[low]:
            x, y = xy_pixel(point[:2], eye[:2])
            if 0 <= x < 480 and 0 <= y < 360:
                cv2.circle(panel, (x, y), 1, (180, 195, 206), -1)
        x0, y0 = xy_pixel(cube + np.array([-.16, .16]), eye[:2])
        x1, y1 = xy_pixel(cube + np.array([.16, -.16]), eye[:2])
        cv2.rectangle(panel, (x0, y0), (x1, y1), (45, 95, 235), 2)
        cv2.circle(panel, (OX, OY), 5, (232, 200, 30), -1)
        gap_cm = audit[frame]["cube_gap_m_evaluator_only"] * 100
        returned_cm = audit[frame]["rtx_low_cube_nearest_horizontal_m_evaluator_only"] * 100
        cv2.putText(panel, "PASSIVE HEAD LIDAR - top view", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(panel, f"True cube-box gap: {gap_cm:.1f} cm", (12, 53),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Nearest cube return: {returned_cm:.1f} cm", (12, 77),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(panel, "Red circle: 20 cm eye proxy", (12, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (225, 225, 225), 1, cv2.LINE_AA)
        left = camera_frames[frame].copy()
        cv2.putText(left, f"SIM RGB frame {frame}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        image = np.hstack((left, panel))
        for _ in range(5):
            writer.write(image)
    writer.release()
    print(output)


if __name__ == "__main__":
    main()
