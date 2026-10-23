"""Render a low ToF cube challenge beside its synchronized RGB video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from near_body_tof import zone_rays


DEFAULT_RUN = Path("artifacts/ape-realistic-people/isaac61-midroom-held-low-tof-v217b")
FLOOR = -.8154
SCALE = 300
OX, OY = 240, 300


def pixel(point_xy, eye_xy):
    return (round(OX + (point_xy[0] - eye_xy[0]) * SCALE),
            round(OY - (point_xy[1] - eye_xy[1]) * SCALE))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--start-frame", type=int, default=8)
    parser.add_argument("--end-frame", type=int, default=18)
    parser.add_argument("--camera-fov", type=int, choices=(70, 90), default=90)
    args = parser.parse_args()
    evidence = args.run / "evidence"
    tof = json.loads((evidence / "low-tof-trace.json").read_text())["samples"]
    fan = json.loads((evidence / "range-fan-trace.json").read_text())["frames"]
    audit = json.loads((args.run / "low-tof-intrusion-audit.json").read_text())["rows"]
    response_file = evidence / "tof-response-trace.json"
    response = (json.loads(response_file.read_text())["samples"]
                if response_file.exists() else None)
    cap = cv2.VideoCapture(str(evidence / f"ape_social_{args.camera_fov}deg.mp4"))
    action_file = evidence / "online-action-trace.json"
    actions = (json.loads(action_file.read_text())["actions"]
               if action_file.exists() else None)
    camera = []
    while True:
        good, image = cap.read()
        if not good:
            break
        camera.append(cv2.resize(image, (480, 360)))
    cap.release()
    assert len(camera) == 36 and len(tof) == len(audit)
    output = args.run / f"low-tof-intrusion-explainer-{args.run.name.rsplit('-', 1)[-1]}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (960, 360))
    for frame in range(args.start_frame, args.end_frame):
        stamp = fan[frame]["timeline_time_s"]
        sample_index = max(index for index, row in enumerate(tof)
                           if row["sim_time_s"] <= stamp + 1e-6)
        sample = tof[sample_index]
        row = audit[sample_index]
        active_flank = max(("front_left", "front_right"), key=lambda name:
            row["sensors"][name]["cube_hit_zones_evaluator_only"])
        eye = np.asarray(sample["eye_xy_m_evaluator_only"])
        cube = np.asarray(sample["cube_center_xy_m_evaluator_only"])
        panel = np.full((360, 480, 3), (22, 25, 29), dtype=np.uint8)
        cv2.circle(panel, (OX, OY), round(.2 * SCALE), (45, 45, 175), 2)
        x0, y0 = pixel(cube + [-.16, .16], eye)
        x1, y1 = pixel(cube + [.16, -.16], eye)
        cv2.rectangle(panel, (x0, y0), (x1, y1), (50, 95, 235), 1)
        for name, origin, directions in zone_rays(eye,
                sample["body_heading_deg_commanded"], FLOOR):
            if name not in (active_flank, "front_center"):
                continue
            sensor = sample["sensors"][name]
            for distance, cube_hit, direction in zip(sensor["range_m_by_zone"],
                    sensor["cube_hit_by_zone_evaluator_only"], directions):
                if distance is None:
                    continue
                xyz = np.asarray(origin) + distance * np.asarray(direction)
                if xyz[2] < FLOOR + .03:
                    continue
                x, y = pixel(xyz[:2], eye)
                if 0 <= x < 480 and 0 <= y < 360:
                    cv2.circle(panel, (x, y), 2,
                               (80, 220, 120) if cube_hit else (140, 150, 160), -1)
            cv2.circle(panel, pixel(origin[:2], eye), 5,
                       (80, 220, 120) if name == active_flank else (220, 170, 70), -1)
        cv2.circle(panel, (OX, OY), 5, (232, 200, 30), -1)
        true_gap = row["cube_gap_m_evaluator_only"] * 100
        observed = row["sensors"][active_flank][
            "nearest_cube_eye_radial_m_evaluator_only"]
        cv2.putText(panel, "LOW 8x8 ToF RAY SCREEN - top view", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(panel, f"True box gap: {true_gap:.1f} cm", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, .53, (245, 245, 245), 1, cv2.LINE_AA)
        observed_label = (f"{observed * 100:.1f} cm" if observed is not None
                          else "unobserved")
        cv2.putText(panel, f"Nearest cube point: {observed_label}", (10, 74),
                    cv2.FONT_HERSHEY_SIMPLEX, .53, (245, 245, 245), 1, cv2.LINE_AA)
        if response is not None:
            response_row = max((entry for entry in response
                                if entry["time_s"] <= stamp + 1e-6),
                               key=lambda entry: entry["time_s"])
            status = response_row["response"]["state"]
            moved_cm = max(entry["response"].get("escape_total_m",
                             entry["response"].get("retreat_total_m", 0.))
                           for entry in response if entry["time_s"] <= stamp + 1e-6) * 100
            cv2.putText(panel, f"Response: {status}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(panel, f"Escaped: {moved_cm:.1f} cm", (10, 123),
                        cv2.FONT_HERSHEY_SIMPLEX, .47, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(panel, "Green hits: evaluator cube label", (10, 345),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (225, 225, 225), 1, cv2.LINE_AA)
        left = camera[frame].copy()
        cv2.putText(left, f"SIM RGB frame {frame}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        if actions is not None:
            social = actions[frame].get("social_actor") or {}
            cv2.putText(left, f"Item: {social.get('intent', 'none')}", (12, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        image = np.hstack((left, panel))
        for _ in range(5):
            writer.write(image)
    writer.release()
    print(output)


if __name__ == "__main__":
    main()
