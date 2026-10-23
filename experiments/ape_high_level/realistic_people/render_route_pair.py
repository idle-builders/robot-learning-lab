"""Render two saved Isaac camera/decision sequences for direct inspection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def load(run: Path):
    actions = json.loads((run / "evidence/online-action-trace.json").read_text())["actions"]
    video = cv2.VideoCapture(str(run / "evidence/ape_social_90deg.mp4"))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open {run} camera video")
    return actions, video


def render(left: Path, right: Path, output: Path, names: tuple[str, str]) -> None:
    a_left, v_left = load(left)
    a_right, v_right = load(right)
    if len(a_left) != len(a_right):
        raise ValueError("Paired sequences differ in frame count")
    width = round(v_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(v_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width <= 0 or height <= 0 or
            round(v_right.get(cv2.CAP_PROP_FRAME_WIDTH)) != width or
            round(v_right.get(cv2.CAP_PROP_FRAME_HEIGHT)) != height):
        raise ValueError("Paired videos need the same nonempty frame size")
    output.parent.mkdir(parents=True, exist_ok=True)
    header, footer = 104, 80
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             5.0, (width * 2, height + header + footer))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write {output}")
    try:
        for index, (left_action, right_action) in enumerate(zip(a_left, a_right)):
            ok_left, image_left = v_left.read()
            ok_right, image_right = v_right.read()
            if not ok_left or not ok_right:
                raise RuntimeError(f"Video ended before frame {index}")
            canvas = np.full((height + header + footer, width * 2, 3), 28,
                             dtype=np.uint8)
            for side, (image, action, name) in enumerate(((image_left, left_action,
                    names[0]), (image_right, right_action, names[1]))):
                x = side * width
                canvas[header:height + header, x:x + width] = image
                route = action.get("route_plan") or {}
                command = action["command_for_next_frame"]
                cv2.putText(canvas, name, (x + 16, 29), cv2.FONT_HERSHEY_SIMPLEX,
                            .78, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"route: {route.get('intent', '?')}  "
                            f"step: {command['forward_step_m']:+.2f} m",
                            (x + 16, 58), cv2.FONT_HERSHEY_SIMPLEX, .58,
                            (130, 230, 255) if command["forward_step_m"] else
                            (225, 225, 225), 1, cv2.LINE_AA)
                reason = route.get("reason", "")
                if reason:
                    cv2.putText(canvas, reason, (x + 16, 87),
                                cv2.FONT_HERSHEY_SIMPLEX, .56, (190, 190, 190),
                                1, cv2.LINE_AA)
                eye = action.get("proxy_eye_xyz_m_evaluator_only")
                cube = action.get("route_cube_center_xy_m_evaluator_only")
                if eye is not None and cube is not None:
                    gap = math.hypot(max(abs(eye[0] - cube[0]) - .16, 0.),
                                     max(abs(eye[1] - cube[1]) - .16, 0.))
                    cv2.putText(canvas, f"evaluator eye-cube gap: {gap * 100:.1f} cm",
                                (x + 16, height + header + 36),
                                cv2.FONT_HERSHEY_SIMPLEX, .54,
                                (100, 180, 255) if gap < .20 else (230, 230, 230),
                                1, cv2.LINE_AA)
            cv2.putText(canvas, f"Frame {index:02d}/{len(a_left) - 1:02d}",
                        (width * 2 - 125, height + header + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230),
                        1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        v_left.release()
        v_right.release()
    check = cv2.VideoCapture(str(output))
    try:
        if not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT)) != len(a_left):
            raise RuntimeError("Rendered pair failed decode/count check")
    finally:
        check.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--left-name", default="Coverage gate")
    parser.add_argument("--right-name", default="No coverage gate")
    args = parser.parse_args()
    render(args.left, args.right, args.output,
           (args.left_name, args.right_name))
    print(args.output)
