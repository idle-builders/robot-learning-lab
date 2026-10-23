"""Fast collision screen for short rigid elbow lips over the saved elbow STEP."""

from __future__ import annotations

import build123d as bd

from elbow_pitch_iteration import (AXIS, CASE_SIZE, MOTOR_CENTER, OUT,
                                   arc_cowl, flexed, volume)


def main() -> None:
    fixed = bd.import_step(OUT / "upper-arm-with-elbow-sleeve.step")
    moving_parts = [bd.import_step(OUT / name) for name in (
        "paw-inner-with-elbow-cheek.step", "paw-outer-with-elbow-cheek.step",
        "elbow-horn-1.step", "elbow-horn-2.step",
        "output-pulley-64T-envelope.step", "output-shaft-envelope.step")]
    motor = bd.Pos(*MOTOR_CENTER) * bd.Box(*CASE_SIZE)
    poses = (0, 30, 60, 90, 115)
    for fixed_high in (20, 35, 50):
        upper = arc_cowl(28, 30, -30, fixed_high, 82, 28)
        fixed_clashes = [sum(volume(upper.intersect(flexed(part, pose)))
                             for part in moving_parts) for pose in poses]
        print(f"fixed high {fixed_high}: core overlaps {list(zip(poses, [round(n,1) for n in fixed_clashes]))}", flush=True)
    for moving_low in (105, 120, 135):
        lower = arc_cowl(25.5, 26.5, moving_low, 180, 84, 24)
        fixed_clashes = [volume(flexed(lower, pose).intersect(fixed)) for pose in poses]
        motor_clashes = [volume(flexed(lower, pose).intersect(motor)) for pose in poses]
        print(f"moving low {moving_low}: fixed {list(zip(poses, [round(n,1) for n in fixed_clashes]))}; motor {list(zip(poses, [round(n,1) for n in motor_clashes]))}", flush=True)


if __name__ == "__main__":
    main()
