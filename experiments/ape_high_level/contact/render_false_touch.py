"""Show a native no-contact reach with the frozen jaw-position rule overlaid."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STEM = (ROOT / "artifacts/ape-cove50-side-reach-20261009"
        / "reach-arm-residual-v1/stress/stress-static-diagonal")
THRESHOLDS = (ROOT / "artifacts/ape-cove50-contact-20261009"
              / "selected-reach-audit.json")
OUT = (ROOT / "artifacts/ape-cove50-contact-20261012"
       / "native-no-contact-false-touch.mp4")


def main() -> None:
    threshold = json.loads(THRESHOLDS.read_text())["thresholds"][
        "jaw_joint_position_rad"]["threshold"]
    with np.load(STEM.with_suffix(".npz"), allow_pickle=False) as data:
        index = list(data["action_names"]).index("right_gripper")
        jaw = data["actor_observation"][:, index].astype(float)
        contact = data["finger_toy_contact"].astype(bool)
        dt = float(data["step_dt_s"])
    if contact.any():
        raise ValueError("The chosen evidence clip is no longer a no-contact case")
    cap = cv2.VideoCapture(str(STEM.with_suffix(".mp4")))
    if not cap.isOpened():
        raise RuntimeError("Could not decode native replay video")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height + 110))
    if not writer.isOpened():
        raise RuntimeError("Could not encode annotated video")
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sample = min(len(jaw) - 1, round((count / fps) / dt))
        predicted = jaw[sample] >= threshold
        canvas = np.zeros((height + 110, width, 3), dtype=np.uint8)
        canvas[110:] = frame
        cv2.putText(canvas, "Native reach | historical Direct Frame", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"t={count / fps:.2f}s | jaw rule: " +
                    ("TOUCH" if predicted else "no touch"), (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .76,
                    (30, 80, 255) if predicted else (180, 220, 180), 2,
                    cv2.LINE_AA)
        cv2.putText(canvas, "Actual finger/toy contact: none (evaluator only)",
                    (12, 92), cv2.FONT_HERSHEY_SIMPLEX, .59,
                    (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(canvas)
        count += 1
    cap.release()
    writer.release()
    if count == 0:
        raise RuntimeError("Native replay video had no frames")
    print(OUT, "frames", count, "fps", fps)


if __name__ == "__main__":
    main()
