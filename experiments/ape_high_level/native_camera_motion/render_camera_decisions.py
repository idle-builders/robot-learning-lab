"""Show raw mounted camera/depth packets beside the stop-gate comparison."""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from capture_native_camera import OUTPUT


def render(case: str) -> None:
    data = np.load(OUTPUT / f"{case}-world0-upper-rgbd.npz")
    result = json.loads((OUTPUT / "result.json").read_text())
    decision = next(r for r in result["rows"] if r["case"] == case and r["world"] == 0)
    with (OUTPUT / "camera-motion-packets.csv").open(newline="") as handle:
        packets = [r for r in csv.DictReader(handle) if r["case"] == case and r["world"] == "0"]
    output = OUTPUT / f"{case}-world0-rgbd-stop-decision.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, (840, 480))
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated RGB-D video")
    try:
        for i, time in enumerate(data["time_s"]):
            rgb = cv2.cvtColor(data["rgb"][i], cv2.COLOR_RGB2BGR)
            depth = data["depth_m"][i]
            near = np.clip((5 - depth) / 4.7, 0, 1)
            colored = cv2.applyColorMap(np.uint8(near * 255), cv2.COLORMAP_TURBO)
            colored[depth >= 5] = 0
            cv2.putText(rgb, "MOUNTED UPPER RGB", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        .56, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(colored, "SIMULATED DEPTH 0.3-5 M", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            panel = np.full((480, 520, 3), (27, 25, 23), dtype=np.uint8)
            def text(label: str, y: int, color=(225, 231, 235), scale=.6) -> None:
                cv2.putText(panel, label, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, color, 1, cv2.LINE_AA)
            def state(label: str, valid: bool, y: int) -> None:
                color = (125, 218, 145) if valid else (92, 99, 103)
                cv2.rectangle(panel, (22, y - 17), (41, y + 2), color, -1)
                cv2.putText(panel, label, (55, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .58, (225, 231, 235), 1, cv2.LINE_AA)
            packet = packets[i]
            text("ON-BOARD SCENE-MOTION EVIDENCE", 32, (125, 222, 247), .64)
            text(f"recorded pose t = {float(time):.2f} s", 68, scale=.72)
            text(f"stop command at {decision['stop_command_s']:.2f} s", 101)
            text(f"depth-backed matches: {packet['matched_depth_features']}", 139)
            text(f"PnP inliers: {packet['pnp_inliers']}", 168)
            text(f"registration: {packet['status']}", 197)
            delta = packet["camera_translation_m"]
            text(f"camera step: {'unavailable' if delta == '' else f'{float(delta)*1000:.2f} mm / 100 ms'}",
                 228)
            state("body-only quiet", float(time) >= decision["body_only"]["first_accept_s"], 279)
            vision = decision["rgbd_only"]["first_accept_s"]
            state("RGB-D scene quiet", vision is not None and float(time) >= vision, 314)
            fused = decision["body_plus_rgbd"]["first_accept_s"]
            state("body + RGB-D authorize view", fused is not None and float(time) >= fused, 349)
            cv2.line(panel, (22, 372), (500, 372), (70, 78, 82), 1)
            state("evaluator stationary (hidden)", float(time) >= decision["evaluator_stationary_s"], 403)
            text("visual room added only to pose replay", 449, (130, 140, 146), .48)
            writer.write(np.hstack((np.vstack((rgb, colored)), panel)))
    finally:
        writer.release()
    print(output)


if __name__ == "__main__":
    for label in ("forward-stop-early", "forward-stop-late"):
        render(label)
