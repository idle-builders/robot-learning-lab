"""Matched RGB-only owner-station control in the same furnished layout-2 scene."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import mujoco
import torch
from PIL import Image

import robocasa_owner_reacquire_rgbd_v560 as detector
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-layout2-rgb-only-v699"


def rgb_person(rgb):
    started = time.perf_counter()
    processor, model, device = detector._model()
    tensors = processor(images=Image.fromarray(rgb), text=["person ."],
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**tensors)
    found = processor.post_process_grounded_object_detection(
        output, tensors.input_ids, threshold=.25, text_threshold=.25,
        target_sizes=[rgb.shape[:2]])[0]
    detections = sorted(({"label": str(label), "score": float(score),
                          "box_xyxy": [float(value) for value in box]}
                         for label, score, box in zip(found["text_labels"],
                                                      found["scores"], found["boxes"])),
                        key=lambda row: row["score"], reverse=True)
    return {"status": "person_2d_only" if detections else "person_unconfirmed",
            "detections": detections,
            "detector_wall_s": time.perf_counter() - started,
            "metric_person_point_available": False}


def rgb_only_hook(*, model, data, renderer, actual, estimate, time_s,
                  grid, localizer, script, rng):
    model.cam_pos[model.camera("head").id, :2] = actual
    mujoco.mj_forward(model, data)
    rgb, depth, capture_wall = owner.capture(renderer, data, "head", False)
    assert depth is None
    Image.fromarray(rgb).save(OUT / "first-person-rgb-only.png")
    observed = rgb_person(rgb)
    elapsed = max(.1, capture_wall + observed["detector_wall_s"])
    guard = full.held_guard(model, data, actual, script, time_s, elapsed)
    time_s += elapsed
    row = {"stage": "initial_person_rgb_only", "time_s": round(time_s, 4),
           "capture_time_s": 0., "capture_wall_s": capture_wall,
           "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
           "actor": observed, "held_guard": guard,
           "decision": "ask_owner_for_guidance_no_metric_station"}
    return {"trace": [row],
            "frames": [owner.owner_frame(rgb, "RGB person; metric station unknown", None, None)],
            "end_time_s": time_s, "goal_xy_m": None,
            "status": "person_seen_metric_station_unknown_ask_help"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-rgb-only-declaration.v1",
                   "comparison": "Same layout2/style1 room, owner/prop positions and first head viewpoint as V697. The same released person prompt receives RGB only. With no actor-supported metric person surface, the current policy holds and asks for guidance; this does not prove all monocular methods fail.",
                   "source_sha256": route.sha256(SOURCE),
                   "full_runner_source_sha256": route.sha256(full.SOURCE),
                   "baseline_range_only_report_sha256": route.sha256(
                       ROOT / "artifacts/ape-robocasa-layout2-owner-transfer-v695/report.json"),
                   "rgbd_report_sha256": route.sha256(full.OUT / "report.json")}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old_out, old_hook = full.OUT, full.pre_route_hook
    try:
        full.OUT = OUT
        full.pre_route_hook = rgb_only_hook
        full.main()
    finally:
        full.OUT, full.pre_route_hook = old_out, old_hook
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["schema"] = "robot-learning-lab.layout2-rgb-only-control.v1"
    report["rgb_only_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    row = report["episode"]["pre_route_trace"][0]
    print(row["actor"]["status"], report["episode"]["outcome"],
          row["actor"]["detections"][:1], flush=True)


if __name__ == "__main__":
    main()
