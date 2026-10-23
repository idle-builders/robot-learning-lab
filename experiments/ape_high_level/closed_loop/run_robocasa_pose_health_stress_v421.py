"""Challenge the frozen V419 pose-health gate with new scan perturbations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_robocasa_layout2_localized_route_v408 as actor
from run_robocasa_layout2_route_v407 import frame_with_map as layout2_frame
from run_robocasa_layout2_route_v407 import model_with_layout2, sha256
from run_robocasa_layout4_transfer_v412 import frame_layout4, layout4_scene
from run_robocasa_pose_health_gate_v419 import HealthGrid, HealthLocalizer, HEALTH, original_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-pose-health-stress-v421"


def perturbed_scan(mode: str):
    frame = 0

    def read(model, data, xy, max_range_m, rng):
        nonlocal frame
        if mode == "ramp_to5deg":
            bias_deg = min(5., max(0., frame - 20) * .25)
            origin = np.array([xy[0], xy[1], .30], dtype=np.float64)
            scan = []
            for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
                actual_angle = angle + math.radians(bias_deg)
                ray = np.array([math.cos(actual_angle), math.sin(actual_angle), 0.], np.float64)
                geom_id = np.array([-1], np.int32)
                distance = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geom_id)
                noise = rng.normal(0., .005)
                scan.append(float(max(.01, distance + noise))
                            if .14 < distance < max_range_m else None)
        else:
            scan = original_scan(model, data, xy, max_range_m, rng)
            if mode == "extra_15mm_noise":
                perturbed = [(value + rng.normal(0., .015) if value is not None else None)
                             for value in scan]
                scan = [(float(value) if value is not None and .14 < value < max_range_m
                         else None) for value in perturbed]
            elif mode == "sparse_10pct":
                scan = [(None if value is not None and rng.random() < .10 else value)
                        for value in scan]
        frame += 1
        return scan

    return read


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.RangeLocalizer = HealthLocalizer
    actor.SensorGrid = HealthGrid
    cases = []
    for layout in (2, 4):
        for mode in ("extra_15mm_noise", "sparse_10pct", "ramp_to5deg"):
            actor.model_with_layout2 = model_with_layout2 if layout == 2 else layout4_scene
            actor.frame_with_map = layout2_frame if layout == 2 else frame_layout4
            actor.planar_scan = perturbed_scan(mode)
            actor.OUT = OUT / f"layout{layout}-{mode}"
            episode = actor.episode(5., .30, "range_growing")
            cases.append({"layout": layout, "mode": mode,
                          "health_trigger": HEALTH["trigger"], "episode": episode})
            print(layout, mode, episode["outcome"], episode["steps"],
                  round(episode["end_pose_error_m"], 3),
                  HEALTH["trigger"]["frame"] if HEALTH["trigger"] else None, flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_gate_source_sha256": sha256(Path(__file__).with_name("run_robocasa_pose_health_gate_v419.py")),
              "frozen_actor_source_sha256": sha256(Path(actor.__file__)),
              "stress": {"extra_noise_std_m": .015, "valid_beam_dropout_probability": .10,
                         "gradual_heading_bias_deg": "0 to +5 from frames20 to40"},
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
