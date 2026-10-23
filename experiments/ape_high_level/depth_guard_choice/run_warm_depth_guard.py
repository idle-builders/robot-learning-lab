"""Prewarm the same depth guard before slow person inference begins."""

from __future__ import annotations

import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image

import run_depth_guard as cold

OUT = cold.ROOT / "artifacts/ape-depth-guard-choice/warm-v2"


class WarmSensor(cold.DepthSectorSensor):
    latest: "WarmSensor | None" = None

    def __init__(self, folder: Path) -> None:
        super().__init__(folder)
        self.prewarm: dict | None = None
        WarmSensor.latest = self

    def warm(self, model, data) -> None:
        if self.prewarm is not None:
            return
        start = time.perf_counter()
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.model_id = id(model)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=cold.interaction.CAMERA)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        self.prewarm = {"wall_s": round(time.perf_counter() - start, 5),
                        "depth_sha256": cold.hashlib.sha256(depth.tobytes()).hexdigest(),
                        "time": "first owner-turn mid-height packet, before person detector worker"}

    def save(self) -> dict:
        saved = super().save()
        saved["prewarm"] = self.prewarm
        return saved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_out, old_joined_out = cold.OUT, cold.joined.OUT
    old_class = cold.DepthSectorSensor
    old_owner_scan = cold.baseline.owner_scan

    def owner_scan(*args, **kwargs):
        scan = old_owner_scan(*args, **kwargs)

        def wrapped(model, data, xy, max_range_m, rng):
            sensor = WarmSensor.latest
            if sensor is not None:
                sensor.warm(model, data)
            return scan(model, data, xy, max_range_m, rng)

        return wrapped

    policy = json.loads(cold.joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(cold.joined.PRIOR)
    cold.OUT = OUT
    cold.joined.OUT = OUT
    cold.joined.owner_turn = cold.turn.owner_turn
    cold.route.MARGIN_M = .15
    cold.DepthSectorSensor = WarmSensor
    cold.baseline.owner_scan = owner_scan
    vision = cold.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(cold.owner.TEACH_CROP).convert("RGB")])[0]
    try:
        episode, geometry = cold.run_case("moving_frozen_mid", case, prior,
                                          policy, vision, exemplar)
    finally:
        cold.OUT, cold.joined.OUT = old_out, old_joined_out
        cold.DepthSectorSensor = old_class
        cold.baseline.owner_scan = old_owner_scan
        cold.route.MARGIN_M = .10
        cold.route.LIDAR_Z_M = .30
    report = {"schema": "robot-learning-lab.depth-guard-choice-warm.v2",
              "task": "T-hwtsgv",
              "scope": "Same V1 depth packet and actor, with depth renderer started on the first owner-turn mid-height range packet before detector inference. This isolates simulator cold initialization from recurring depth capture cost; no physical camera warm-up guarantee.",
              "source_sha256": {"wrapper": cold.sha(Path(__file__)),
                                "cold_runner": cold.sha(Path(cold.__file__)),
                                "owner_turn": cold.sha(Path(cold.turn.__file__)),
                                "joined_route": cold.sha(Path(cold.joined.__file__)),
                                "geometry_audit": cold.sha(Path(cold.audit.__file__))},
              "episode": episode, "geometry_audit": geometry}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("warm_depth", episode["outcome"],
          geometry["person_exact_nearest"]["gap_m"],
          episode["depth_guard_sensor"]["maximum_packet_start_gap_wall_s"], flush=True)


if __name__ == "__main__":
    main()
