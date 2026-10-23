"""Freeze plausible owner-sector depth returns while the mid sector is stale.

The unchanged V597 RoboCasa owner actor receives a prewarmed camera-depth
secondary guard. Only the depth bins around its own last owner track repeat
the first worker packet; all other depth bins remain live. The adapter retains
the raw rendered depth and both live and delivered packets for evaluation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/depth_guard_choice"))
import run_depth_guard as cold  # noqa: E402
import run_warm_depth_guard as warm  # noqa: E402

OUT = ROOT / "artifacts/ape-depth-guard-faults/v1"


class SectorFrozenWarmSensor(warm.WarmSensor):
    """Corrupt the delivered owner sector, never the retained raw rendering."""

    def __init__(self, folder: Path) -> None:
        super().__init__(folder)
        self.frozen: list[float | None] | None = None
        self.track_center_xy: np.ndarray | None = None
        self.injected_packets = 0
        self.masked_beams = 0

    def scan(self, model, data, xy, max_range_m, rng):
        live = super().scan(model, data, xy, max_range_m, rng)
        if self.track_center_xy is None:
            tracks = cold.multi.CAPTURED[-1].multi_history[-1]["tracks"]
            self.track_center_xy = np.asarray(max(
                tracks, key=lambda row: row["hits"])["center_xy_m"], dtype=float)
        if self.frozen is None:
            self.frozen = list(live)
        direction = self.track_center_xy - np.asarray(xy)
        bearing = math.atan2(direction[1], direction[0])
        angles = np.linspace(-math.pi, math.pi, len(live), endpoint=False)
        delta = (angles - bearing + math.pi) % (2. * math.pi) - math.pi
        mask = np.abs(delta) <= math.radians(40.)
        delivered = [self.frozen[i] if mask[i] else value
                     for i, value in enumerate(live)]
        self.injected_packets += 1
        self.masked_beams += int(mask.sum())
        self.logs[-1].update({
            "fault": "owner_sector_repeats_first_depth_packet",
            "actor_track_center_xy_m": self.track_center_xy.round(5).tolist(),
            "owner_sector_half_width_degrees": 40.,
            "frozen_owner_beams": int(mask.sum()),
            "live_owner_sector_return_count": int(sum(
                mask[i] and live[i] is not None for i in range(len(live)))),
            "delivered_owner_sector_return_count": int(sum(
                mask[i] and delivered[i] is not None for i in range(len(live)))),
            "live_to_delivered_changed_beams": int(sum(
                a != b for a, b in zip(live, delivered))),
            "actor_delivered_packet_72_world_bearings_m": delivered,
        })
        return delivered

    def save(self) -> dict:
        saved = super().save()
        saved.update({"owner_sector_fault_packets": self.injected_packets,
                      "frozen_owner_sector_beams_total": self.masked_beams,
                      "owner_sector_half_width_degrees": 40.,
                      "track_center_xy_actor": self.track_center_xy.round(5).tolist()
                      if self.track_center_xy is not None else None})
        return saved


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(cold.joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(cold.joined.PRIOR)
    old_out, old_joined_out = cold.OUT, cold.joined.OUT
    old_class = cold.DepthSectorSensor
    old_owner_scan = cold.baseline.owner_scan

    def owner_scan(*args, **kwargs):
        underlying = old_owner_scan(*args, **kwargs)

        def scan(model, data, xy, max_range_m, rng):
            sensor = warm.WarmSensor.latest
            if sensor is not None:
                sensor.warm(model, data)
            return underlying(model, data, xy, max_range_m, rng)

        return scan

    cold.OUT = OUT
    cold.joined.OUT = OUT
    cold.joined.owner_turn = cold.turn.owner_turn
    cold.route.MARGIN_M = .15
    cold.DepthSectorSensor = SectorFrozenWarmSensor
    cold.baseline.owner_scan = owner_scan
    vision = cold.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(cold.owner.TEACH_CROP).convert("RGB")])[0]
    try:
        episode, geometry = cold.run_case("moving_frozen_mid", case,
                                          prior, policy, vision, exemplar)
    finally:
        cold.OUT, cold.joined.OUT = old_out, old_joined_out
        cold.DepthSectorSensor = old_class
        cold.baseline.owner_scan = old_owner_scan
        cold.route.MARGIN_M = .10
        cold.route.LIDAR_Z_M = .30
    report = {"schema": "robot-learning-lab.shared-owner-sector-fault.v1",
              "task": "T-n584kj",
              "scope": "Same late-moving-owner RoboCasa route/turn as the warm depth control; both 30cm virtual range and prewarmed owner-view depth retain plausible first-packet values only in the actor-tracked owner sector during slow person recognition. Other depth bins remain live. Fault is injected by evaluator; actor receives only the delivered packet. Kinematic body/scripted owner/ideal optics.",
              "source_sha256": {"runner": cold.sha(Path(__file__)),
                                "cold_depth_adapter": cold.sha(Path(cold.__file__)),
                                "warm_depth_adapter": cold.sha(Path(warm.__file__)),
                                "owner_turn": cold.sha(Path(cold.turn.__file__)),
                                "joined_route": cold.sha(Path(cold.joined.__file__)),
                                "geometry_audit": cold.sha(Path(cold.audit.__file__)),
                                "config": cold.sha(cold.joined.CONFIG),
                                "prior_map": cold.sha(cold.joined.PRIOR)},
              "episode": episode, "geometry_audit": geometry}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    cancelled = [row for row in episode["trace"]
                 if row.get("stage") == "owner_reacquire_image_cancelled"]
    print("shared_sector_fault", episode["outcome"],
          "cancelled", len(cancelled),
          "minimum_person_gap_m", geometry["person_exact_nearest"]["gap_m"],
          "depth_fault_packets", episode["depth_guard_sensor"]["owner_sector_fault_packets"],
          flush=True)


if __name__ == "__main__":
    main()
