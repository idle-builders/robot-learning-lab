"""Join typed planar scans to the existing RoboCasa owner re-show loop.

The V410 actor remains responsible for image recognition, ask, fresh-track and
outcome. This adapter changes only its range source, map update and scan
matching decimation. It does not simulate hardware acquisition timing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, sha
import run_robocasa_joined_owner_route_v410 as joined
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_spec_planar_route_v202p import (DEVICES, MAX_RANGE_M, StatusGrid,
                                         packet)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-spec-planar-joined-v204p"
RESPONSES = ("compliant", "ignored", "withdrawn")


def run_case(device: str, response: str, vision, exemplar) -> dict:
    profile = DEVICES[device]
    branch = OUT / device / response
    branch.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    active = {"status": None, "ranges": None}

    def scan(model, data, xy, max_range_m, rng):
        if max_range_m != MAX_RANGE_M:
            raise AssertionError("Joined actor range cap changed")
        values, status = packet(model, data, xy, profile, rng)
        active["status"], active["ranges"] = status, values
        path = branch / f"range-{len(records):03d}.npz"
        np.savez_compressed(path, ranges_m=values, status=status,
                            sample_index=len(records))
        records.append({
            "packet_path": str(path.relative_to(ROOT)),
            "packet_sha256": sha(path),
            "status_counts": {
                "valid_hit": int(np.count_nonzero(status == 0)),
                "too_near": int(np.count_nonzero(status == 1)),
                "no_target": int(np.count_nonzero(status == 2)),
            },
        })
        return [float(v) if s == 0 else None
                for v, s in zip(values, status)]

    class Grid(StatusGrid):
        def update(self, ranges, pose, max_range_m):
            if active["status"] is None or len(ranges) != len(active["status"]):
                raise AssertionError("Range packet and status lost alignment")
            self.update_packet(active["ranges"], active["status"], pose)

    class Localizer(RangeLocalizer):
        def correct(self, ranges, predicted):
            stride = {"legacy_72": 1, "a1_spec_proxy": 4,
                      "a2_spec_proxy": 8}[device]
            return super().correct(ranges[::stride], predicted)

    old = (joined.planar_scan, joined.SensorGrid,
           joined.RangeLocalizer, joined.OUT)
    joined.planar_scan = scan
    joined.SensorGrid = Grid
    joined.RangeLocalizer = Localizer
    joined.OUT = branch
    try:
        result = joined.episode(response, True, vision, exemplar)
    finally:
        joined.planar_scan, joined.SensorGrid, joined.RangeLocalizer, joined.OUT = old
    route_rows = [row for row in result["trace"] if row["stage"] == "navigate"]
    if len(route_rows) != len(records):
        raise AssertionError("Saved scan does not match route observations")
    result["device"] = device
    result["sensor_packets"] = records
    result["route_packet_count"] = len(records)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference body status changed")
    vision = joined.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(joined.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    for response in RESPONSES:
        for device in DEVICES:
            result = run_case(device, response, vision, exemplar)
            episodes.append(result)
            print(device, response, result["outcome"],
                  result["route_packet_count"],
                  round(result["final_pose_error_m"], 3), flush=True)
    report = {
        "schema": "robot-learning-lab.ape-spec-planar-joined.v1",
        "source_sha256": sha(SOURCE),
        "v202_source_sha256": sha(Path(__file__).with_name(
            "run_spec_planar_route_v202p.py")),
        "v410_source_sha256": sha(Path(joined.__file__)),
        "current_design_status": pointer["status"],
        "taught_crop_sha256": sha(joined.TEACH_CROP),
        "device_profiles": DEVICES,
        "scope": "Same V410 layout2 kinematic eye rig, 30cm accumulated "
                 "slip, scripted owner responses and released RGB/RGB-D "
                 "recognition. Typed 5m planar first-hit packets feed 2-D "
                 "map and scan matcher; invalid rays remain unknown. The "
                 "existing loop's wall-time action clock remains unchanged: "
                 "this joined adapter does not model A1/A2 scan cadence, "
                 "rolling acquisition, self occlusion, moving body, owner "
                 "person, independent near-body stop or device mounting. "
                 "Exact item confirmation is a scripted owner reply.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
