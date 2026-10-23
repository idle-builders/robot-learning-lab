"""Height-ablate typed planar observations in the live moving-owner loop.

V471's route, motion forecast, independent mid-height guard, RGB-D owner
recognition and kinematic action stay intact. Only the moving prop height and
actor route-scan profiles change. The low ring is an optimistic virtual bound.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from run import ROOT, sha
import run_robocasa_dynamic_owner_v471 as dynamic
import run_robocasa_layout2_route_v407 as route
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_spec_planar_route_v202p import StatusGrid


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-sensor-v206p"
PROP_CENTER_Z_M = .09
PROP_HALF_HEIGHT_M = .09
RANGE_CAP_M = 5.
NOISE_STD_M = .005
PROFILES = {
    "mid_a1": ((.30, 360, .15),),
    "low_a1": ((.10, 360, .15),),
    "mid_a1_low72_virtual": ((.30, 360, .15), (.10, 72, .06)),
}
CASES = ("stationary", "returning_crossing", "one_way_crossing")


def low_scene(start_xy):
    model = _ORIGINAL_SCENE(start_xy)
    model.body_pos[model.body("route_obstacle").id, 2] = PROP_CENTER_Z_M
    model.geom("route_obstacle_geom").size[1] = PROP_HALF_HEIGHT_M
    return model


def plane(model, data, xy, height_m, beams, minimum_m, rng):
    origin = np.array((xy[0], xy[1], height_m), dtype=np.float64)
    values = np.full(beams, np.nan, dtype=np.float32)
    statuses = np.full(beams, 2, dtype=np.uint8)
    for index in range(beams):
        angle = -math.pi + 2 * math.pi * index / beams
        direction = np.array((math.cos(angle), math.sin(angle), 0.),
                             dtype=np.float64)
        hit_geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, direction,
                                 None, 1, -1, hit_geom)
        if distance < 0 or distance > RANGE_CAP_M:
            continue
        if distance < minimum_m:
            statuses[index] = 1
            continue
        statuses[index] = 0
        values[index] = max(minimum_m,
                            distance + rng.normal(0., NOISE_STD_M))
    return values, statuses


def fuse(scans, beams):
    values = np.full(beams, np.nan, dtype=np.float32)
    statuses = np.full(beams, 2, dtype=np.uint8)
    for source_values, source_status in scans:
        stride = beams // len(source_values)
        if stride * len(source_values) != beams:
            raise AssertionError("Sensor angles are not co-aligned")
        for source_index in range(len(source_values)):
            index = stride * source_index
            status = int(source_status[source_index])
            if status == 0:
                value = float(source_values[source_index])
                if statuses[index] != 0 or value < values[index]:
                    values[index], statuses[index] = value, 0
            elif status == 1 and statuses[index] != 0:
                statuses[index] = 1
    return values, statuses


_ORIGINAL_SCENE = dynamic.scene_with_owner_and_prop


def one(profile_name, case, prior, vision, exemplar):
    sources = PROFILES[profile_name]
    branch = OUT / profile_name
    branch.mkdir(parents=True, exist_ok=True)
    records = []
    active = {"ranges": None, "statuses": None}

    def scan(model, data, xy, max_range_m, rng):
        if max_range_m != RANGE_CAP_M:
            raise AssertionError("Actor range cap changed")
        raw = [plane(model, data, xy, h, beams, minimum, rng)
               for h, beams, minimum in sources]
        ranges, statuses = fuse(raw, max(s[1] for s in sources))
        active["ranges"], active["statuses"] = ranges, statuses
        index = len(records)
        path = branch / case["id"] / f"scan-{index:03d}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ranges_m=ranges, status=statuses,
                            sample_index=index,
                            decision_time_s=index * .4)
        records.append({
            "path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "status_counts": {
                "valid_hit": int(np.count_nonzero(statuses == 0)),
                "too_near": int(np.count_nonzero(statuses == 1)),
                "no_target": int(np.count_nonzero(statuses == 2)),
            },
        })
        return [float(value) if status == 0 else None
                for value, status in zip(ranges, statuses)]

    class Grid(StatusGrid):
        def update(self, ranges, pose, max_range_m):
            if active["statuses"] is None or len(ranges) != len(active["statuses"]):
                raise AssertionError("Typed scan lost alignment")
            self.update_packet(active["ranges"], active["statuses"], pose)

    class Localizer(RangeLocalizer):
        def correct(self, ranges, predicted):
            stride = 4 if len(ranges) == 360 else 1
            return super().correct(ranges[::stride], predicted)

    original = (dynamic.scene_with_owner_and_prop, dynamic.SCAN,
                dynamic.RangeLocalizer, dynamic.OUT, route.SensorGrid)
    dynamic.scene_with_owner_and_prop = low_scene
    dynamic.SCAN = scan
    dynamic.RangeLocalizer = Localizer
    dynamic.OUT = branch
    route.SensorGrid = Grid
    try:
        result = dynamic.episode(case, "compliant", True, prior,
                                 vision, exemplar)
    finally:
        (dynamic.scene_with_owner_and_prop, dynamic.SCAN,
         dynamic.RangeLocalizer, dynamic.OUT, route.SensorGrid) = original
    if len(records) != len(result["route_episode"]["trace"]):
        raise AssertionError("One saved scan per route decision is required")
    result["profile"] = profile_name
    result["sensor_packets"] = records
    result["low_prop_geometry"] = {
        "center_z_m": PROP_CENTER_Z_M,
        "half_height_m": PROP_HALF_HEIGHT_M,
        "radius_m": .24,
    }
    return result


def main():
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Reference-body status changed")
    cases = {case["id"]: case for case in json.loads(dynamic.CONFIG.read_text())["cases"]}
    prior = np.load(dynamic.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    old_margin = route.MARGIN_M
    route.MARGIN_M = .15
    vision = dynamic.owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(dynamic.owner.TEACH_CROP).convert("RGB")])[0]
    episodes = []
    try:
        for case_id in CASES:
            for profile_name in PROFILES:
                result = one(profile_name, cases[case_id], prior, vision,
                             exemplar)
                episodes.append(result)
                print(case_id, profile_name, result["route_state"],
                      result["outcome"],
                      round(result["route_episode"]["min_true_prop_gap_m"], 3),
                      flush=True)
    finally:
        route.MARGIN_M = old_margin
    report = {
        "schema": "robot-learning-lab.ape-low-prop-sensor-join.v1",
        "source_sha256": sha(SOURCE),
        "v471_source_sha256": sha(Path(dynamic.__file__)),
        "config_sha256": sha(dynamic.CONFIG),
        "prior_sha256": sha(dynamic.PRIOR),
        "current_design_status": pointer["status"],
        "profiles": PROFILES,
        "prop_geometry": {"center_z_m": PROP_CENTER_Z_M,
                          "half_height_m": PROP_HALF_HEIGHT_M,
                          "radius_m": .24},
        "scope": "Same V471 layout4 scripted moving-owner task, but a "
                 "lower 18cm prop. Typed ideal first-hit scans are taken "
                 "at each decision instant with 5mm injected noise; A1 "
                 "angular/minimum-range geometry is a spec proxy, not its "
                 "actual optical/cadence/noise model. The optional 72-ray "
                 "low plane is an optimistic virtual companion. The "
                 "unchanged independent 30cm 20ms guard is blind to this "
                 "low prop. No self occlusion, qualified body stop, owner "
                 "person or physical sensor mount is simulated.",
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
