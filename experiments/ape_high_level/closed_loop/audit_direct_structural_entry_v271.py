"""V271: frozen owner-taught entry and sensor-role screen on V270 rooms."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import audit_origin_route_v152 as route
import run_color_grounded_entry_v151 as entry
from audit_chest_short_depth_v223 import ideal_mask
from audit_head_short_depth_v225 import SHORT
from direct_structural_rooms_v270 import (DirectStructuralRoom, MODES, OUT as V270_OUT,
                                          SCENES, make_room, selected_visual, target_xy)
from run import ROOT, Vision, locate_world, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v271"
V270 = V270_OUT / "structural-room-preflight.json"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
V150 = ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"


def room_factory(kind: str, mode: str, seed: int):
    def make(positions: dict):
        return DirectStructuralRoom(positions, kind=kind, mode=mode,
                                    side=1 if seed % 2 else -1)
    return make


def direct_depth_observation(entry_row: dict, frame_root: Path) -> dict:
    rows = []
    for view in entry_row["rows"]:
        box = view["accepted_box"]
        if box is None:
            continue
        path = (frame_root /
                f"seed{entry_row['seed']}-light1.00-"
                f"{'absent' if entry_row['absent'] else 'present'}" /
                f"lower-angle-{view['yaw_command_rad']:+.1f}.npz")
        with np.load(path) as saved:
            pose = (saved["camera_origin_m"].copy(),
                    saved["camera_rotation"].copy())
            depth = saved["depth_m"].copy()
        short = ideal_mask(depth, SHORT)
        long = depth.copy()
        long[(long < .08) | (long > 1.5)] = np.nan
        rows.append({"yaw_rad": view["yaw_command_rad"],
                     "rgb_floor_ray_xyz_m": locate_world(box, pose, None),
                     "short_depth_xyz_m": locate_world(box, pose, short),
                     "long_depth_xyz_m": locate_world(box, pose, long),
                     "frame_sha256": sha(path),
                     "short_valid_pixel_fraction": round(float(np.isfinite(short).mean()), 4)})
    supports = {}
    for name, field in (("short", "short_depth_xyz_m"),
                        ("long", "long_depth_xyz_m")):
        values = [np.asarray(row[field]) for row in rows if row[field] is not None]
        z = [float(v[2]) for v in values]
        supports[name] = {"valid_target_views": len(values),
                          "median_target_z_m": (None if not z else
                                                round(float(np.median(z)), 4)),
                          "floor_support": len(z) >= 2 and
                                           all(-.03 <= value <= .15 for value in z),
                          "raised_support": len(z) >= 2 and
                                            all(value > .2 for value in z)}
    return {"views": rows, "support": supports}


def masked_copy(entry_row: dict, source: Path, target: Path) -> list[dict]:
    rows = []
    branch = (f"seed{entry_row['seed']}-light1.00-"
              f"{'absent' if entry_row['absent'] else 'present'}")
    for view in entry_row["rows"]:
        name = f"lower-angle-{view['yaw_command_rad']:+.1f}.npz"
        original = source / branch / name
        output = target / branch / name
        output.parent.mkdir(parents=True, exist_ok=True)
        with np.load(original) as saved:
            payload = {key: saved[key].copy() for key in saved.files}
        payload["depth_m"] = ideal_mask(payload["depth_m"], SHORT)
        np.savez_compressed(output, **payload)
        rows.append({"source_sha256": sha(original),
                     "short_depth_sha256": sha(output)})
    return rows


def floor_entry(entry_row: dict, source: Path) -> dict:
    result = copy.deepcopy(entry_row)
    branch = source / f"seed{entry_row['seed']}-light1.00-present"
    for view in result["rows"]:
        box = view["accepted_box"]
        if box is None:
            continue
        with np.load(branch / f"lower-angle-{view['yaw_command_rad']:+.1f}.npz") as saved:
            pose = (saved["camera_origin_m"].copy(),
                    saved["camera_rotation"].copy())
        view["accepted_world_xyz_m"] = locate_world(box, pose, None)
        view["location_source"] = "RGB_box_floor_ray"
    return result


def origin_range(seed: int, kind: str) -> dict:
    room = make_room(seed, kind, "present")
    try:
        ranges = {str(height): sensed_range(room, seed, 271000 + int(height * 100),
                                            height_m=height)
                  for height in (.04, .08, .11)}
        return {"valid_returns": {height: sum(x is not None for x in scan)
                                  for height, scan in ranges.items()},
                "nearest_return_m": {height: min((x for x in scan if x is not None),
                                                 default=None)
                                     for height, scan in ranges.items()},
                "scans_m": ranges}
    finally:
        room.close()


def main() -> None:
    if not V270.exists():
        raise FileNotFoundError("Run V270 scene preflight first")
    preflight = json.loads(V270.read_text())
    if preflight["selected_visual_sha256"] != sha(selected_visual()):
        raise ValueError("Direct Frame changed since preflight")
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    previous = (entry.OUT, entry.MovedPropRoom, route.MovedPropRoom)
    rows = []
    try:
        for seed, kind in SCENES:
            for mode in MODES:
                source = OUT / "ideal" / mode
                source.mkdir(parents=True, exist_ok=True)
                entry.OUT = source
                entry.MovedPropRoom = room_factory(kind, mode, seed)
                variant = {"variant": f"direct-{kind}-{mode}",
                           "cream_cheese_xy_m": (.95, -.16),
                           "parmesan_xy_m": (.98, .18)}
                outcome = entry.episode(seed, variant, mode != "present", 1.,
                                        vision, exemplar, owner_hist)
                observation = direct_depth_observation(outcome, source)
                masked = masked_copy(outcome, source, OUT / "short" / mode)
                row = {"seed": seed, "kind": kind, "mode": mode,
                       "entry": outcome, "depth_observation": observation,
                       "masked_frame_hashes": masked}
                rows.append(row)
                (OUT / "direct-structural-entry-audit.json").write_text(
                    json.dumps({"partial": True, "rows": rows}, indent=2) + "\n")
                print(seed, kind, mode, "correct", outcome["correct_acceptances"],
                      "wrong", outcome["wrong_acceptances"],
                      "support", observation["support"], flush=True)
        routes = []
        for seed, kind in SCENES:
            source = OUT / "ideal" / "present"
            row = next(x for x in rows if x["seed"] == seed and
                       x["mode"] == "present")
            support = row["depth_observation"]["support"]["long"]
            if not support["floor_support"]:
                routes.append({"seed": seed, "kind": kind,
                               "status": "floor_support_unverified_stop"})
                continue
            floor = floor_entry(row["entry"], source)
            route.MovedPropRoom = room_factory(kind, "present", seed)
            candidate = route.screen(seed,
                                     {"variant": f"direct-{kind}-floor",
                                      "cream_cheese_xy_m": (.95, -.16),
                                      "parmesan_xy_m": (.98, .18)},
                                     floor, frame_root=OUT / "short" / "present",
                                     light_multiplier=1.)
            routes.append(candidate)
            print(seed, kind, "route", candidate["status"], flush=True)
    finally:
        entry.OUT, entry.MovedPropRoom, route.MovedPropRoom = previous
    result = {"scope": "V271 first frozen actor RGB identity entry on V270 exact Direct Frame structural rooms: same V151 owner-taught DINO>=.35/hue>=.70 five-view rule; present, ordinary absent, visually identical lookalike. RGB floor ray supplies XY only after multi-view long-depth floor-support evidence; raised or unknown support forbids route selection. Compare ideal co-aligned short 7-50cm and ideal co-aligned long 8-150cm depth on the same retained target boxes; virtual noisy72-beam 4/8/11cm range at origin is diagnostic. Evaluator truth scores correct/wrong only. No base rollout yet, no deployable optical/latency model, and exact identical impostor requires owner identity clarification.",
              "source_sha256": sha(SOURCE), "v270_sha256": sha(V270),
              "selected_visual_sha256": sha(selected_visual()),
              "thresholds": {"dino": entry.DINO_MIN,
                             "owner_hue": entry.HUE_MIN,
                             "floor_target_z_m": [-.03, .15],
                             "raised_target_z_min_m": .2,
                             "minimum_depth_views": 2},
              "origin_range": {str(seed): origin_range(seed, kind)
                               for seed, kind in SCENES},
              "rows": rows, "routes": routes}
    (OUT / "direct-structural-entry-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
