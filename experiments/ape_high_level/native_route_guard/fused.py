"""Paired low-only and mapped mid/low/depth route guards on close crossings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import cKDTree

import run as base


OUT = base.ROOT / "artifacts/ape-native-route-guard/v5-sensor-fusion"


def ray_points(name: str, packet: dict, pose: np.ndarray) -> list[np.ndarray]:
    spec = base.replay.SENSORS[name]
    yaw = base.replay.yaw_from_quat(pose[3:7])
    ch, sh = math.cos(yaw), math.sin(yaw)
    local = spec["origin"]
    origin = pose[:3] + np.array([ch*local[0]-sh*local[1],
                                   sh*local[0]+ch*local[1], local[2]])
    points = []
    index = 0
    for pitch_deg in spec["pitch_deg"]:
        pitch = math.radians(pitch_deg)
        for yaw_deg in spec["yaw_deg"]:
            distance = packet["ranges_m"][index]
            index += 1
            if distance is None:
                continue
            angle = yaw+math.radians(yaw_deg)
            point = origin + distance*np.array([math.cos(pitch)*math.cos(angle),
                                                 math.cos(pitch)*math.sin(angle),
                                                 math.sin(pitch)])
            if point[0] > pose[0]-.05 and abs(point[1]-pose[1]) <= .85:
                points.append(point)
    return points


def static_map(model, saved, name: str) -> np.ndarray:
    model.body_pos[model.body("person").id, :2] = (10., -1.)
    model.body_pos[model.body("pet").id, :2] = (10., -1.)
    data = mujoco.MjData(model)
    points = []
    for world in range(4):
        for t in np.arange(.02, 3.0, .06):
            pose = base.replay.put_pose(model, data, saved, world, float(t), 3.0)
            packet = base.ORIGINAL_READ(model, data, name, pose,
                                        "static_offpath", float(t))
            points.extend(ray_points(name, packet, pose))
    bins = {tuple(np.round(p/.02).astype(int)) for p in points}
    return np.asarray(sorted(bins), dtype=float)*.02


class ExtraLayer:
    def __init__(self, name: str, background: np.ndarray):
        self.name = name
        self.background = cKDTree(background)
        self.tracks: list[list[tuple[float, np.ndarray]]] = []
        self.previous_pose: tuple[float, np.ndarray] | None = None

    def update(self, packet: dict, pose: np.ndarray, t: float) -> dict:
        points = [p for p in ray_points(self.name, packet, pose)
                  if self.background.query(p)[0] > base.BACKGROUND_MATCH_M]
        clusters: list[list[np.ndarray]] = []
        for point in points:
            match = next((c for c in clusters
                          if any(np.linalg.norm(point[:2]-p[:2]) <= .16 for p in c)), None)
            if match is None:
                clusters.append([point])
            else:
                match.append(point)
        observations = [min(cluster, key=lambda p: p[1])[:2] for cluster in clusters]
        available = set(range(len(observations)))
        updated = []
        for track in self.tracks:
            if not available:
                continue
            choice = min(available, key=lambda i: np.linalg.norm(observations[i]-track[-1][1]))
            if np.linalg.norm(observations[choice]-track[-1][1]) <= base.TRACK_ASSOCIATION_M:
                updated.append((track+[(t, observations[choice])])[-7:])
                available.remove(choice)
        for i in available:
            updated.append([(t, observations[i])])
        self.tracks = updated

        yaw = base.replay.yaw_from_quat(pose[3:7])
        ego_xy = np.asarray(pose[:2])
        ego_v = np.array([.35*math.cos(yaw), .35*math.sin(yaw)])
        if self.previous_pose:
            pt, pxy = self.previous_pose
            if t > pt:
                ego_v = (ego_xy-pxy)/(t-pt)
        self.previous_pose = (t, ego_xy)
        details = []
        risk = False
        for track in self.tracks:
            point = track[-1][1]
            rel = point-ego_xy
            dx, dy = map(float, rel)
            mature = len(track) >= 3 and track[-1][0]-track[0][0] >= .12
            if mature:
                times = np.array([row[0] for row in track])
                xy = np.array([row[1] for row in track])
                centered = times-times.mean()
                velocity = (centered[:, None]*(xy-xy.mean(axis=0))).sum(axis=0)/(centered@centered)
                rv = velocity-ego_v
                vv = float(rv@rv)
                cpa_s = float(np.clip(-(rel@rv)/vv, 0, base.HORIZON_S)) if vv > 1e-8 else 0.
                cpa_m = float(np.linalg.norm(rel+cpa_s*rv))
                threat = dx > -.10 and cpa_m < base.RISK_RADIUS_M
            else:
                velocity = None
                cpa_m = None
                threat = 0 <= dx <= base.UNKNOWN_FORWARD_M and abs(dy) <= base.UNKNOWN_LATERAL_M
            risk |= threat
            details.append({"point_xy_m": point.round(4).tolist(),
                            "velocity_xy_mps": velocity.round(4).tolist() if mature else None,
                            "closest_approach_m": round(cpa_m, 4) if cpa_m is not None else None,
                            "risk": bool(threat)})
        return {"risk": bool(risk), "tracks": details}


LOW: base.FullScanGuard | None = None
EXTRA: dict[str, ExtraLayer] = {}
RECENT: dict[str, tuple[float, dict]] = {}


def read_fused(model, data, name, pose, case, t):
    packet = base.ORIGINAL_READ(model, data, name, pose, case, t)
    if name == "low":
        if LOW is None:
            raise RuntimeError("Missing low layer")
        low = LOW.update(packet, pose, t)
        fresh = {key: row for key, (seen, row) in RECENT.items()
                 if t-seen < .20}
        risk = low["risk"] or any(row["risk"] for row in fresh.values())
        packet["route_guard"] = {"risk": risk, "low": low, "extra": fresh}
        packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
        packet["min_forward_range_m"] = base.STOP_SIGNAL_M if risk else None
    elif name in EXTRA:
        decision = EXTRA[name].update(packet, pose, t)
        RECENT[name] = (t, decision)
        packet["route_guard"] = decision
        packet["raw_min_forward_range_m"] = packet["min_forward_range_m"]
        packet["min_forward_range_m"] = None
    return packet


def main() -> None:
    global LOW, EXTRA, RECENT
    OUT.mkdir(parents=True, exist_ok=True)
    scene_path = OUT / "scene.xml"
    scene_path.write_text(base.replay.make_scene())
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    saved = np.load(base.replay.TRACE, allow_pickle=False)
    maps = {name: static_map(model, saved, name) for name in ("low", "mid", "depth")}
    map_paths = {}
    for name, points in maps.items():
        path = OUT / f"background_{name}.npy"
        np.save(path, points)
        map_paths[name] = {"path": str(path), "sha256": base.replay.sha(path),
                           "points": len(points)}
    stop = json.loads(base.replay.STOP_REPORT.read_text())
    delays = {r["world"]: r["command_to_observed_stop_s"]
              for r in stop["rows"] if r["case"] == "forward-stop-late"}
    base.replay.hazard_xy = base.close_hazard
    base.replay.read_rays = read_fused
    base.replay.overlay = base.overlay_guard
    base.replay.STOP_THRESHOLD_M = .75
    base.replay.OUT = OUT
    results = []
    for world in range(4):
        for case in base.CASES:
            for mode in ("low15", "mid_low_depth"):
                LOW = base.FullScanGuard(maps["low"][:, :2])
                EXTRA = ({"mid": ExtraLayer("mid", maps["mid"]),
                          "depth": ExtraLayer("depth", maps["depth"])}
                         if mode == "mid_low_depth" else {})
                RECENT = {}
                result = base.replay.run_case(model, saved, world, case, mode,
                                              delays[world], video=(world == 0))
                results.append(result)
                print(world, case, mode, result["stop_command_s"],
                      result["min_modeled_body_gap_m"], flush=True)
    report = {"schema": "robot-learning-lab.native-route-guard.sensor-fusion.v1",
              "task": "T-9v3p3n",
              "purpose": "Paired low-only and map-filtered mid+low+depth motion tracks on frozen close crossings",
              "static_maps": map_paths,
              "scene_sha256": base.replay.sha(scene_path),
              "source_sha256": base.replay.sha(Path(__file__)),
              "low_guard_sha256": base.replay.sha(Path(base.__file__)),
              "native_trace_sha256": base.replay.sha(base.replay.TRACE),
              "body_sha256": base.replay.sha(base.replay.BODY),
              "mujoco_version": mujoco.__version__,
              "results": results}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    saved.close()


if __name__ == "__main__":
    main()
