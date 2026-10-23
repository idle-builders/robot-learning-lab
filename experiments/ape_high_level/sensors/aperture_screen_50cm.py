"""Screen sightlines through the current 50 cm shell, without camera hardware.

Uses the exported neutral 500 mm STL shells, hypothetical optical centers and
ideal pinhole lenses. It checks only five points per target; it is not a
rendered semantic or moving-pose evaluation. A hypothetical head pitch rotates
the front shell with the camera. The 22-axis v8-v10 case-envelope simulation
proxies include head pitch; current 50 cm shell screens retain the inherited
21-axis yaw-only layout. Neither establishes clearance or a buildable optic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STL = ROOT / "artifacts/sketch-ape-shell/exports/500mm/stl"
HEAD_PARTS = ("head_cranium_front", "head_cranium_back", "face_mask",
              "muzzle_shell", "L_eye_bezel", "R_eye_bezel")
FIXED_PARTS = ("neck_collar", "torso_front", "torso_back")
IMAGE = (640, 480)
VFOV = 67.0
HEAD_PIVOT_MM = np.array([0.0, 0.0, 400.0])  # provisional
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                       ("attribute", "<u2")])


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _triangles(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * STL_RECORD.itemsize:
        raise ValueError(f"Unexpected STL size: {path}")
    return np.frombuffer(raw, dtype=STL_RECORD, offset=84,
                         count=count)["vertices"].astype(np.float64)


def _first_hit(triangles: np.ndarray, origin: np.ndarray,
               target: np.ndarray) -> float | None:
    """Return first segment/triangle hit using vectorized Möller–Trumbore."""
    direction = target - origin
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    det = np.einsum("ij,ij->i", edge1, p)
    valid = np.abs(det) > 1e-8
    inverse = np.zeros_like(det)
    inverse[valid] = 1.0 / det[valid]
    tvec = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, p) * inverse
    q = np.cross(tvec, edge1)
    v = q @ direction * inverse
    fraction = np.einsum("ij,ij->i", edge2, q) * inverse
    crossed = valid & (u >= 0) & (v >= 0) & (u + v <= 1)
    crossed &= (fraction > 0.001) & (fraction < 0.999)
    return float(fraction[crossed].min()) if np.any(crossed) else None


def _target_points(x: float, y: float, z: float, half_width: float,
                   half_height: float) -> list[np.ndarray]:
    return [np.array([x, y, z]),
            *[np.array([x + sx * half_width, y, z + sz * half_height])
              for sx in (-1, 1) for sz in (-1, 1)]]


TARGETS = {
    "held_eye_level_0p35m": _target_points(0, -350, 446, 35, 35),
    "held_eye_level_0p60m": _target_points(0, -600, 446, 35, 35),
    "adult_face_1m": _target_points(0, -1000, 1500, 90, 90),
    "adult_face_1p5m": _target_points(0, -1500, 1500, 90, 90),
    "adult_face_2m": _target_points(0, -2000, 1500, 90, 90),
    "child_face_1m": _target_points(0, -1000, 1100, 80, 80),
    "adult_face_left_1m": _target_points(-500, -1000, 1500, 90, 90),
    "adult_face_right_1m": _target_points(500, -1000, 1500, 90, 90),
    "floor_3cm_40cm": _target_points(0, -400, 15, 15, 15),
    "floor_3cm_80cm": _target_points(0, -800, 15, 15, 15),
    **{f"floor_8cm_{cm}cm": _target_points(0, -10 * cm, 40, 40, 40)
       for cm in (12, 25, 40, 60, 80)},
    "floor_8cm_left_40cm": _target_points(-250, -400, 40, 40, 40),
    "floor_8cm_right_40cm": _target_points(250, -400, 40, 40, 40),
}

VIEWS = {
    "yaw_only_eye_level": {"eye_mm": [-25.7, -46, 446], "lens_elevation_deg": 0,
                           "head_pitch_deg": 0},
    "fixed_eye_up": {"eye_mm": [-25.7, -46, 446], "lens_elevation_deg": 45,
                     "head_pitch_deg": 0},
    "fixed_eye_up25": {"eye_mm": [-25.7, -46, 446],
                       "lens_elevation_deg": 25, "head_pitch_deg": 0},
    "fixed_eye_down": {"eye_mm": [25.7, -46, 446], "lens_elevation_deg": -45,
                       "head_pitch_deg": 0},
    "fixed_eye_down40": {"eye_mm": [25.7, -46, 446],
                         "lens_elevation_deg": -40, "head_pitch_deg": 0},
    "fixed_eye_down40_5mm_forward": {"eye_mm": [25.7, -51, 446],
                                     "lens_elevation_deg": -40,
                                     "head_pitch_deg": 0},
    "fixed_eye_down_5mm_forward": {"eye_mm": [25.7, -51, 446],
                                   "lens_elevation_deg": -45,
                                   "head_pitch_deg": 0},
    "fixed_chest_down": {"eye_mm": [0, -48, 340], "lens_elevation_deg": -40,
                         "head_pitch_deg": 0},
    "pitch_eye_up": {"eye_mm": [-25.7, -46, 446], "lens_elevation_deg": 0,
                     "head_pitch_deg": -45},
    "pitch_eye_down": {"eye_mm": [-25.7, -46, 446], "lens_elevation_deg": 0,
                       "head_pitch_deg": 45},
    "pitch_eye_down_yaw_left40": {"eye_mm": [-25.7, -46, 446],
                                  "lens_elevation_deg": 0, "head_pitch_deg": 45,
                                  "head_yaw_deg": -40},
    "pitch_eye_down_yaw_right40": {"eye_mm": [-25.7, -46, 446],
                                   "lens_elevation_deg": 0, "head_pitch_deg": 45,
                                   "head_yaw_deg": 40},
    "fixed_eye_down40_yaw_left40": {"eye_mm": [25.7, -46, 446],
                                    "lens_elevation_deg": -40, "head_pitch_deg": 0,
                                    "head_yaw_deg": -40},
    "fixed_eye_down40_yaw_right40": {"eye_mm": [25.7, -46, 446],
                                     "lens_elevation_deg": -40, "head_pitch_deg": 0,
                                     "head_yaw_deg": 40},
    "fixed_eye_down40_5mm_forward_yaw_right40": {
        "eye_mm": [25.7, -51, 446], "lens_elevation_deg": -40,
        "head_pitch_deg": 0, "head_yaw_deg": 40},
    "fixed_eye_down40_10mm_forward_yaw_right40": {
        "eye_mm": [25.7, -56, 446], "lens_elevation_deg": -40,
        "head_pitch_deg": 0, "head_yaw_deg": 40},
}


def _screen(view: dict, target: np.ndarray, meshes: dict[str, np.ndarray]) -> dict:
    origin_neutral = np.array(view["eye_mm"], dtype=float)
    head_rotation = _rotation_z(view.get("head_yaw_deg", 0)) @ _rotation_x(
        view["head_pitch_deg"])
    origin = HEAD_PIVOT_MM + head_rotation @ (origin_neutral - HEAD_PIVOT_MM)
    angle = math.radians(view["lens_elevation_deg"])
    forward = head_rotation @ np.array([0.0, -math.cos(angle), math.sin(angle)])
    up = head_rotation @ np.array([0.0, math.sin(angle), math.cos(angle)])
    right = head_rotation @ np.array([1.0, 0.0, 0.0])
    delta = target - origin
    distance_forward = float(delta @ forward)
    if distance_forward <= 0:
        return {"in_frame": False, "occluder": None}
    focal = IMAGE[1] / (2 * math.tan(math.radians(VFOV) / 2))
    u = IMAGE[0] / 2 + focal * float(delta @ right) / distance_forward
    v = IMAGE[1] / 2 - focal * float(delta @ up) / distance_forward
    in_frame = 0 <= u < IMAGE[0] and 0 <= v < IMAGE[1]
    if not in_frame:
        return {"in_frame": False, "px": [round(u, 1), round(v, 1)],
                "occluder": None}
    inverse = head_rotation.T
    target_head = HEAD_PIVOT_MM + inverse @ (target - HEAD_PIVOT_MM)
    hits = []
    for name, triangles in meshes.items():
        start, end = ((origin_neutral, target_head) if name in HEAD_PARTS
                      else (origin, target))
        fraction = _first_hit(triangles, start, end)
        if fraction is not None:
            hits.append((fraction, name))
    return {"in_frame": True, "px": [round(u, 1), round(v, 1)],
            "occluder": min(hits)[1] if hits else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = HEAD_PARTS + FIXED_PARTS
    meshes = {name: _triangles(STL / f"{name}.stl") for name in names}
    rows = []
    for view_name, view in VIEWS.items():
        for target_name, points in TARGETS.items():
            samples = [_screen(view, point, meshes) for point in points]
            clear = [sample["px"] for sample in samples
                     if sample["in_frame"] and sample["occluder"] is None]
            pixel_box = ([round(min(p[0] for p in clear), 1),
                          round(min(p[1] for p in clear), 1),
                          round(max(p[0] for p in clear), 1),
                          round(max(p[1] for p in clear), 1)] if clear else None)
            rows.append({"view": view_name, "target": target_name,
                         "clear_points": len(clear), "sample_points": len(samples),
                         "clear_pixel_box": pixel_box, "samples": samples})
    result = {
        "scope": ("Neutral 50 cm exported shell sightline screen, not a physical "
                  "camera placement, articulated CAD, detector, or gait visibility test. "
                  "Five sample points per target and ideal pinhole projection. "
                  "Head pitch and all optical placements are hypothetical."),
        "source": str(STL.relative_to(ROOT)),
        "part_sha256": {name: hashlib.sha256((STL / f"{name}.stl").read_bytes()).hexdigest()
                        for name in names},
        "image_size": IMAGE, "vertical_fov_deg": VFOV,
        "head_pivot_mm_assumed": HEAD_PIVOT_MM.tolist(),
        "views": VIEWS, "targets_mm": {name: [p.tolist() for p in points]
                                     for name, points in TARGETS.items()},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for view_name in VIEWS:
        print(view_name, [(r["target"], r["clear_points"]) for r in rows
                          if r["view"] == view_name])


if __name__ == "__main__":
    main()
