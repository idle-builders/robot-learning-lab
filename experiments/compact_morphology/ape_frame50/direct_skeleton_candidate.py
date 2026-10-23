"""Direct-drive packaging study for the 50 cm ape, in MuJoCo world millimetres.

Every motor case belongs to the parent link of its driven axis. Each output is
coaxial with the corresponding MJCF joint and attaches to the moving link with
a short hub. This is a packaging proposal, not a torque-qualified mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import build123d as bd


@dataclass(frozen=True)
class Drive:
    name: str
    model: str
    parent: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    pivot: tuple[float, float, float]
    axis: str
    output_sign: int

    @property
    def center_to_axis(self) -> float:
        i = "xyz".index(self.axis)
        return math.dist(self.center, self.pivot)

    @property
    def output_gap(self) -> float:
        i = "xyz".index(self.axis)
        face = self.center[i] + self.output_sign * self.size[i] / 2
        return abs(self.pivot[i] - face)


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def shaft_between(a, b, radius=4.5):
    a, b = tuple(a), tuple(b)
    i = next(i for i in range(3) if abs(a[i] - b[i]) > 1e-6)
    length = abs(a[i] - b[i])
    mid = tuple((a[k] + b[k]) / 2 for k in range(3))
    rotation = {0: (0, 90, 0), 1: (90, 0, 0), 2: (0, 0, 0)}[i]
    return bd.Pos(*mid) * bd.Cylinder(radius, length, rotation=rotation)


def head_sensor_cage():
    outer = box((1, 0, 446), (68, 76, 55))
    hollow = box((1, 0, 443), (62, 70, 55))  # open underside
    camera_opening = box((34, 15, 457), (12, 30, 20))
    camera_mount = box((38, 15, 457), (17, 38, 6))
    return (outer - hollow - camera_opening) + camera_mount


def drives() -> list[Drive]:
    result = []
    for side, s in (("left", 1), ("right", -1)):
        def add(name, model, parent, center, size, pivot, axis, output_sign):
            y_center = center[1] * s
            y_pivot = pivot[1] * s
            sign = output_sign * (s if axis == "y" else 1)
            result.append(Drive(f"{side}_{name}", model, parent.format(side=side),
                                (center[0], y_center, center[2]), size,
                                (pivot[0], y_pivot, pivot[2]), axis, sign))

        # Full supplier STEP envelopes include output features: 37.4/37.8 mm
        # depth, slightly more than the catalog case-body dimensions.
        add("hip_yaw", "ST3250", "pelvis", (-2, 46.5, 265.7),
            (24.72, 45.22, 37.4), (-2, 59, 247), "z", -1)
        add("hip_roll", "ST3250", "{side}_hip_yaw_body", (-37.5, 59, 234.5),
            (37.4, 24.72, 45.22), (-2, 59, 247), "x", 1)
        add("hip_pitch", "ST3250", "{side}_hip_roll_body", (-2, 95.5, 206.5),
            (24.72, 37.4, 45.22), (-2, 59, 219), "y", -1)
        add("knee", "ST3250", "{side}_thigh", (-3, 31.5, 151.5),
            (24.72, 37.4, 45.22), (-3, 61, 139), "y", 1)
        add("ankle_pitch", "ST3215", "{side}_shin", (-2, 33.5, 55.5),
            (24.72, 37.8, 45.22), (-2, 63, 43), "y", 1)
        add("shoulder_yaw", "ST3215", "torso", (-2, 69.5, 373.25),
            (24.72, 45.22, 37.8), (-2, 82, 347), "z", -1)
        add("shoulder_roll", "ST3215", "{side}_shoulder_yaw_body", (-38.25, 92.5, 347),
            (37.8, 45.22, 24.72), (-2, 105, 347), "x", 1)
        add("shoulder_pitch", "ST3215", "{side}_shoulder_roll_body", (-2, 151.25, 334.5),
            (24.72, 37.8, 45.22), (-2, 119, 347), "y", -1)
        add("elbow_pitch", "ST3215", "{side}_upper_arm", (0, 109.1, 285.5),
            (24.72, 37.8, 45.22), (0, 128, 273), "y", 1)
        add("gripper", "XL330", "{side}_forearm", (22, 150, 132),
            (34, 20, 26), (5, 150, 132), "x", -1)
    result.append(Drive("head_yaw", "XL330", "torso", (-1, 0, 384),
                        (20, 34, 26), (-1, 0, 397), "z", 1))
    return result


def stage_frames():
    """One structural stage per moving MJCF body; no decorative shell."""
    yield "pelvis", {"pelvis_crossbeam_frame": box((0, 0, 231), (28, 136, 16)),
                     "pelvis_spine_root_frame": box((0, 0, 248), (25, 30, 28)),
                     "left_yaw_upright_frame": box((14, 48, 252), (12, 18, 36)),
                     "right_yaw_upright_frame": box((14, -48, 252), (12, 18, 36))}
    yield "torso", {"torso_spine_frame": box((0, 0, 298), (26, 28, 118)),
                    "torso_shoulder_bar_frame": box((0, 0, 347), (28, 176, 17)),
                    "torso_neck_side_strut_frame": box((14, 0, 365), (10, 20, 32)),
                    "battery_4s850": box((-23, 0, 279), (60, 30, 30)),
                    "imu_reserve": box((5, 0, 311), (14, 14, 8))}
    yield "head_yaw_body", {"head_neck_hub_frame": box((-1, 0, 405), (22, 22, 14)),
                            "head_neck_mast_frame": box((-1, 0, 420), (14, 14, 32))}
    yield "head", {"head_open_frame": head_sensor_cage(),
                   "head_cage_spine_frame": box((-1, 0, 449), (14, 14, 51)),
                   "camera_module": box((49, 15, 457), (25, 24, 13))}
    for side, s in (("left", 1), ("right", -1)):
        def b(c, sz):
            return box((c[0], c[1] * s, c[2]), sz)
        yield f"{side}_hip_yaw_body", {
            "hip_yaw_to_roll_frame": b((-25, 59, 240), (58, 12, 14)),
            "hip_yaw_to_roll_mount_frame": b((-39, 59, 236), (18, 28, 16)),
            "hip_yaw_inboard_tab_frame": b((-8, 51, 239), (14, 24, 12)),
            "hip_yaw_inboard_bypass_frame": b((-28, 42, 239), (52, 8, 8))}
        yield f"{side}_hip_roll_body", {
            "hip_roll_to_pitch_frame": b((-2, 77, 233), (14, 52, 28)),
            "hip_roll_pitch_mount_frame": b((-2, 93, 219), (28, 24, 16)),
            "hip_roll_outside_bypass_frame": b((11, 74, 233), (14, 14, 9))}
        yield f"{side}_thigh", {
            "thigh_bone_frame": b((-2, 61, 179), (20, 18, 84)),
            "thigh_knee_mount_frame": b((-3, 47, 139), (27, 39, 15))}
        yield f"{side}_shin", {
            "shin_bone_frame": b((-2, 63, 91), (18, 16, 98)),
            "shin_knee_lug_frame": b((-3, 61, 139), (20, 20, 14)),
            "shin_ankle_mount_frame": b((-2, 46, 43), (27, 43, 14))}
        yield f"{side}_foot", {
            "foot_sole_frame": b((12, 63, 5), (120, 70, 9)),
            "foot_ankle_post_frame": b((-2, 63, 26), (19, 18, 39))}
        yield f"{side}_shoulder_yaw_body", {
            "shoulder_yaw_to_roll_frame": b((-24, 94, 347), (61, 34, 12)),
            "shoulder_yaw_upper_tab_frame": b((-6, 105, 353), (14, 13, 22)),
            "shoulder_yaw_upper_bypass_frame": b((-28, 105, 363), (52, 10, 8))}
        yield f"{side}_shoulder_roll_body", {
            "shoulder_roll_to_pitch_frame": b((-2, 128, 347), (17, 58, 16)),
            "shoulder_roll_outside_bypass_frame": b((17, 132, 347), (24, 14, 9))}
        yield f"{side}_upper_arm", {
            "upper_arm_bone_frame": b((-1, 122.5, 309), (18, 18, 80)),
            "upper_arm_elbow_mount_frame": b((0, 117, 273), (27, 32, 15))}
        yield f"{side}_forearm", {
            "forearm_bone_frame": b((4, 139, 202), (19, 18, 145)),
            "forearm_elbow_lug_frame": b((0, 136, 273), (20, 14, 14)),
            "forearm_hand_frame": b((12, 150, 129), (35, 23, 20))}
        yield f"{side}_moving_jaw", {
            "jaw_frame": b((-3, 165, 130), (14, 28, 18))}


def visual_parts():
    by_body = dict(stage_frames())
    mounted = {}
    hubs = {}
    for drive in drives():
        mounted.setdefault(drive.parent, []).append(drive)
        hubs.setdefault(child_body(drive), {})[f"{drive.name}_rotating_hub"] = rotating_hub(drive)
    for body, raw in by_body.items():
        structural = [s for name, s in raw.items() if name.endswith("_frame")]
        other = {name: s for name, s in raw.items() if not name.endswith("_frame")}
        frame = structural[0]
        for shape in structural[1:]:
            frame = frame + shape
        for drive in mounted.get(body, []):
            frame = frame + motor_cradle(drive)
            # The case needs a real empty pocket, not a beam passing through it.
            frame = frame - box(drive.center, tuple(v + 1 for v in drive.size))
        parts = {f"{body}_integrated_frame": frame, **other, **hubs.get(body, {})}
        for drive in mounted.get(body, []):
            parts[f"{drive.name}_{drive.model}_case"] = box(drive.center, drive.size)
        yield body, body, parts


def motor_cradle(drive: Drive):
    """A simple three-sided pocket fused into the parent frame."""
    outside = box(drive.center, tuple(v + 6 for v in drive.size))
    inner_center = list(drive.center)
    inner_center["xyz".index(drive.axis)] += 3 * drive.output_sign
    inside = box(inner_center, tuple(v + 1 for v in drive.size))
    return outside - inside


def child_body(drive: Drive) -> str:
    if drive.name == "head_yaw":
        return "head_yaw_body"
    side, joint = drive.name.split("_", 1)
    suffix = {
        "hip_yaw": "hip_yaw_body", "hip_roll": "hip_roll_body",
        "hip_pitch": "thigh", "knee": "shin", "ankle_pitch": "foot",
        "shoulder_yaw": "shoulder_yaw_body", "shoulder_roll": "shoulder_roll_body",
        "shoulder_pitch": "upper_arm", "elbow_pitch": "forearm",
        "gripper": "moving_jaw",
    }[joint]
    return f"{side}_{suffix}"


def rotating_hub(drive: Drive):
    i = "xyz".index(drive.axis)
    face = list(drive.center)
    face[i] += drive.output_sign * drive.size[i] / 2
    end = list(drive.pivot)
    end[i] += drive.output_sign * 6
    return shaft_between(face, end, radius=5.5)


def motor_report():
    return [{"joint": d.name, "motor": d.model, "case_parent": d.parent,
             "case_center_mm": d.center, "pivot_mm": d.pivot,
             "center_to_axis_mm": round(d.center_to_axis, 2),
             "direct_output_gap_mm": round(d.output_gap, 2)} for d in drives()]


def neutral_case_overlaps():
    items = drives()
    overlaps = []
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            penetration = [min(a.center[k] + a.size[k] / 2,
                               b.center[k] + b.size[k] / 2) -
                           max(a.center[k] - a.size[k] / 2,
                               b.center[k] - b.size[k] / 2)
                           for k in range(3)]
            if all(v > 0 for v in penetration):
                overlaps.append({"a": a.name, "b": b.name,
                                 "xyz_overlap_mm": [round(v, 2) for v in penetration]})
    return overlaps
