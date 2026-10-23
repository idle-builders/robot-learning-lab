"""First industrial-design/structure pass for the 35 cm ape.

Run from the repository root with `.venv/bin/python .../ape_design/build.py`.
CAD coordinates are millimetres in each MJCF body's frame. The neutral STEP,
per-part STEP/STL and GLB are generated from the same solids. This is an
assembled design candidate, not a strength- or fit-certified prototype.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from ape_cad.build import battery_pose, poses  # noqa: E402

MODEL_PATH = HERE.parent / "ape_cad/ape_spaced_v10_shoulders_forward.xml"
OUT = ROOT / "artifacts/ape-design-20260930"
PRINT_DENSITY_G_CM3 = 1.23  # PETG-like nominal material, not measured X1 parts.

COLORS = {
    "shell": [226, 233, 225, 255],
    "back": [98, 122, 127, 255],
    "frame": [61, 75, 82, 255],
    "accent": [230, 123, 85, 255],
    "face": [25, 36, 44, 255],
    "eye": [93, 231, 210, 255],
    "sole": [48, 65, 71, 255],
    "motor": [58, 106, 160, 255],
    "battery": [88, 169, 105, 255],
}


@dataclass
class Part:
    name: str
    body: str
    kind: str
    shape: bd.Shape
    role: str

    @property
    def volume_mm3(self) -> float:
        return float(self.shape.volume)

    @property
    def mass_kg(self) -> float:
        return self.volume_mm3 * PRINT_DENSITY_G_CM3 * 1e-6

    @property
    def com_m(self) -> np.ndarray:
        return np.array(tuple(self.shape.center(bd.CenterOf.MASS))) / 1000

    @property
    def inertia_kg_m2(self) -> np.ndarray:
        # OpenCascade returns the volume moment in mm^5 for unit density.
        return np.array(self.shape.matrix_of_inertia) * PRINT_DENSITY_G_CM3 * 1e-12


PARTS: list[Part] = []


def add(name: str, body: str, kind: str, shape: bd.Shape, role: str) -> None:
    if shape.volume <= 0:
        raise ValueError(f"Empty printed solid: {name}")
    PARTS.append(Part(name, body, kind, shape, role))


def lofted(spec: list[tuple[float, float, float, float]]) -> bd.Shape:
    return bd.loft([
        bd.Plane(origin=(0, 0, z)) * bd.RectangleRounded(x, y, radius)
        for z, x, y, radius in spec
    ])


def split_shell(spec: list[tuple[float, float, float, float]], wall: float = 1.6):
    outer = lofted(spec)
    inner = lofted([
        (z + (-2 if i == 0 else 2 if i == len(spec) - 1 else 0),
         x - 2 * wall, y - 2 * wall, max(2, radius - wall))
        for i, (z, x, y, radius) in enumerate(spec)
    ])
    shell = outer - inner
    front = shell & (bd.Pos(100, 0, 0) * bd.Box(200, 300, 500))
    back = shell & (bd.Pos(-100, 0, 0) * bd.Box(200, 300, 500))
    return front, back


def soft_box(x: float, y: float, z: float, r: float) -> bd.Shape:
    box = bd.Box(x, y, z)
    return bd.fillet(box.edges(), min(r, x / 2 - .1, y / 2 - .1, z / 2 - .1))


def open_cover(x: float, y: float, z: float, radius: float,
               wall: float = 2.6) -> bd.Shape:
    """Rounded cover with a real underside opening, not a heavy solid block."""
    outside = soft_box(x, y, z, radius)
    cavity = bd.Pos(0, 0, -wall / 2) * bd.Box(
        x - 2 * wall, y - 2 * wall, z - wall + 1)
    return outside - cavity


def angled_bridge(dx: float, dy: float, height: float = 16) -> bd.Shape:
    length = math.hypot(dx, dy) + 18
    angle = math.degrees(math.atan2(dy, dx))
    return bd.Pos(dx / 2, dy / 2, 0) * bd.Rot(0, 0, angle) * open_cover(
        length, 19, height, 5, 2.7)


def outer_spine(length: float, z_center: float,
                side: int, outboard_mm: float) -> bd.Shape:
    """Case-clear load rib on the outside of a yaw joint's swept path."""
    core = bd.Pos(-10, side * outboard_mm, z_center) * soft_box(17, 10, length, 4)
    rib = bd.Pos(-10, side * (outboard_mm + 4), 0) * lofted([
        (z_center - length / 2 + 6, 12, 4, 1.5),
        (z_center - length / 2 + 20, 16, 6, 2),
        (z_center + length / 2 - 20, 16, 6, 2),
        (z_center + length / 2 - 6, 12, 4, 1.5),
    ])
    return core + rib


def side_rails(length: float, width: float, *, width_x: float = 4.0) -> bd.Shape:
    """Light twin webs; motor pockets and output hubs are separate interfaces."""
    half = width / 2
    rail = None
    for y in (-half, half):
        piece = bd.Pos(0, y, -length / 2) * bd.Box(width_x, 3.5, length - 32)
        rail = piece if rail is None else rail + piece
    # End ties fell inside the neighboring motor cases and were completely
    # removed by the real keepouts, leaving two disconnected strips. Tie the
    # webs in the clear center span instead.
    for z in (-length / 2 - 7, -length / 2 + 7):
        rail += bd.Pos(0, 0, z) * bd.Box(11, width + 3.5, 3.5)
    return rail


def make_body() -> None:
    torso = [(-10, 78, 64, 20), (20, 82, 72, 23),
             (61, 96, 112, 27), (95, 58, 70, 19)]
    front, back = split_shell(torso, 1.8)
    # The narrowed waist makes a detached lower rear lip after motor relief;
    # open the service cover at its bottom instead of exporting that lip.
    back = back & (bd.Pos(0, 0, 60) * bd.Box(200, 200, 120))
    add("torso_face_shell", "pelvis", "shell", front,
        "Removable front carapace, protects battery and hip cases")
    add("torso_back_shell", "pelvis", "back", back,
        "Removable rear service cover")
    # The rear spine and the battery cradle are distinct printed load paths.
    spine = bd.Pos(-43, 0, 39) * soft_box(8, 26, 106, 3)
    spine += bd.Pos(-29, 0, -4) * bd.Box(32, 86, 5)
    spine += bd.Pos(-28, 0, 80) * bd.Box(33, 66, 5)
    tray = bd.Pos(0, 0, 22) * bd.Box(68, 40, 3)
    for y in (-20, 20):
        tray += bd.Pos(0, y, 33) * bd.Box(68, 3, 21)
    add("battery_cradle", "pelvis", "frame", tray,
        "60×30×30.5 mm pack pocket; connector and retention unresolved")
    for side, sign in (("left", 1), ("right", -1)):
        bridge = bd.Pos(-30, sign * 30, 85) * angled_bridge(107, sign * 50, 11)
        spine += bridge
        cape_sections = ((-20, 27, 42, 31),
                         (28, 55, 30, 25),
                         (61, 74, 18, 17))
        shoulder_cape = bd.loft([
            bd.Plane(origin=(x, sign * y, 85), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
            * bd.RectangleRounded(width, height, min(7, width / 3, height / 3))
            for x, y, width, height in cape_sections
        ])
        cape_void = bd.loft([
            bd.Plane(origin=(x + (-2 if i == 0 else 2 if i == 2 else 0),
                             sign * y, 85), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
            * bd.RectangleRounded(width - 4, height - 4,
                                   min(5, (width - 4) / 3, (height - 4) / 3))
            for i, (x, y, width, height) in enumerate(cape_sections)
        ])
        shoulder_cape -= cape_void
        add(f"{side}_shoulder_cape", "pelvis", "shell", shoulder_cape,
            "Tapered torso-side fairing; ends before moving shoulder seam")
        # A narrow outer spine remains continuous alongside both cases. A
        # wraparound cosmetic motor cover is intentionally separate from the
        # load path and is deferred until a source horn drawing is available.
        add(f"{side}_shoulder_yaw_link", f"{side}_shoulder_yaw_body",
            "frame", outer_spine(76, -25, sign, 35),
            "Outer yaw-to-roll spine; shaft flanges require source horn CAD")
        add(f"{side}_shoulder_roll_bridge", f"{side}_shoulder_roll_body",
            "frame", angled_bridge(25, sign * 50),
            "Connects shoulder roll output to the pitch-servo cradle")
        add(f"{side}_hip_yaw_link", f"{side}_hip_yaw",
            "frame", outer_spine(88, -30, sign, 50),
            "Outer hip yaw-to-roll spine; shaft flanges require source horn CAD")
        add(f"{side}_hip_roll_bridge", f"{side}_hip_roll",
            "frame", angled_bridge(25, sign * 43),
            "Connects hip roll output to hip pitch shaft")

    add("torso_chassis", "pelvis", "frame", spine,
        "Single print joins hip deck, shoulder yoke and neck support")

    # Head is smaller than the previous 104 mm sphere and has a recognisable
    # ape face: dark visor, soft muzzle, wide-set eyes and small ears.
    helmet = [(-35, 46, 58, 15), (-21, 78, 86, 24),
              (16, 84, 94, 27), (42, 54, 64, 19)]
    front, back = split_shell(helmet, 1.6)
    add("head_front_shell", "head", "shell", front, "Front helmet half")
    add("head_back_shell", "head", "back", back, "Rear helmet and head-pitch access")
    visor = bd.Pos(41, 0, 5) * soft_box(7, 72, 52, 15)
    add("face_visor", "head", "face", visor, "Replaceable dark face insert")
    muzzle = bd.Pos(46, 0, -23) * soft_box(17, 59, 20, 8)
    add("muzzle", "head", "shell", muzzle, "Rounded lower face integrated with mask")
    nose = bd.Pos(53, 0, -17) * soft_box(4, 12, 5, 2)
    add("nose", "head", "face", nose, "Small face detail")
    brow = bd.Pos(43, 0, 28) * soft_box(11, 79, 10, 5)
    add("brow_rim", "head", "shell", brow,
        "Soft upper rim around face visor; eye apertures stay open")
    for side, sign in (("left", 1), ("right", -1)):
        eye = bd.Pos(46, sign * 19, 10) * bd.Sphere(7.2)
        add(f"{side}_eye_diffuser", "head", "eye", eye,
            "Passive eye insert; lighting is not designed")
        ear = bd.Pos(-3, sign * 47, 0) * soft_box(22, 12, 23, 8)
        add(f"{side}_ear", "head", "back", ear, "Rounded ear inset")
    collar = bd.Pos(0, 0, 19) * bd.Cylinder(23, 9)
    add("neck_collar", "head_yaw_body", "frame", collar,
        "Visible guard between yaw and pitch axes")


def make_limbs() -> None:
    for side, sign in (("left", 1), ("right", -1)):
        # Covers are separate front/back prints. The hidden paired rails carry
        # the pitched segment loads; the covers are not treated as the frame.
        for segment, body, length, sections, rail_width in (
            ("thigh", f"{side}_thigh", 73,
             [(-15, 43, 46, 13), (-38, 52, 54, 17), (-66, 36, 40, 12)], 23),
            ("shin", f"{side}_shin", 76,
             [(-15, 39, 41, 12), (-40, 46, 48, 16), (-67, 34, 37, 11)], 22),
            ("upper_arm", f"{side}_arm", 95,
             [(-40, 40, 43, 13), (-56, 51, 53, 17), (-83, 36, 40, 12)], 22),
            ("forearm", f"{side}_forearm", 75,
             [(-16, 37, 40, 12), (-39, 47, 48, 16), (-65, 35, 38, 11)], 21),
        ):
            front, back = split_shell(sections, 1.45)
            if segment == "thigh":
                # The back cover clears the folding knee and hip cases; the
                # full front profile remains for the ape silhouette.
                back = back & (bd.Pos(0, 0, -37) * bd.Box(200, 200, 24))
            elif segment == "shin":
                back = back & (bd.Pos(0, 0, -58) * bd.Box(200, 200, 40))
            add(f"{side}_{segment}_front", body, "shell", front,
                f"{segment.replace('_', ' ').title()} front protective half")
            add(f"{side}_{segment}_back", body, "back", back,
                f"{segment.replace('_', ' ').title()} rear service half")
            add(f"{side}_{segment}_rails", body, "frame",
                side_rails(length, rail_width),
                "Paired printed webs tied to pitch-axis output flange")
        shoulder = bd.Pos(25, sign * 82, -13) * soft_box(50, 8, 29, 4)
        add(f"{side}_shoulder_cap", f"{side}_shoulder_roll_body", "back",
            shoulder, "Outboard shoulder fairing; independent of load path")
        hip = bd.Pos(13, sign * 78, -48) * soft_box(43, 8, 28, 4)
        add(f"{side}_hip_link_cover", f"{side}_hip_roll", "frame", hip,
            "Outboard hip fairing; independent of load path")
        foot = bd.Pos(14, 0, -20) * (
            bd.fillet(bd.Box(95, 65, 18).edges().filter_by(bd.Axis.Z), 13)
            - bd.Pos(0, 0, 3) * bd.Box(89, 59, 15))
        add(f"{side}_foot_chassis", f"{side}_foot", "shell", foot,
            "Rigid rounded foot with interchangeable outsole land")
        sole = bd.Pos(14, 0, -31) * bd.fillet(
            bd.Box(91, 61, 4).edges().filter_by(bd.Axis.Z), 12)
        add(f"{side}_sole_pad", f"{side}_foot", "sole", sole,
            "Removable friction pad; material and retention unverified")
        paw = bd.Pos(0, 0, -82) * open_cover(35, 33, 21, 6, 2.1)
        add(f"{side}_palm_shell", f"{side}_forearm", "back", paw,
            "Paw housing around opposed gripper jaws")
        finger = bd.Pos(0, 0, -23) * soft_box(17, 9, 28, 4)
        add(f"{side}_moving_finger", f"{side}_finger", "accent", finger,
            "Separate servo-driven opposed jaw")


def body_tf(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[bid].reshape(3, 3)
    transform[:3, 3] = data.xpos[bid]
    return transform


def cad_place(shape: bd.Shape, rotation: np.ndarray,
              translation_mm: np.ndarray) -> bd.Shape:
    """Place an OCC solid with the same world transform as a MuJoCo frame.

    build123d's Rot(x,y,z) composes X,Y,Z in a different order from SciPy's
    extrinsic xyz Euler decomposition. Compose Z*Y*X explicitly instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x, y, z = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return (bd.Pos(*translation_mm) * bd.Rot(0, 0, z) *
            bd.Rot(0, y, 0) * bd.Rot(x, 0, 0) * shape)


def cut_component_keepouts(model: mujoco.MjModel, data: mujoco.MjData,
                           clearance_mm: float = 0.8) -> list[dict]:
    """Make actual cavities around component envelopes in the neutral assembly.

    This is deliberately a geometric operation on exported parts, not merely
    a collision-mask exclusion. Coupled joint sweeps still have to clear.
    """
    envelopes = poses(model, data) + [battery_pose(model, data)]
    edits = []
    for part in PARTS:
        inv_body = np.linalg.inv(body_tf(model, data, part.body))
        for component in envelopes:
            center = inv_body[:3, :3] @ component["center"] + inv_body[:3, 3]
            rotation = inv_body[:3, :3] @ component["rot"]
            cutter = cad_place(
                bd.Box(*(component["half"] * 2000 + 2 * clearance_mm)),
                rotation, center * 1000)
            part_bounds = part.shape.bounding_box(optimal=False)
            cutter_bounds = cutter.bounding_box(optimal=False)
            if not all(getattr(part_bounds.min, axis) <
                       getattr(cutter_bounds.max, axis) and
                       getattr(cutter_bounds.min, axis) <
                       getattr(part_bounds.max, axis)
                       for axis in "XYZ"):
                continue
            old_volume = part.shape.volume
            trimmed = part.shape - cutter
            if trimmed is None or trimmed.volume < 5:
                raise ValueError(f"Component {component['name']} consumes {part.name}")
            if old_volume - trimmed.volume > 0.1:
                part.shape = trimmed
                edits.append({"part": part.name, "component": component["name"],
                              "removed_mm3": round(old_volume - trimmed.volume, 2),
                              "clearance_mm": clearance_mm})
    return edits


def fit_printed_parts(model: mujoco.MjModel, data: mujoco.MjData,
                      clearance_mm: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Cut actual neutral-pose mortises where separately printed solids overlap.

    Cosmetic inserts win over their receiving shell; load-bearing frames win
    over covers. This is a first assembly fit, followed by a pose sweep. A
    failed cutter is retained as an explicit unresolved interference.
    """
    def rank(part: Part) -> int:
        if part.name == "torso_face_shell":
            return 8
        if "eye_diffuser" in part.name or part.name == "nose":
            return 9
        if part.name in ("muzzle", "left_ear", "right_ear"):
            return 8
        if part.name == "face_visor":
            return 7
        return {"frame": 6, "sole": 5, "accent": 4,
                "shell": 2, "back": 2, "face": 4, "eye": 8}[part.kind]

    def relocated(shape: bd.Shape, source_body: str, target_body: str) -> bd.Shape:
        tf = np.linalg.inv(body_tf(model, data, target_body)) @ body_tf(model, data, source_body)
        return cad_place(shape, tf[:3, :3], tf[:3, 3] * 1000)

    def bounds_overlap(keeper: Part, victim: Part) -> bool:
        # Fast non-optimal OCC bounds first: transformed detailed BReps make
        # optimal bounds expensive after the first few fit subtractions.
        a = keeper.shape.bounding_box(optimal=False)
        b = victim.shape.bounding_box(optimal=False)
        matrix = np.linalg.inv(body_tf(model, data, victim.body)) @ body_tf(model, data, keeper.body)
        vertices = np.array([[x, y, z, 1] for x in (a.min.X, a.max.X)
                             for y in (a.min.Y, a.max.Y)
                             for z in (a.min.Z, a.max.Z)])
        shifted = (matrix @ np.diag([.001, .001, .001, 1]) @ vertices.T).T[:, :3] * 1000
        lower, upper = shifted.min(axis=0), shifted.max(axis=0)
        return all(lower[i] < getattr(b.max, axis) - .01 and
                   getattr(b.min, axis) < upper[i] - .01
                   for i, axis in enumerate("XYZ"))

    ordered = sorted(PARTS, key=lambda p: (-rank(p), p.name))
    edits, unresolved = [], []
    for index, keeper in enumerate(ordered):
        expanded = None
        for victim in ordered[index + 1:]:
            if not bounds_overlap(keeper, victim):
                continue
            source = relocated(keeper.shape, keeper.body, victim.body)
            intersection = source.intersect(victim.shape)
            volume = sum(s.volume for s in intersection) if intersection is not None else 0
            if volume <= 5:
                continue
            try:
                if expanded is None:
                    bounds = keeper.shape.bounding_box(optimal=False)
                    size = bounds.size
                    expanded = keeper.shape.scale(tuple(
                        1 + 2 * clearance_mm / max(dimension, 0.1)
                        for dimension in (size.X, size.Y, size.Z)
                    ), about=bounds.center())
                cutter = relocated(expanded, keeper.body, victim.body)
                trimmed = victim.shape - cutter
                if trimmed is None or trimmed.volume < max(5, victim.shape.volume * .20):
                    raise ValueError("cut consumes too much of the receiving part")
                # Scaling a complex keeper about its overall bounding box is
                # only an approximate relief. It can move a small internal
                # face away from the original intersection. Finish against
                # the exact, unscaled keeper whenever that happens.
                residual = trimmed.intersect(source)
                if residual is not None and sum(s.volume for s in residual) > 5:
                    trimmed = trimmed - source
                removed = victim.shape.volume - trimmed.volume
                victim.shape = trimmed
                edits.append({"keeper": keeper.name, "receiving_part": victim.name,
                              "removed_mm3": round(removed, 2),
                              "nominal_axis_clearance_mm": clearance_mm,
                              "clearance_method": "nonuniform keeper scaling before subtract"})
                if len(edits) % 10 == 0:
                    print(f"neutral print fit: {len(edits)} completed", flush=True)
            except Exception as exc:
                unresolved.append({"a": keeper.name, "b": victim.name,
                                   "overlap_mm3": round(volume, 2),
                                   "reason": str(exc)[:160]})
    return edits, unresolved


def prune_boolean_islands() -> list[dict]:
    """Discard tiny loose chips made by visor/spine fit cuts, never load paths."""
    discarded = []
    for part in PARTS:
        solids = sorted(part.shape.solids(), key=lambda solid: solid.volume, reverse=True)
        if len(solids) < 2:
            continue
        volume = sum(solid.volume for solid in solids)
        loose = sum(solid.volume for solid in solids[1:])
        if loose > volume * .02 or max(solid.volume for solid in solids[1:]) > 500:
            continue
        part.shape = solids[0]
        discarded.append({"part": part.name, "discarded_mm3": round(loose, 2),
                          "discarded_solid_count": len(solids) - 1})
    return discarded


def seat_torso_shells() -> list[dict]:
    """Remove the final tiny chassis overlap from the two service shells."""
    chassis = next(part for part in PARTS if part.name == "torso_chassis").shape
    edits = []
    for name in ("torso_face_shell", "torso_back_shell"):
        shell = next(part for part in PARTS if part.name == name)
        old_volume = shell.shape.volume
        shell.shape = shell.shape - chassis
        edits.append({"shell": name,
                      "removed_mm3": round(old_volume - shell.shape.volume, 2),
                      "note": "exact CAD chassis subtraction; print tolerance still unresolved"})
    return edits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def print_mesh(shape: bd.Shape) -> trimesh.Trimesh:
    vertices, triangles = shape.tessellate(0.35, 0.15)
    mesh = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]) / 1000,
                           faces=np.array(triangles), process=True)
    return mesh


def component_ledger(model: mujoco.MjModel, data: mujoco.MjData,
                     envelopes: list[dict]) -> dict:
    """CAD-volume mass properties plus sourced motor/battery point locations.

    Motor and battery inertias use their published case envelope as a uniform
    box until a matching official inertia/STEP becomes available. Neither
    wiring nor compute is silently allocated to the print.
    """
    components: dict[str, list[dict]] = {}

    def record(body: str, name: str, kind: str, mass: float,
               center: np.ndarray, inertia: np.ndarray, provenance: str) -> None:
        components.setdefault(body, []).append({
            "name": name, "kind": kind, "mass_kg": mass,
            "com_local_m": center, "inertia_com_local_kg_m2": inertia,
            "provenance": provenance,
        })

    for part in PARTS:
        record(part.body, part.name, "printed_" + part.kind, part.mass_kg,
               part.com_m, part.inertia_kg_m2,
               "CAD solid volume × assumed 1.23 g/cm³ PETG-like density; solid-equivalent print")
    for env in envelopes:
        body = env["owner"]
        tf = body_tf(model, data, body)
        rotation = tf[:3, :3].T @ env["rot"]
        center = tf[:3, :3].T @ (env["center"] - tf[:3, 3])
        size = env["half"] * 2
        mass = env["mass"]
        inertia_box = np.diag(mass * (np.array([
            size[1] ** 2 + size[2] ** 2,
            size[0] ** 2 + size[2] ** 2,
            size[0] ** 2 + size[1] ** 2,
        ]) / 12))
        record(body, env["name"], "battery" if env["family"] == "battery" else "motor",
               mass, center, rotation @ inertia_box @ rotation.T,
               "published component mass; uniform-box inertia approximation at sourced/inferred case pose")

    def parallel_axis(mass: float, displacement: np.ndarray) -> np.ndarray:
        return mass * (np.dot(displacement, displacement) * np.eye(3)
                       - np.outer(displacement, displacement))

    result = {}
    total_mass = 0.0
    whole_first_moment = np.zeros(3)
    for body, items in components.items():
        mass = sum(item["mass_kg"] for item in items)
        center = sum((item["mass_kg"] * item["com_local_m"] for item in items),
                     np.zeros(3)) / mass
        inertia = sum((item["inertia_com_local_kg_m2"] +
                       parallel_axis(item["mass_kg"], item["com_local_m"] - center)
                       for item in items), np.zeros((3, 3)))
        tf = body_tf(model, data, body)
        world_center = tf[:3, :3] @ center + tf[:3, 3]
        total_mass += mass
        whole_first_moment += mass * world_center
        result[body] = {
            "mass_kg": round(mass, 6),
            "com_local_m": np.round(center, 7).tolist(),
            "inertia_com_local_kg_m2": np.round(inertia, 10).tolist(),
            "components": [{**{k: v for k, v in item.items()
                                if k not in ("com_local_m", "inertia_com_local_kg_m2")},
                            "com_local_m": np.round(item["com_local_m"], 7).tolist(),
                            "inertia_com_local_kg_m2": np.round(item["inertia_com_local_kg_m2"], 10).tolist()}
                           for item in items],
        }
    return {"total_kg": round(total_mass, 6),
            "whole_com_neutral_m": np.round(whole_first_moment / total_mass, 7).tolist(),
            "links": result,
            "not_allocated": "compute, wiring, fasteners, adhesive, final pad and infill variation"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    make_body()
    make_limbs()
    neutral_keepouts = cut_component_keepouts(model, data)
    neutral_print_fits, unresolved_print_fits = fit_printed_parts(model, data)
    # The first pass can leave small residuals where a scaled, curved keeper
    # only partly encloses its original contact patch. Settle those against
    # the final shapes, then strip only the documented sub-2% loose chips.
    extra_fits, extra_unresolved = fit_printed_parts(model, data)
    neutral_print_fits += extra_fits
    unresolved_print_fits += extra_unresolved
    shell_seating = seat_torso_shells()
    discarded_islands = prune_boolean_islands()
    scene = trimesh.Scene()
    assembled: list[bd.Shape] = []
    records = []
    for part in PARTS:
        stem = OUT / "printed-parts" / part.name
        stem.parent.mkdir(exist_ok=True)
        bd.export_step(part.shape, stem.with_suffix(".step"))
        bd.export_stl(part.shape, stem.with_suffix(".stl"), tolerance=0.35)
        # OCC occasionally emits duplicate triangles at fillet seams. Those
        # make a valid single BRep solid appear non-manifold to a slicer.
        stl = trimesh.load(stem.with_suffix(".stl"), process=True)
        stl.update_faces(stl.nondegenerate_faces())
        stl.update_faces(stl.unique_faces())
        stl.remove_unreferenced_vertices()
        stl.merge_vertices()
        stl.fill_holes()
        trimesh.repair.fix_normals(stl)
        stl.export(stem.with_suffix(".stl"))
        transform = body_tf(model, data, part.body)
        assembled.append(cad_place(part.shape, transform[:3, :3],
                                   transform[:3, 3] * 1000))
        mesh = print_mesh(part.shape)
        mesh.visual.face_colors = (COLORS["back"] if part.name.endswith("_yaw_link")
                                   else COLORS[part.kind])
        scene.add_geometry(mesh, node_name=part.name, geom_name=part.name,
                           transform=transform)
        bounds = part.shape.bounding_box()
        records.append({
            "name": part.name, "body": part.body, "kind": part.kind,
            "role": part.role, "volume_mm3": round(part.volume_mm3, 2),
            "mass_kg_assumed": round(part.mass_kg, 6),
            "print_bounds_mm": [round(v, 2) for v in
                (bounds.max.X - bounds.min.X, bounds.max.Y - bounds.min.Y,
                 bounds.max.Z - bounds.min.Z)],
            "mesh_watertight": bool(mesh.is_watertight),
            "stl_watertight": bool(stl.is_watertight),
            "stl_volume_mm3": round(float(stl.volume), 2),
            "solid_count": len(part.shape.solids()),
            "cad_solid_valid": bool(part.shape.is_valid),
            "step": f"printed-parts/{part.name}.step",
            "stl": f"printed-parts/{part.name}.stl",
        })
    bd.export_step(bd.Compound(children=assembled), OUT / "ape-assembled.step")

    # The exact current motor-case and battery envelopes remain visible as an
    # optional overlay in the viewer. They are not counted as printed material.
    envelopes = poses(model, data) + [battery_pose(model, data)]
    motors = []
    reference_component_shapes = []
    for p in envelopes:
        motors.append({
            "name": p["name"], "body": p["owner"],
            "center_m": p["center"].tolist(),
            "rotation": p["rot"].tolist(),
            "size_m": (p["half"] * 2).tolist(),
            "source_part": p["part"], "mass_kg": p["mass"],
        })
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = p["rot"], p["center"]
        mesh = trimesh.creation.box(extents=p["half"] * 2)
        mesh.visual.face_colors = ([88, 122, 104, 255] if p["family"] == "battery"
                                   else [76, 89, 99, 255])
        scene.add_geometry(mesh, node_name=f"component_{p['name']}",
                           geom_name=f"component_{p['name']}", transform=transform)
        reference_component_shapes.append(cad_place(
            bd.Box(*(p["half"] * 2000)), p["rot"], p["center"] * 1000))
    bd.export_step(bd.Compound(children=reference_component_shapes),
                   OUT / "reference-component-envelopes.step")
    scene.export(OUT / "ape-assembled.glb")
    joints = []
    for jid in range(1, model.njnt):
        joints.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
            "body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                      int(model.jnt_bodyid[jid])),
            "axis_local": model.jnt_axis[jid].tolist(),
            "range_rad": model.jnt_range[jid].tolist(),
        })
    report = {
        "status": "design candidate; print fit, load capacity and shell sweeps not certified",
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "model_sha256": sha256(MODEL_PATH),
        "source_sha256": sha256(Path(__file__)),
        "printed_material_density_g_cm3_assumed": PRINT_DENSITY_G_CM3,
        "printed_mass_kg_assumed": round(sum(p.mass_kg for p in PARTS), 4),
        "maximum_print_extent_mm": max(max(r["print_bounds_mm"]) for r in records),
        "parts": records,
        "joints": joints,
        "motor_and_battery_envelopes": motors,
        "mass_properties": component_ledger(model, data, envelopes),
        "neutral_component_keepouts": neutral_keepouts,
        "neutral_print_fits": neutral_print_fits,
        "unresolved_neutral_print_fits": unresolved_print_fits,
        "discarded_boolean_islands": discarded_islands,
        "torso_shell_seating": shell_seating,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    shutil.copyfile(HERE / "viewer.html", OUT / "index.html")
    print(json.dumps({k: report[k] for k in
                      ("status", "printed_mass_kg_assumed", "maximum_print_extent_mm")}, indent=2))
    print(f"{len(records)} printed CAD parts → {OUT}")


if __name__ == "__main__":
    main()
