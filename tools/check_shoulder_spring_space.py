"""Check a hypothetical shoulder tension-spring reservation against retained CAD."""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from build123d import (
    BoundBox,
    Compound,
    Cylinder,
    Pos,
    Sphere,
    export_step,
    export_stl,
    import_step,
)
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from lab.config import load_config
from lab.leg_mechanism import Part, bounds_overlap, overlap, transform_shape
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from lab.recording import _source_inputs
from tools.compose_forearm_enclosure import verify_report
from tools.compose_forearm_enclosure import kinematic_contract

THRESHOLD_MM3 = 0.02
POSES = (
    "neutral_roll_-1.3",
    "neutral_roll_0",
    "neutral_roll_0.56",
    "neutral_roll_1.3",
    "upper_accounted_9s",
)
X_VARIANTS = (30.0, 40.0, 50.0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


class ExactQueries:
    def __init__(self):
        self._bounds = {}

    def bounds(self, s):
        k = id(s)
        if k not in self._bounds:
            b = Bnd_Box()
            BRepBndLib.Add_s(s.wrapped, b, False)
            if b.IsVoid():
                raise ValueError("empty shape bound")
            self._bounds[k] = (s, BoundBox(b))
        return self._bounds[k][1]

    def collisions(self, s, parts):
        out = []
        sb = self.bounds(s)
        for p in parts:
            if bounds_overlap(sb, self.bounds(p.shape)):
                v = overlap(s, p.shape)
                if v > THRESHOLD_MM3:
                    out.append({"part": p.name, "overlap_mm3": v})
        return out


def endpoint_world(frame, xyz_mm):
    v = np.r_[np.asarray(xyz_mm, float) / 1000, 1.0]
    return (np.asarray(frame) @ v)[:3] * 1000


def spring_shape(a, b, radius=4.0):
    a = np.asarray(a)
    b = np.asarray(b)
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise ValueError("coincident spring anchors")
    z = vec / length
    x = np.cross([0, 0, 1], z)
    if np.linalg.norm(x) < 1e-8:
        x = np.cross([0, 1, 0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    matrix = np.column_stack([x, y, z])
    mid = (a + b) / 2
    frame = np.eye(4)
    frame[:3, :3] = matrix
    frame[:3, 3] = mid / 1000
    rod = transform_shape(Cylinder(radius, length), frame)
    return Compound([rod, Pos(*a) * Sphere(radius), Pos(*b) * Sphere(radius)])


def source_check(assembly):
    report = verify_report(assembly)
    for n in (
        "report.json",
        "resolved-config.json",
        "interfaces.json",
        "assembly.step",
    ):
        if not (assembly / n).exists():
            raise ValueError(f"missing retained output {assembly / n}")
    return report


def pose_from_trajectory(robot, name, commanded=False):
    path = Path(f"artifacts/{name}-forearm-accounted-upper-task-v1/trajectory.jsonl")
    for line in path.read_text().splitlines():
        row = json.loads(line)
        js = row["joint_state"]
        if abs(js["time_s"] - 9.0) < 1e-9:
            key = "action" if commanded else "joint_state.position_rad"
            values = row["action"] if commanded else js["position_rad"]
            return dict(zip(js["names"], values, strict=True)), {
                "source": str(path),
                "time_s": js["time_s"],
                "phase": row["phase"],
                "joint_state_field": key,
            }
    raise ValueError(f"no exact 9 s row in {path}")


def run(output, commanded=False):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    report = {
        "status": "pass",
        "scope": "Hypothetical straight tension-spring reservation versus retained complete composed CAD; geometry screen only, no mounted/real-spring qualification",
        "source_bindings": {},
        "robots": {},
    }
    for name in ("pip", "scout"):
        assembly = Path(f"artifacts/{name}-integral-head-composed-v2")
        source = source_check(assembly)
        variant = Path(f"variants/{name}-distal-grip.yaml")
        robot = make_robot(load_config(variant))
        assembly_interface = json.loads((assembly / "interfaces.json").read_text())
        if json.loads(json.dumps(kinematic_contract(robot.interface()))) != json.loads(
            json.dumps(kinematic_contract(assembly_interface))
        ):
            raise ValueError(f"assembly interface differs from current variant: {name}")
        retained_config = json.loads((assembly / "resolved-config.json").read_text())
        if json.loads(json.dumps(robot.config.model_dump(mode="json"))) != json.loads(
            json.dumps(retained_config)
        ):
            raise ValueError(f"assembly config differs from current variant: {name}")
        raw = list(import_step(assembly / "assembly.step").children)
        parts = parts_from_step(raw, source["parts"])
        neutral_frames = robot.forward_kinematics({})
        localized_parts = [
            Part(
                p.name,
                p.owner,
                transform_shape(p.shape, np.linalg.inv(neutral_frames[p.owner])),
                p.role,
                p.module,
                p.color,
            )
            for p in parts
        ]
        exact9, exact9_meta = pose_from_trajectory(robot, name, commanded=commanded)
        inputs, _, _, _, _ = _source_inputs(
            Path(f"artifacts/{name}-forearm-accounted-upper-task-v1"), "task"
        )
        report["source_bindings"].update({str(p): sha(p) for p in inputs.values()})
        report["source_bindings"].update(
            {
                str(p): sha(p)
                for p in [
                    variant,
                    assembly / "report.json",
                    assembly / "resolved-config.json",
                    assembly / "interfaces.json",
                    assembly / "assembly.step",
                    Path(
                        f"artifacts/{name}-forearm-accounted-upper-task-v1/report.json"
                    ),
                    Path(
                        f"artifacts/{name}-forearm-accounted-upper-task-v1/trajectory.jsonl"
                    ),
                ]
            }
        )
        robot_rows = []
        for x in X_VARIANTS:
            variant_rows = []
            for label in POSES:
                pose = (
                    {"left_shoulder_roll_joint": float(label.rsplit("_", 1)[-1])}
                    if label.startswith("neutral")
                    else exact9
                )
                frames = robot.forward_kinematics(pose, enforce_limits=False)
                fixed_local = (x, 35 * math.sin(0.35), 35 * math.cos(0.35))
                moving_local = (x, 0, -20)
                fixed = endpoint_world(
                    neutral_frames["left_shoulder_roll"], fixed_local
                )
                moving = endpoint_world(frames["left_shoulder_roll"], moving_local)
                spring = spring_shape(fixed, moving)
                located = [
                    Part(
                        p.name,
                        p.owner,
                        transform_shape(p.shape, frames[p.owner]),
                        p.role,
                        p.module,
                        p.color,
                    )
                    for p in localized_parts
                ]
                queries = ExactQueries()
                collisions = queries.collisions(spring, located)
                gaps = (
                    []
                    if collisions
                    else sorted(float(spring.distance_to(p.shape)) for p in located)[:5]
                )
                stem = f"{name}-x{x:g}-{label.replace('-', '_')}"
                export_step(spring, output / (stem + ".step"))
                export_stl(spring, output / (stem + ".stl"))
                variant_rows.append(
                    {
                        "x_mm": x,
                        "pose": label,
                        "pose_rad": pose,
                        "pose_source": exact9_meta
                        if label.endswith("9s")
                        else "constructed joint pose; all other joints zero",
                        "anchors_world_mm": {
                            "fixed": fixed.tolist(),
                            "moving": moving.tolist(),
                        },
                        "length_mm": float(np.linalg.norm(moving - fixed)),
                        "radius_mm": 4.0,
                        "diameter_assumption_mm": 8.0,
                        "collisions": collisions,
                        "min_nominal_gaps_mm": gaps,
                        "step": stem + ".step",
                        "stl": stem + ".stl",
                    }
                )
            robot_rows.append({"x_mm": x, "cases": variant_rows})
        report["source_bindings"].update(
            {
                str(assembly / n): sha(assembly / n)
                for n in (
                    "report.json",
                    "resolved-config.json",
                    "interfaces.json",
                    "assembly.step",
                )
            }
        )
        report.setdefault("source_seals", {})[name] = {
            "source_hashes": source.get("source_hashes", {}),
            "output_hashes": source.get("output_hashes", {}),
            "report_status": source.get("status"),
            "geometry_build_status": source.get("geometry_build_status"),
        }
        report["robots"][name] = {
            "source_assembly_status": source.get("status"),
            "geometry_build_status": source.get("geometry_build_status"),
            "part_count": len(parts),
            "exact_9s": exact9_meta,
            "variants": robot_rows,
        }
    report["pose_mode"] = (
        "commanded_action_at_exact_9s"
        if commanded
        else "observed_joint_state_at_exact_9s"
    )
    report["geometry_status"] = (
        "pass"
        if all(
            not c["collisions"]
            for r in report["robots"].values()
            for x in r["variants"]
            for c in x["cases"]
        )
        else "fail"
    )
    report["inherited_source_failure"] = any(
        r["source_assembly_status"] != "pass" for r in report["robots"].values()
    )
    report["status"] = (
        "fail"
        if report["geometry_status"] == "fail"
        else ("partial" if report["inherited_source_failure"] else "pass")
    )
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)
    shutil.copy2(Path(__file__), output / "checker.py")
    # Recompute output hashes after checker copy, leaving report as the attestation file.
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--commanded", action="store_true")
    a = p.parse_args()
    print(run(a.output, a.commanded)["status"])
