"""Retain fitted split-cover geometry, sampled arm clearance and service evidence."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml
from build123d import Color, Compound, Pos, export_step, import_step

from lab.artifact_store import snapshot_file
from lab.arm_mechanism import arm_parts
from lab.config import load_config
from lab.forearm_cover import candidate
from lab.leg_mechanism import Part, VENDOR, located_parts, render_core, transform_shape
from lab.robot import make_robot
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_camera_support import CameraQueries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def poses(robot):
    yield "neutral", {}
    for side in ["left", "right"]:
        names = {
            s: f"{side}_{s}_joint"
            for s in ["forearm", "gripper", "upper_arm", "shoulder_roll"]
        }
        limits = {
            s: next(j.limits for j in robot.joints if j.name == n)
            for s, n in names.items()
        }
        for suffix in names:
            for i, value in enumerate(limits[suffix]):
                yield f"{side}-{suffix}-{i}", {names[suffix]: value}
        for a, b in [("forearm", "gripper"), ("upper_arm", "forearm")]:
            for i, av in enumerate(limits[a]):
                for j, bv in enumerate(limits[b]):
                    yield f"{side}-{a}{i}-{b}{j}", {names[a]: av, names[b]: bv}


def run(variant, spec_path, output, *, neutral_only=False):
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        *Path("lab").glob("*.py"),
        variant,
        spec_path,
        Path("robot/base.yaml"),
        Path("robot/fastener-catalog.yaml"),
        Path(__file__).resolve(),
        Path("tools/check_camera_support.py"),
        Path("tools/check_neck_collar.py"),
        Path("tools/cad_identity.py"),
        VENDOR,
    ]
    source_hashes = {}
    for path in sources:
        path = path.absolute()
        rel = path.relative_to(Path.cwd())
        dest = output / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, dest)
        source_hashes[str(rel)] = sha(dest)
    spec = yaml.safe_load(spec_path.read_text())
    robot = make_robot(load_config(variant))
    write(output / "resolved-config.json", robot.config.model_dump(mode="json"))
    print("Build detailed arms", flush=True)
    arm, _ = arm_parts(robot)
    old = {p.name: p for p in arm}
    additions, replacements, candidates, stock_probes = [], {}, {}, []
    checks = []
    for side in ["left", "right"]:
        owner = f"{side}_forearm"
        original = old[f"{side}_forearm_spine"]
        c = candidate(original.shape, spec, right=side == "right")
        candidates[side] = c
        replacements[original.name] = replace(original, shape=c["spine"])
        for category in ["halves", "screws", "inserts"]:
            for key, shape in c[category].items():
                name = f"{side}_cover_{category}_{key}"
                color = (
                    (0.28, 0.66, 0.61) if category == "halves" else (0.55, 0.57, 0.57)
                )
                additions.append(Part(name, owner, shape, category, owner, color))
        for name, shape in c["halves"].items():
            overlap = float((shape & c["spine"]).volume)
            checks.append(
                dict(
                    kind="cover_spine_seat",
                    side=side,
                    part=name,
                    overlap_mm3=overlap,
                    distance_mm=shape.distance_to(c["spine"]),
                    status="pass" if overlap < 0.02 else "fail",
                )
            )
        checks.append(
            dict(
                kind="split_seam",
                side=side,
                distance_mm=c["halves"]["front"].distance_to(c["halves"]["rear"]),
                status="pass"
                if c["halves"]["front"].distance_to(c["halves"]["rear"])
                >= spec["split_gap"] - 1e-5
                else "fail",
            )
        )
        stock_targets = [p for p in arm if p.owner == owner and p.name != original.name]
        added_stock = c["spine"] - original.shape
        stock_probes.append(
            Part(
                f"{side}_added_spine_stock",
                owner,
                added_stock,
                "stock",
                owner,
                (0.36, 0.64, 0.57),
            )
        )
        stock_hits = CameraQueries().collide(added_stock, stock_targets)
        checks.append(
            dict(
                kind="added_spine_stock",
                side=side,
                hits=stock_hits,
                status="pass" if not stock_hits else "fail",
            )
        )
    all_parts = [replacements.get(p.name, p) for p in arm] + additions
    sample_list = [("neutral", {})] if neutral_only else list(poses(robot))
    rows = []
    # Query in forearm-local coordinates and cache repeated relative poses.
    # Rounded transforms form cache keys only, never substitute queried geometry.
    cache = {}
    for label, pose in sample_list:
        frames = robot.forward_kinematics(pose)
        print(f"Check {label}", flush=True)
        for side in ["left", "right"]:
            owner = f"{side}_forearm"
            inverse = np.linalg.inv(frames[owner])
            targets = []
            for p in arm:
                if p.name == f"{side}_forearm_spine":
                    continue
                matrix = inverse @ frames[p.owner]
                key = (p.name, tuple(np.round(matrix.ravel(), 10)))
                if key not in cache:
                    cache[key] = replace(p, shape=transform_shape(p.shape, matrix))
                targets.append(cache[key])
            query = CameraQueries()
            for p in [p for p in additions + stock_probes if p.owner == owner]:
                key = (p.name, tuple(id(t.shape) for t in targets))
                if key not in cache:
                    hits = query.collide(p.shape, targets)
                    distance, nearest = query.nearest(p.shape, targets)
                    cache[key] = dict(hits=hits, distance_mm=distance, nearest=nearest)
                data = cache[key]
                rows.append(
                    dict(
                        pose=label,
                        part=p.name,
                        **data,
                        status="pass"
                        if not data["hits"]
                        and data["distance_mm"] >= spec["clearance_target"] - 1e-5
                        else "fail",
                    )
                )
    print("Check service, seats and export", flush=True)
    # Removal at neutral only, after all four screws have been removed per arm.
    service = []
    frames = robot.forward_kinematics()
    for side in ["left", "right"]:
        owner = f"{side}_forearm"
        c = candidates[side]
        targets = [
            replace(
                p,
                shape=transform_shape(
                    p.shape, np.linalg.inv(frames[owner]) @ frames[p.owner]
                ),
            )
            for p in arm
            if p.name != f"{side}_forearm_spine"
        ]
        targets.append(replacements[f"{side}_forearm_spine"])
        for half, sign in [("front", 1), ("rear", -1)]:
            other = "rear" if half == "front" else "front"
            target_list = targets + [
                Part(
                    "other_cover", owner, c["halves"][other], "cover", owner, (1, 1, 1)
                )
            ]
            for travel in [0.5, 2, 10, 35]:
                shape = Pos(sign * travel, 0, 0) * c["halves"][half]
                hits = CameraQueries().collide(shape, target_list)
                service.append(
                    dict(
                        side=side,
                        half=half,
                        travel_mm=travel,
                        hits=hits,
                        status="pass" if not hits else "fail",
                    )
                )
        for key, tool in c["tools"].items():
            hits = CameraQueries().collide(tool, targets)
            checks.append(
                dict(
                    kind="driver_access",
                    side=side,
                    mount=key,
                    hits=hits,
                    status="pass" if not hits else "fail",
                )
            )
    # Validate detection with a known overlapping witness against retained stock.
    from build123d import Box

    witness = Pos(0, 16.865, -60) * Box(4, 4, 4)
    negative = CameraQueries().collide(witness, [old["left_forearm_spine"]])
    changed = list(replacements.values()) + additions
    shapes = []
    identity = {}
    for p in changed:
        shape = p.shape
        shape.label = p.name
        shape.color = Color(*p.color)
        shapes.append(shape)
        identity[p.name] = geometry_record(shape)
    export_step(Compound(children=shapes), output / "cover-assembly.step")
    restored = import_step(output / "cover-assembly.step")
    restored_map = {s.label: s for s in restored.children}
    step_ok = set(restored_map) == set(identity) and all(
        geometry_agrees(geometry_record(restored_map[n]), identity[n]) for n in identity
    )
    render_core(all_parts, located_parts(robot, all_parts), output, robot)
    mass = {}
    for side, c in candidates.items():
        old_spine = old[f"{side}_forearm_spine"].shape
        polymer = (
            sum(s.volume for s in c["halves"].values())
            + c["spine"].volume
            - old_spine.volume
        )
        brass = sum(s.volume for s in c["inserts"].values())
        steel = sum(s.volume for s in c["screws"].values())
        mass[side] = dict(
            added_polymer_mm3=polymer,
            added_polymer_g=polymer * spec["density_g_cm3"] / 1000,
            hardware_g=brass * 0.0085 + steel * 0.008,
            total_added_g=polymer * spec["density_g_cm3"] / 1000
            + brass * 0.0085
            + steel * 0.008,
        )
    report = dict(
        status="pass"
        if step_ok
        and negative
        and all(r["status"] == "pass" for r in rows + service + checks)
        else "fail",
        scope="Added split forearm covers and hardware versus detailed arm parts; sampled motion and neutral service only",
        variant=str(variant),
        source_hashes=source_hashes,
        poses=[n for n, _ in sample_list],
        checks=checks,
        clearance_target_mm=spec["clearance_target"],
        rows=rows,
        service=service,
        negative_control=negative,
        step_identity_pass=step_ok,
        geometry=identity,
        mass=mass,
        not_run=[
            "full integrated torso/head/leg clearance",
            "continuous motion and pinch safety",
            "harness and cable bend radii",
            "physical fit, insert installation and strength",
            "updated robot dynamics",
            "manufacturing and exact production quotes",
        ],
    )
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()
    }
    write(output / "report.json", report)
    print(
        json.dumps(
            dict(
                status=report["status"],
                failed_rows=sum(r["status"] == "fail" for r in rows),
                mass=mass,
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant", type=Path, default=Path("variants/scout-head-service.yaml")
    )
    p.add_argument("--spec", type=Path, default=Path("robot/forearm-cover.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--neutral-only", action="store_true")
    a = p.parse_args()
    run(a.variant, a.spec, a.output, neutral_only=a.neutral_only)
