"""Sample visual shell interference against the Cove-50 direct frame.

MuJoCo supplies actual joint transforms. Manifold3D intersects watertight
visual panel meshes, so this is a sampled CAD screen, not continuous clearance
or a controller-qualified motion corridor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from experiments.compact_morphology.ape_frame50 import build_cove_shell as shell
from experiments.compact_morphology.ape_frame50 import direct_skeleton_candidate as direct


def transform(mesh: trimesh.Trimesh, pos, mat) -> trimesh.Trimesh:
    result = mesh.copy()
    matrix = np.eye(4)
    matrix[:3, :3] = mat
    matrix[:3, 3] = pos
    result.apply_transform(matrix)
    return result


def overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    if np.any(np.minimum(a.bounds[1], b.bounds[1]) <= np.maximum(a.bounds[0], b.bounds[0])):
        return 0.0
    try:
        intersection = trimesh.boolean.intersection([a, b], engine="manifold")
        return max(0.0, float(intersection.volume) * 1e9) if intersection is not None else 0.0
    except ValueError:
        return -1.0  # invalid mesh is unknown, not clear


def body_meshes(folder: Path, report: dict) -> tuple[dict[str, trimesh.Trimesh], list[str]]:
    groups = defaultdict(list)
    excluded = []
    for entry in report["shell_parts_by_body"]:
        name = entry["name"]
        mesh = trimesh.load_mesh(folder / "shell-meshes" / f"shell_{name}.obj", process=True)
        if entry["role"] == "soft":
            # Deformation is unspecified: a rigid intersection test of this
            # visual sleeve would give a false motion stop or false clearance.
            excluded.append(name + " (flex envelope)")
        elif mesh.is_volume:
            groups[entry["body"]].append(mesh)
        else:
            excluded.append(name)
    return {body: trimesh.util.concatenate(parts) for body, parts in groups.items()}, excluded


def pose_bodies(model, data, meshes) -> dict[str, trimesh.Trimesh]:
    result = {}
    for body, mesh in meshes.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        result[body] = transform(mesh, data.xpos[bid], data.xmat[bid].reshape(3, 3))
    return result


def collisions(world: dict[str, trimesh.Trimesh], threshold_mm3=10.0) -> list[dict]:
    names = list(world)
    hits = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            volume = overlap(world[a], world[b])
            if volume > threshold_mm3:
                hits.append({"a": a, "b": b, "overlap_mm3": round(volume, 1)})
            elif volume < 0:
                hits.append({"a": a, "b": b, "overlap_mm3": None})
    return hits


def motor_shell_hits(model, data, shell_world, threshold_mm3=10.0):
    hits = []
    for drive in direct.drives():
        case = trimesh.creation.box(extents=np.array(drive.size) / 1000)
        center = np.array(drive.center) / 1000
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, drive.parent)
        rotation = data.xmat[bid].reshape(3, 3)
        case.vertices += (center - data.xpos[bid]) @ rotation
        case = transform(case, data.xpos[bid], rotation)
        for body, surface in shell_world.items():
            volume = overlap(case, surface)
            if volume > threshold_mm3:
                hits.append({"motor": drive.name, "shell_body": body, "overlap_mm3": round(volume, 1)})
    return hits


def pose_record(model, data, body_names):
    return [[round(float(v), 6) for v in (*data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)],
                                         *data.xquat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)])]
            for body in body_names]


def screen(folder: Path, samples: int = 9) -> dict:
    report = json.loads((folder / "report.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "cove50-shell-visual.xml"))
    data = mujoco.MjData(model)
    meshes, excluded = body_meshes(folder, report)
    body_names = list(report["frame_body_glbs"])
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    neutral_world = pose_bodies(model, data, meshes)
    neutral_hits = collisions(neutral_world)
    motor_hits = motor_shell_hits(model, data, neutral_world)
    baseline_pairs = {tuple(sorted((hit["a"], hit["b"]))) for hit in neutral_hits}
    sweeps = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        low, high = model.jnt_range[jid]
        values = sorted(set([float(low), 0.0, float(high), *np.linspace(low, high, samples)]))
        frames = []
        for angle in values:
            data.qpos[:] = model.qpos0
            data.qpos[model.jnt_qposadr[jid]] = angle
            mujoco.mj_forward(model, data)
            world = pose_bodies(model, data, meshes)
            hits = collisions(world)
            new_hits = [h for h in hits if tuple(sorted((h["a"], h["b"]))) not in baseline_pairs]
            frames.append({"angle_deg": round(float(np.degrees(angle)), 2),
                           "body_poses": pose_record(model, data, body_names),
                           "shell_hits": hits,
                           "new_shell_hits": new_hits,
                           "has_shell_interference": bool(hits)})
        sweeps[name] = {"model_limit_deg": [round(float(np.degrees(low)), 2),
                                            round(float(np.degrees(high)), 2)],
                        "samples": frames}
    result = {
        "method": "Manifold3D exact triangle-volume intersection of watertight shell panels at independent sampled joint angles; 10 mm3 reporting threshold",
        "source_shell_visual_sha256": shell.digest(folder / "cove50-shell-visual.xml"),
        "source_report_sha256": shell.digest(folder / "report.json"),
        "body_names": body_names,
        "excluded_nonvolume_decorative_panels": excluded,
        "neutral_shell_hits": neutral_hits,
        "neutral_motor_shell_hits": motor_hits,
        "joint_sweeps": sweeps,
        "limitations": ["Independent one-joint samples do not qualify simultaneous motion or continuous paths.",
                       "Dark hip, knee and ankle gaiters are visual flexible envelopes and are excluded from rigid interference; their bend, stretch and clearance are unqualified.",
                       "The screen excludes non-watertight decorative panels; camera apertures and smooth paws need direct CAD clearance checks.",
                       "Shell geoms remain visual-only, so these results do not constrain the saved learned controller."],
    }
    trace = json.loads((folder / "kinematics.json").read_text())
    trace_names = trace["body_names"]
    walk = []
    for index in range(0, len(trace["frames"]), 4):
        frame = trace["frames"][index]
        world = {}
        for body, mesh in meshes.items():
            row = frame[trace_names.index(body)]
            quat = np.array(row[3:])
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, quat)
            world[body] = transform(mesh, np.array(row[:3]), rotation.reshape(3, 3))
        hits = collisions(world)
        walk.append({"frame": index, "time_s": round(index * trace["dt"], 3),
                     "shell_hits": hits,
                     "new_shell_hits": [h for h in hits if tuple(sorted((h["a"], h["b"]))) not in baseline_pairs]})
    result["saved_walk_sample_stride"] = 4
    result["saved_walk_shell_samples"] = walk
    (folder / "motion-screen.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=shell.DEFAULT_OUT)
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    result = screen(args.out.resolve(), args.samples)
    counts = {joint: sum(p["has_shell_interference"] for p in value["samples"])
              for joint, value in result["joint_sweeps"].items()}
    print(json.dumps({"neutral_shell_hits": len(result["neutral_shell_hits"]),
                      "neutral_motor_shell_hits": len(result["neutral_motor_shell_hits"]),
                      "sample_collision_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
