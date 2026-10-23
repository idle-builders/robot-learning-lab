"""Reuse verified CAD display assets for full-body task-scene recordings."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from tools.check_camera_support import copy_source, sha, write
from tools.check_forearm_task_scene import frame
from tools.compose_forearm_enclosure import verify_report


def run(display, assembly, scenes, episodes, names, output, fragment):
    output.mkdir(parents=True, exist_ok=False)
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                *Path("tools").glob("*.py"),
                Path("ui/full-scene-motion.fragment.html"),
            ]
        },
    )
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    old = verify_report(display)
    original = verify_report(assembly)
    assert original["assembly_step_roundtrip"] == "pass"
    assert old["source_bindings"][str(assembly / "report.json")] == sha(
        assembly / "report.json"
    )
    bind(display / "report.json")
    bind(assembly / "report.json")
    data = json.loads(bind(display / "dataset.json").read_text())
    base = next(r for r in data["robots"] if r["source"] == str(assembly))
    robot_parts = [p for p in base["parts"] if p["role"] != "scene"]
    assert {p["name"] for p in robot_parts} == {p["name"] for p in original["parts"]}
    assets = data["assets"]
    robots = []
    for scene_dir, episode, name in zip(scenes, episodes, names, strict=True):
        report = verify_report(scene_dir)
        assert report["source_hashes"]["full_assembly/report.json"] == sha(
            assembly / "report.json"
        )
        assert set(report["part_names"]) == {p["name"] for p in robot_parts}
        bind(scene_dir / "report.json")
        for item in ["report.json", "trajectory.jsonl", "scene.xml", "interfaces.json"]:
            assert report["source_hashes"]["episode/" + item] == sha(
                bind(episode / item)
            )
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
        parts = deepcopy(robot_parts)
        scenery = []
        owners = [o for o in base["owners"] if not o.startswith("environment_")]
        for i in range(model.ngeom):
            geom_name = model.geom(i).name
            if geom_name not in report["target_names"]:
                continue
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            body = model.body(int(model.geom_bodyid[i])).name
            relative = frame(model.geom_pos[i], model.geom_quat[i], scalar_first=True)
            obj = trimesh.creation.box(extents=2 * model.geom_size[i])
            index = len(assets)
            assets.append(
                [
                    np.rint(obj.vertices * 10000).astype(int).ravel().tolist(),
                    obj.faces.ravel().tolist(),
                ]
            )
            owner = "environment_" + geom_name
            owners.append(owner)
            parts.append(
                dict(
                    name=geom_name,
                    owner=owner,
                    role="scene",
                    asset=index,
                    color=[0.95, 0.49, 0.17]
                    if geom_name == "task_block_geom"
                    else [0.58, 0.62, 0.56],
                )
            )
            scenery.append((owner, body, relative))
        assert len(scenery) == 7
        samples = []
        for s in report["samples"]:
            row = rows[s["source_row"]]
            root_inverse = np.linalg.inv(np.asarray(s["root_transform"]))
            bodies = {
                n: root_inverse @ frame(b["position_m"], b["quaternion_xyzw"])
                for n, b in row["scene_state"]["bodies"].items()
            }
            for owner, body, relative in scenery:
                bodies[owner] = bodies[body] @ relative
            flagged = set()
            pairs = []
            for failure in s["failures"]:
                flagged.update([failure["part"], failure["nearest"]])
                for hit in failure["hits"]:
                    flagged.add(hit["part"])
                    pairs.append(
                        [failure["part"], hit["part"], round(hit["overlap_mm3"], 3)]
                    )
            assert flagged <= {p["name"] for p in parts}
            samples.append(
                dict(
                    name=s["name"],
                    kind="task",
                    time=s["time_s"],
                    phase=s["phase"],
                    status=s["status"],
                    scene_status=s["status"],
                    self_status="not_run",
                    gap=round(s["minimum_distance_mm"], 3),
                    pair=s["closest_pair"],
                    flagged=sorted(flagged),
                    pairs=pairs,
                    frames=[np.round(bodies[o], 7).ravel().tolist() for o in owners],
                )
            )
        robots.append(
            dict(
                name=name,
                parts=parts,
                owners=owners,
                samples=samples,
                status=report["status"],
                scene_status=report["status"],
                task_status="not_run",
                physics_status=json.loads((episode / "report.json").read_text())[
                    "status"
                ],
                source=str(scene_dir),
                assembly_part_count=len(robot_parts),
            )
        )
    payload = dict(assets=assets, robots=robots)
    compact = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    packed = base64.b64encode(gzip.compress(compact.encode(), mtime=0)).decode()
    template = bind(Path("ui/full-scene-motion.fragment.html"))
    text = template.read_text().replace("__FOREARM_DATA__", packed)
    assert len(text.encode()) < 1_000_000
    if fragment.exists():
        raise FileExistsError(fragment)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(text)
    (output / "motion.fragment.html").write_text(text)
    write(output / "dataset.json", payload)
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Display binding only; sampled full-body scene failures retained; no self-clearance claim",
            source_hashes=sources,
            source_bindings=bindings,
            output_hashes={
                n: sha(output / n) for n in ["dataset.json", "motion.fragment.html"]
            },
            fragment=str(fragment),
            fragment_sha256=sha(fragment),
            bytes=len(text.encode()),
            samples=[len(r["samples"]) for r in robots],
            parts=[r["assembly_part_count"] for r in robots],
        ),
    )
    print(
        json.dumps(
            dict(status="pass", fragment=str(fragment), bytes=len(text.encode()))
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["display", "assembly", "output", "fragment"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["scenes", "episodes"]:
        p.add_argument("--" + name, nargs="+", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    run(**vars(p.parse_args()))
