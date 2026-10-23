"""Check opposite forearm additions against each other in the producer poses."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

from build123d import import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape
from tools.check_forearm_cover import poses, sha, write
from tools.check_camera_support import CameraQueries


def run(cover, output):
    output.mkdir(parents=True, exist_ok=False)
    r = json.loads((cover / "report.json").read_text())
    for name, h in r["output_hashes"].items():
        if sha(cover / name) != h:
            raise ValueError("changed cover output")
    for name in [
        "lab/config.py",
        "lab/robot.py",
        "lab/leg_mechanism.py",
        "tools/check_forearm_cover.py",
        "tools/check_camera_support.py",
        "tools/check_neck_collar.py",
    ]:
        if sha(Path(name)) != r["source_hashes"][name]:
            raise ValueError(f"changed query source {name}")
    robot = make_robot(
        RobotConfig.model_validate_json((cover / "resolved-config.json").read_text())
    )
    raw = import_step(cover / "cover-assembly.step")
    if len(raw.children) != len(r["geometry"]) or {
        p.label for p in raw.children
    } != set(r["geometry"]):
        raise ValueError("part inventory differs")
    parts = [
        Part(
            p.label,
            p.label.split("_")[0] + "_forearm",
            p,
            "cover_probe",
            "cover_probe",
            (1, 1, 1),
        )
        for p in raw.children
    ]
    rows = []
    for label, pose in poses(robot):
        frames = robot.forward_kinematics(pose)
        placed = [
            replace(p, shape=transform_shape(p.shape, frames[p.owner])) for p in parts
        ]
        query = CameraQueries()
        right = [p for p in placed if p.owner == "right_forearm"]
        hits = []
        for p in placed:
            if p.owner == "left_forearm":
                hits.extend(
                    dict(left=p.name, **hit) for hit in query.collide(p.shape, right)
                )
        rows.append(dict(pose=label, hits=hits, status="fail" if hits else "pass"))
    assert [x["pose"] for x in rows] == r["poses"]
    source = output / "source"
    source.mkdir()
    shutil.copy2(__file__, source / "check_cover_pairs.py")
    result = dict(
        status="fail" if any(x["hits"] for x in rows) else "pass",
        scope="Opposite-arm new covers, hardware and modified spines, sampled poses only",
        rows=rows,
        source_bindings={
            str(cover / n): sha(cover / n)
            for n in ["report.json", "resolved-config.json", "cover-assembly.step"]
        },
        source_hashes={"check_cover_pairs.py": sha(source / "check_cover_pairs.py")},
    )
    write(output / "report.json", result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                failed_poses=[x["pose"] for x in rows if x["hits"]],
            )
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cover", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.cover, a.output)
