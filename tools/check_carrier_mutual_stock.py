"""Bounded mutual added-web collisions, separate from stock versus old assembly."""

import argparse
from itertools import combinations
from pathlib import Path
import runpy
from build123d import Compound
from lab.config import load_config
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_carrier_robot_stock import bbox_disjoint
from tools.check_neck_collar import ExactQueries
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(study, output):
    output.mkdir(parents=True, exist_ok=False)
    registration = verify_report(Path("artifacts/carrier-registration-v1"))
    assert registration["status"] == "pass"
    archived = study / "source/tools/check_carrier_robot_stock.py"
    assert str(archived) == registration["checked_mapping_source"]
    m = runpy.run_path(str(archived))
    _, _, added = m["carrier_shapes"]()
    web = Compound(added)
    files = [
        Path(__file__).relative_to(Path.cwd()),
        archived,
        Path("tools/check_carrier_robot_stock.py"),
        Path("tools/check_neck_collar.py"),
        Path("artifacts/carrier-registration-v1/report.json"),
        *[
            Path("lab") / n
            for n in [
                "config.py",
                "robot.py",
                "arm_mechanism.py",
                "neck_mechanism.py",
                "leg_mechanism.py",
                "yoke_spacers.py",
            ]
        ],
        Path("robot/base.yaml"),
        Path("variants/pip-distal-grip.yaml"),
        Path("variants/scout-distal-grip.yaml"),
    ]
    hashes = copy_source(output, {str(p): p for p in files})
    rows = []
    for name in ["pip", "scout"]:
        robot = make_robot(load_config(Path(f"variants/{name}-distal-grip.yaml")))
        parts = m["expected_spacers"](robot)
        for pose, overrides in m["POSES"]:
            fk = robot.forward_kinematics(overrides)
            stocks = {
                n: transform_shape(m["oriented_web"](n, web), fk[p.owner])
                for n, p in parts.items()
            }
            bounds = ExactQueries()
            for a, b in combinations(sorted(stocks), 2):
                skip = bbox_disjoint(bounds.bounds(stocks[a]), bounds.bounds(stocks[b]))
                overlap = 0 if skip else m["volume"](stocks[a].intersect(stocks[b]))
                rows.append(
                    dict(
                        robot=name,
                        pose=pose,
                        module_a=a,
                        module_b=b,
                        aabb_disjoint=skip,
                        overlap_mm3=overlap,
                        status="pass" if overlap <= 1e-6 else "fail",
                    )
                )
    assert len(rows) == 396
    write(
        output / "report.json",
        dict(
            status="pass" if all(r["status"] == "pass" for r in rows) else "fail",
            scope=__doc__,
            source_hashes=hashes,
            rows=rows,
            physical_status="not_run",
            broadphase_method="Cached BRepBndLib.Add_s(shape, box, False); direct unpruned Boolean narrow phase",
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.study, a.output)
