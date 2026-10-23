"""Verify archived carrier orientation mapping against source mechanism spacers."""

import argparse
from pathlib import Path
import runpy
from build123d import export_step, import_step
from lab.config import load_config
from lab.robot import make_robot
from lab.leg_mechanism import yoke_parts
from tools.check_camera_support import copy_source, sha, write


def run(study, output):
    output.mkdir(parents=True, exist_ok=False)
    archived = study / "source/tools/check_carrier_robot_stock.py"
    module = runpy.run_path(str(archived))
    files = [
        Path("tools/check_carrier_registration.py"),
        archived,
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
    bindings = copy_source(output, {str(p): p for p in files})
    rows = []
    _, base, _ = yoke_parts()
    for robot_name in ["pip", "scout"]:
        robot = make_robot(load_config(Path(f"variants/{robot_name}-distal-grip.yaml")))
        expected = module["expected_spacers"](robot)
        assert len(expected) == 12
        for name, part in expected.items():
            mapped = module["oriented_web"](name, base)
            stem = f"{robot_name}-{name}"
            export_step(mapped, output / f"{stem}-mapped.step")
            export_step(part.shape, output / f"{stem}-source.step")
            a = import_step(output / f"{stem}-mapped.step")
            b = import_step(output / f"{stem}-source.step")

            def vol(shape):
                return 0 if shape is None else sum(x.volume for x in shape.solids())

            missing = vol(b - a)
            added = vol(a - b)
            rows.append(
                dict(
                    robot=robot_name,
                    module=name,
                    added_mm3=added,
                    missing_mm3=missing,
                    status="pass" if max(added, missing) < 1e-4 else "fail",
                )
            )
    write(
        output / "report.json",
        dict(
            status="pass" if all(r["status"] == "pass" for r in rows) else "fail",
            scope=__doc__,
            checked_mapping_source=str(archived),
            source_hashes=bindings,
            rows=rows,
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
