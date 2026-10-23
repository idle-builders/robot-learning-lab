"""Select a bounded arm-rest hypothesis from static mass, before control trials."""

import argparse
from pathlib import Path
import yaml
from lab.config import load_config
from lab.robot import make_robot
from lab.mass_model import apply_mass_model
from lab.balance import neutral_mass_trim
from tools.check_camera_support import copy_source, sha, write


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        Path(__file__).relative_to(Path.cwd()),
        Path("variants/scout-distal-grip.yaml"),
        Path("robot/base.yaml"),
        *Path("lab").glob("*.py"),
    ]
    bindings = {}
    suites = {}
    for kind, source in [
        ("normal", "experiments/balance-composed-observed.yaml"),
        ("bounds", "experiments/balance-composed-observed-bounds.yaml"),
        ("stress", "experiments/balance-composed-observed-stress.yaml"),
    ]:
        p = Path(source)
        bindings[str(p)] = sha(p)
        suites[kind] = yaml.safe_load(p.read_text())
    rest = suites["normal"]["controller"]["arm_rest_targets_rad"]
    grid = [0, 0.02, 0.04, 0.06]
    rows = []
    for profile in ("lower", "nominal", "upper"):
        p = Path(f"artifacts/scout-shin-covered-mass-v1/{profile}/mass-properties.json")
        bindings[str(p)] = sha(p)
        robot = apply_mass_model(
            make_robot(load_config(Path("variants/scout-distal-grip.yaml"))), p
        )
        for angle in grid:
            pose = dict(rest, left_upper_arm_joint=angle, right_upper_arm_joint=angle)
            trim = neutral_mass_trim(robot, pose)
            rows.append(
                dict(
                    profile=profile,
                    angle_rad=angle,
                    pose_rad=pose,
                    static_trim_rad=trim["pitch_rad"],
                )
            )
    feasible = [
        a
        for a in grid
        if all(abs(r["static_trim_rad"]) <= 0.022 for r in rows if r["angle_rad"] == a)
    ]
    if not feasible:
        raise ValueError("No posture meets the predeclared static margin")
    selected = min(feasible)
    selected_pose = dict(
        rest, left_upper_arm_joint=selected, right_upper_arm_joint=selected
    )
    for kind, suite in suites.items():
        suite["controller"]["arm_rest_targets_rad"] = selected_pose
        (output / f"{kind}.yaml").write_text(yaml.safe_dump(suite, sort_keys=False))
    write(
        output / "report.json",
        dict(
            status="partial",
            scope=__doc__,
            selected_angle_rad=selected,
            selected_pose_rad=selected_pose,
            selection_grid_rad=grid,
            maximum_static_lean_rad=0.022,
            static_rows=rows,
            physical_status="not_run",
            dynamics_status="not_run",
            cad_status="not_run",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in paths}),
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )
    print(selected, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
