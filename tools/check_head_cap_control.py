"""Check translated head-cap negative controls against retained head shell."""

import json
from pathlib import Path

import numpy as np
from build123d import Pos, import_step

from lab.config import RobotConfig
from lab.leg_mechanism import overlap, transform_shape
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


ROOT = Path(__file__).resolve().parents[1]


def input_files(name, version):
    cap = ROOT / f"artifacts/{name}-head-cap-stock-{version}"
    assembly = ROOT / f"artifacts/{name}-distal-grip-integrated-v1"
    return {
        "cap_step": cap / "head_lower_front_left.step",
        "cap_config": cap / "resolved-config.json",
        "assembly_step": assembly / "assembly.step",
        "assembly_report": assembly / "report.json",
        "assembly_config": assembly / "resolved-config.json",
    }


def retained_head_shell(name, cache):
    if name in cache:
        return cache[name]
    assembly = ROOT / f"artifacts/{name}-distal-grip-integrated-v1"
    report = verify_report(assembly)
    cfg = RobotConfig.model_validate_json(
        (assembly / "resolved-config.json").read_text()
    )
    robot = make_robot(cfg)
    neutral = robot.forward_kinematics()
    parts = parts_from_step(
        list(import_step(assembly / "assembly.step").children), report["parts"]
    )
    shell = next(p.shape for p in parts if p.name == "head_shell")
    result = transform_shape(shell, np.linalg.inv(neutral["head"]))
    cache[name] = result
    return result


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "Supplemental translated front-left head-cap negative control against retained head shell; no fit or service claim",
        "displacements_mm": [1.0, 2.0],
        "cases": [],
    }
    source_names = [
        Path("lab/head_cap.py"),
        Path("lab/mechanical.py"),
        Path("lab/robot.py"),
        Path("tools/check_head_cap_control.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("tools/check_camera_support.py"),
        Path("robot/base.yaml"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): ROOT / p for p in source_names})
    shell_cache = {}
    for name in ("pip", "scout"):
        for version in ("v1", "v2"):
            files = input_files(name, version)
            missing = [str(p) for p in files.values() if not p.is_file()]
            if missing:
                report["cases"].append(
                    {"name": name, "version": version, "status": "not_run", "missing": missing}
                )
                continue
            bindings = {str(k): sha(v) for k, v in files.items()}
            case = {
                "name": name,
                "version": version,
                "input_paths": {str(k): str(v.relative_to(ROOT)) for k, v in files.items()},
                "input_hashes": bindings,
            }
            try:
                assembly_cfg = json.loads(files["assembly_config"].read_text())
                cap_cfg = json.loads(files["cap_config"].read_text())
                if cap_cfg != assembly_cfg:
                    raise ValueError("cap config differs from retained assembly config")
                RobotConfig.model_validate(cap_cfg)
                case["cap_source_dimensions"] = cap_cfg.get("dimensions", {})
                shell = retained_head_shell(name, shell_cache)
                cap = import_step(files["cap_step"])
                if len(cap.solids()) != 1:
                    raise ValueError("cap STEP is not one solid")
                case["head_frame"] = "retained assembly neutral head frame"
                rows = []
                for displacement in (1.0, 2.0):
                    moved = Pos(0, 0, displacement) * cap
                    volume = overlap(moved, shell)
                    rows.append(
                        {
                            "displacement_z_mm": displacement,
                            "overlap_mm3": volume,
                            "status": "pass" if volume > 0.02 else "fail",
                        }
                    )
                case["controls"] = rows
                case["status"] = "pass" if rows[0]["status"] == "pass" else "fail"
            except Exception as exc:
                case.update(status="fail", error=str(exc))
            report["cases"].append(case)
    runnable = [c for c in report["cases"] if c["status"] != "not_run"]
    report["status"] = (
        "pass"
        if len(runnable) == 4 and all(c["status"] == "pass" for c in runnable)
        else "fail"
    )
    report["output_hashes"] = {}
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(result["status"])
