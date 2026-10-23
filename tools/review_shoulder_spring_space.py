"""Inspect hypothetical shoulder-spring reservations against complete robot views."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
from build123d import import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import sha, write, copy_source
from tools.compose_forearm_enclosure import verify_report, kinematic_contract
from tools.review_integral_head_service import embedded_dataset, add_asset
from tools.build_product_review import wrapper


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    folder = Path("artifacts/shoulder-spring-space-v3")
    screen = verify_report(folder)
    original = Path("artifacts/integral-head-composed-review-v2")
    retained = verify_report(original)
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    bind(folder / "report.json")
    bind(original / "report.json")
    data = embedded_dataset(bind(original / "index.html"))
    candidates = []
    for baseline in data["robots"]:
        name = baseline["name"].split(" · ")[0].lower()
        assembly = Path(f"artifacts/{name}-integral-head-composed-v2")
        verify_report(assembly)
        if (
            screen["source_bindings"][str(assembly / "report.json")]
            != retained["source_bindings"][str(assembly / "report.json")]
        ):
            raise ValueError("Spring and display assembly identity differs")
        robot = make_robot(
            RobotConfig.model_validate_json(
                bind(assembly / "resolved-config.json").read_text()
            )
        )
        task = Path(f"artifacts/{name}-forearm-accounted-upper-task-v1")
        interface = json.loads(bind(task / "interfaces.json").read_text())
        original_interface = json.loads(bind(assembly / "interfaces.json").read_text())
        if kinematic_contract(interface) != kinematic_contract(original_interface):
            raise ValueError("Spring source kinematic contract differs")
        rows = [
            json.loads(line)
            for line in bind(task / "trajectory.jsonl").read_text().splitlines()
        ]
        task_row = min(rows, key=lambda r: abs(r["joint_state"]["time_s"] - 9))
        if abs(task_row["joint_state"]["time_s"] - 9) > 1e-9:
            raise ValueError("9s command missing")
        for variant in screen["robots"][name]["variants"]:
            candidate = deepcopy(baseline)
            for part in candidate["parts"]:
                part.pop("download", None)
            candidate["owners"].append("spring_world")
            candidate["parts"].append(
                dict(
                    name="proposed_spring_reservation",
                    owner="spring_world",
                    role="spring_reservation",
                    asset=0,
                    color=[0.86, 0.39, 0.15],
                    download="placeholder",
                )
            )
            candidate["samples"] = []
            for row in variant["cases"]:
                if row["pose"].startswith("neutral"):
                    pose = {
                        "left_shoulder_roll_joint": float(
                            row["pose"].rsplit("_", 1)[-1]
                        )
                    }
                else:
                    if row["pose_source"].get("joint_state_field") != "action":
                        raise ValueError("Expected commanded pose")
                    pose = dict(
                        zip(
                            task_row["joint_state"]["names"],
                            task_row["action"],
                            strict=True,
                        )
                    )
                if "pose_rad" in row and pose != row["pose_rad"]:
                    raise ValueError("Displayed pose differs from checked pose")
                frames = robot.forward_kinematics(pose, enforce_limits=False)
                asset, _ = add_asset(
                    data,
                    import_step(bind(folder / row["step"])),
                    output / (name + "-" + row["step"].replace(".step", ".stl")),
                )
                hits = [c["part"] for c in row["collisions"]]
                candidate["samples"].append(
                    dict(
                        name=row["pose"],
                        kind="pose",
                        phase="",
                        status="fail" if hits else "pass",
                        clearance_status="fail" if hits else "pass",
                        flagged=hits
                        + (["proposed_spring_reservation"] if hits else []),
                        pair=hits or ["No reservation overlap at this sample"],
                        gap_note=f"8 mm assumed capsule · {row['length_mm']:.2f} mm pin spacing · {len(hits)} intersections. No anchors/brackets, complete spring shape or continuous-motion qualification.",
                        frames=[
                            np.eye(4).flatten().tolist()
                            if o == "spring_world"
                            else frames[o].flatten().tolist()
                            for o in candidate["owners"]
                        ],
                        spring_asset=asset,
                        download="../" + folder.name + "/" + row["stl"],
                    )
                )
            candidate["parts"][-1]["asset"] = candidate["samples"][0]["spring_asset"]
            candidate.update(
                name=f"{name.title()} · spring plane X={variant['x_mm']:g} mm",
                source=str(folder),
                inherited_status=screen["robots"][name]["source_assembly_status"],
                stock_summary="Hypothetical pin-to-pin capsule only · primary robot unchanged · whole assembly FAIL · mounting/dynamics/physical NOT RUN",
            )
            candidates.append(candidate)
    data["robots"] = candidates
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    template = Path("ui/shoulder-spring.fragment.html")
    (output / "index.html").write_text(
        wrapper(template.read_text().replace("__HEAD_MOUNT_DATA__", encoded))
    )
    (output / "review.css").write_bytes(Path("ui/product-review.css").read_bytes())
    sources = [
        Path(__file__).relative_to(Path.cwd()),
        template,
        Path("ui/product-review.css"),
        Path("tools/review_integral_head_service.py"),
        Path("tools/cap_mesh.py"),
        Path("tools/build_product_review.py"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Display/source identity only; hypothetical spring reservation is not a mounted mechanism",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in sources}),
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            physical="not_run",
            candidate_dynamics="not_run",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
