"""Show complete retained CAD at explicit control rest postures."""

import argparse
import base64
import gzip
import json
from pathlib import Path
from lab.config import load_config
from lab.robot import make_robot
from tools.review_integral_head_service import embedded_dataset
from tools.compose_forearm_enclosure import verify_report
from tools.check_camera_support import sha, write
from tools.build_product_review import wrapper


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/integral-head-composed-review-v2")
    verify_report(source)
    bindings = {}

    def bind(p):
        p = Path(p)
        bindings[str(p)] = sha(p)
        return p

    bind(source / "report.json")
    data = embedded_dataset(bind(source / "index.html"))
    for candidate in data["robots"]:
        name = candidate["name"].split(" · ")[0].lower()
        variant = bind(f"variants/{name}-distal-grip.yaml")
        robot = make_robot(load_config(variant))
        pair_folder = Path("artifacts/balance-arm-clearance-complete-v1")
        verify_report(pair_folder)
        pair_report = json.loads(bind(pair_folder / "report.json").read_text())
        pairs = next(r for r in pair_report["robots"] if r["robot"] == name)
        samples = []
        for label, roll in (
            ("zero_pose", None),
            ("elbows_parked", 0),
            ("elbows_parked_shoulders_outward", 0.04),
        ):
            targets = (
                {}
                if roll is None
                else {
                    "left_forearm_joint": -0.4,
                    "right_forearm_joint": -0.4,
                    "left_shoulder_roll_joint": roll,
                    "right_shoulder_roll_joint": -roll,
                }
            )
            fk = robot.forward_kinematics(targets)
            failed = []
            if roll is None:
                failed = [
                    p
                    for p in pairs["poses"][0]["pairs"]
                    if p["intersection_volume_mm3"] > 1e-6
                ]
                note = "Exact wheel/arm pair collisions highlighted. Full-robot failures also remain."
                status = "fail"
            elif roll == 0:
                note = "Selected wheel/arm CAD pairs clear. Primitive thigh touching/overlap remains. No full fit claim."
                status = "partial"
            else:
                note = "Proposed control rest: elbows −0.4 rad, shoulder rolls +0.04/−0.04 rad. Primitive initial contact list empty. Separate detailed thigh checks; no full fit claim."
                status = "partial"
            samples.append(
                dict(
                    name=label,
                    kind="pose",
                    phase="",
                    status=status,
                    clearance_status=status,
                    flagged=sorted(
                        {p[k] for p in failed for k in ("wheel_part", "arm_part")}
                    ),
                    pair=[failed[0]["wheel_part"], failed[0]["arm_part"]]
                    if failed
                    else ["No highlighted pair is not proof of full clearance"],
                    gap_note=note,
                    frames=[fk[o].flatten().tolist() for o in candidate["owners"]],
                    joint_targets_rad=targets,
                )
            )
        candidate["samples"] = samples
        candidate["stock_summary"] = (
            "Complete composed CAD, explicit static rest postures. Inherited assembly FAIL; display is not manufacturing or motion qualification."
        )
        for p in candidate["parts"]:
            p.pop("download", None)
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    fragment = bind("ui/composed-head.fragment.html").read_text()
    fragment = fragment.replace(
        "Complete robot candidates · geometry and planning mass",
        "Control rest postures · inspect the complete geometry",
    )
    fragment = fragment.replace("Neutral changed-stock screen:", "Posture evidence:")
    fragment = fragment.replace(
        "Four replacement groups are composed into each complete robot: two integral shells and two socketed bridge stacks. Nominal geometry and planning mass are separate from fit, load, print qualification and control. Full robot failures remain explicit. No new motion or dynamics is claimed; simplified display meshes preserve source identity.",
        "Three static postures use the same composed-assembly display assets and shared kinematic tree. Neutral wheel/finger collisions are highlighted; absence of highlight is not full clearance. Arm/leg geometry, movement, print strength and physical fit need separate evidence. This is not a trajectory replay.",
    )
    fragment = fragment.replace('type="checkbox" checked', 'type="checkbox"')
    (output / "index.html").write_text(
        wrapper(fragment.replace("__HEAD_MOUNT_DATA__", encoded))
    )
    (output / "review.css").write_text(bind("ui/product-review.css").read_text())
    bind(Path(__file__).relative_to(Path.cwd()))
    bind("lab/robot.py")
    bind("robot/base.yaml")
    write(
        output / "report.json",
        dict(
            status="partial",
            scope="Complete-CAD static posture presentation, not full clearance",
            source_bindings=bindings,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
