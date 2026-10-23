"""Show the separate upper-arm guard proposal on its exact retained robots."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path

from build123d import import_step
from lab.artifact_store import retain_dependency
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_integral_head_service import add_asset, embedded_dataset


def motion_poses(report):
    def convert(label, pose, note):
        return dict(
            name=label,
            pose_rad=pose["pose"],
            status=pose["status"],
            note=note,
            rows=[
                dict(
                    part=f["part"],
                    hits=[{"part": f["target"]}],
                    status="fail",
                    nearest=None,
                )
                for f in pose["failures"]
            ],
        )

    rows = [
        convert(
            "current_commanded_rest",
            report["rest_pose"],
            "Current commanded rest; no observed balance trajectory.",
        )
    ]
    for case in report["cases"]:
        rows += [
            convert(
                f"{case['name']} · {p['time_s']:.2f}s · {p['phase']}",
                p,
                f"Recorded task outcome: {case['source_status'].upper()} · {case['name']} · row {p['source_row']} · {len(p['joint_limit_excursions'])} recorded joint-limit excursions. Root held neutral for self-clearance; scene objects and continuous paths are not checked.",
            )
            for p in case["poses"]
        ]
    return rows


def run(study, output, motion_pip=None, motion_scout=None):
    if bool(motion_pip) != bool(motion_scout):
        raise ValueError("Both robot motion reports are required")
    output.mkdir(parents=True, exist_ok=False)
    result = verify_report(study)
    base = Path("artifacts/covered-carriers-composed-review-v1")
    original = verify_report(base)
    data = embedded_dataset(base / "index.html")
    bindings = {
        str(study / "report.json"): sha(study / "report.json"),
        str(base / "report.json"): sha(base / "report.json"),
        str(base / "index.html"): sha(base / "index.html"),
    }
    robots = []
    dependencies = dict(
        study=retain_dependency(study), baseline=retain_dependency(base)
    )
    for baseline in data["robots"]:
        name = baseline["name"].split(" · ")[0].lower()
        r = result["robots"][name]
        selected_poses = r["poses"]
        checked_status = r["status"]
        if motion_pip:
            folder = motion_pip if name == "pip" else motion_scout
            motion = verify_report(folder)
            if motion["robot"] != name or motion["source_bindings"].get(
                str(study / "report.json")
            ) != sha(study / "report.json"):
                raise ValueError("Motion checks refer to a different guard candidate")
            bindings[str(folder / "report.json")] = sha(folder / "report.json")
            dependencies[f"motion-{name}"] = retain_dependency(folder)
            selected_poses = motion_poses(motion)
            checked_status = motion["added_guard_status"]
        source = Path(r["source"])
        key = str(source / "report.json")
        if original["source_bindings"].get(key) != result["source_bindings"][key]:
            raise ValueError("Guard and displayed robot source differ")
        robot = make_robot(
            RobotConfig.model_validate_json(
                (source / "resolved-config.json").read_text()
            )
        )
        for enabled in (False, True):
            item = deepcopy(baseline)
            for p in item["parts"]:
                p.pop("download", None)
            if enabled:
                for p in r["parts"]:
                    path = study / p["step"]
                    bindings[str(path)] = sha(path)
                    asset, _ = add_asset(
                        data,
                        import_step(path),
                        output / (name + "-" + p["name"] + ".stl"),
                    )
                    part = dict(
                        name=p["name"],
                        owner=p["owner"],
                        role=p["role"],
                        color=p["color"],
                        asset=asset,
                        download=name + "-" + p["name"] + ".stl",
                    )
                    if p["replaces_original"]:
                        index = next(
                            i
                            for i, v in enumerate(item["parts"])
                            if v["name"] == p["name"]
                        )
                        item["parts"][index] = part
                    else:
                        item["parts"].append(part)
            item["samples"] = []
            for pose in selected_poses:
                frames = robot.forward_kinematics(
                    pose["pose_rad"], enforce_limits=False
                )
                hits = [h["part"] for row in pose["rows"] for h in row["hits"]]
                sources = [
                    row["part"].replace("_upper_arm_added_stock", "_upper_spine")
                    for row in pose["rows"]
                    if row["status"] == "fail"
                ]
                nearest = [
                    row["nearest"][1]
                    for row in pose["rows"]
                    if row.get("nearest") and row["status"] == "fail"
                ]
                item["samples"].append(
                    dict(
                        name=pose["name"],
                        kind="pose",
                        phase="",
                        status=pose["status"] if enabled else "not_run",
                        clearance_status=pose["status"] if enabled else "not_run",
                        flagged=sorted(set(hits + sources + nearest))
                        if enabled
                        else [],
                        pair=sorted(set(hits + nearest))
                        or ["No failed added-part pairs at this sample"],
                        gap_note=pose.get("note", "")
                        + " New guard, mounting stock and hardware against retained assembly; opposite arm additions. 0.5 mm guard clearance screen. Same-owner hardware qualification, continuous motion and physical fit remain open."
                        if enabled
                        else "Earlier complete robot; no new qualification of its existing pairs.",
                        frames=[frames[o].flatten().tolist() for o in item["owners"]],
                    )
                )
            item.update(
                name=name.title()
                + (" · upper-arm guard proposal" if enabled else " · current assembly"),
                source=str(study) if enabled else str(source),
                inherited_status="fail",
                stock_summary=(
                    f"Separate proposal · {r['polymer_added_g']:.1f} g polymer + {r['nominal_hardware_added_g']:.1f} g hardware by assumed density · {sum(p['role'] == 'screws' for p in r['parts'])} screws + {sum(p['role'] == 'inserts' for p in r['parts'])} inserts · {'recorded self-clearance' if motion_pip else 'bounded geometry'} {checked_status.upper()} · whole robot FAIL · physical NOT RUN"
                    if enabled
                    else baseline["stock_summary"]
                ),
            )
            robots.append(item)
    data["robots"] = robots
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    template = Path("ui/shin-cover.fragment.html")
    html = template.read_text().replace("__HEAD_MOUNT_DATA__", encoded)
    html = html.replace(
        "Printed shin covers · complete robot comparison",
        "Upper-arm guards · separate design proposal",
    )
    html = html.replace("['shin_cover']", "['upper_arm_guard']")
    html = html.replace(
        "../shin-cover-study-v1/report.json", f"../{study.name}/report.json"
    )
    html = html.replace(
        '<a href="../shin-cover-qualification-v1/report.json">Mutual hardware and service checks</a> · ',
        "",
    )
    html = html.replace(
        "../../docs/design/shin-cover-study.md", "../../docs/design/upper-arm-guard.md"
    )
    html = html.replace("Whole-body sampled check", "Static guard and service checks")
    if motion_pip:
        html = html.replace(
            "<details><summary>Scope and source</summary>",
            f'<p>Recorded self-clearance: <a href="../{motion_pip.name}/report.json">Pip</a> · '
            f'<a href="../{motion_scout.name}/report.json">Scout</a>. '
            "A passing guard sample does not change the recorded task outcome.</p>"
            "<details><summary>Scope and source</summary>",
        )
    html = html.replace(
        "Separate candidate front covers with open backs and wheel relief.",
        "Separate upper-arm guards with open backs, elbow openings and trimmed inner wings. Shoulder housings remain exposed.",
    )
    (output / "index.html").write_text(wrapper(html))
    (output / "review.css").write_bytes(Path("ui/product-review.css").read_bytes())
    files = [
        Path(__file__).relative_to(Path.cwd()),
        template,
        Path("ui/product-review.css"),
        Path("tools/build_product_review.py"),
        Path("tools/review_integral_head_service.py"),
        Path("tools/cap_mesh.py"),
    ]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Display identity only; failures retained",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in files}),
            dependency_manifests=dependencies,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            physical_status="not_run",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-pip", type=Path)
    parser.add_argument("--motion-scout", type=Path)
    args = parser.parse_args()
    run(args.study, args.output, args.motion_pip, args.motion_scout)
