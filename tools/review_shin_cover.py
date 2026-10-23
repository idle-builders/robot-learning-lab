"""Compare original complete robots with source-bound shin-cover additions."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
from build123d import import_step

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_integral_head_service import add_asset, embedded_dataset


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    study = Path("artifacts/shin-cover-study-v1")
    report = verify_report(study)
    qualification_folder = Path("artifacts/shin-cover-qualification-v1")
    qualification = verify_report(qualification_folder)
    original = Path("artifacts/integral-head-composed-review-v2")
    source = verify_report(original)
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    bind(study / "report.json")
    bind(qualification_folder / "report.json")
    if qualification["source_bindings"][str(study / "report.json")] != sha(
        study / "report.json"
    ):
        raise ValueError("Supplemental geometry study differs")
    bind(original / "report.json")
    data = embedded_dataset(bind(original / "index.html"))
    candidates = []
    for baseline in data["robots"]:
        name = baseline["name"].split(" · ")[0].lower()
        assembly = Path(f"artifacts/{name}-integral-head-composed-v2")
        if (
            source["source_bindings"][str(assembly / "report.json")]
            != report["source_bindings"][str(assembly / "report.json")]
        ):
            raise ValueError("Displayed and checked source robots differ")
        robot = make_robot(
            RobotConfig.model_validate_json(
                bind(assembly / "resolved-config.json").read_text()
            )
        )
        result = report["robots"][name]
        for enabled in (False, True):
            item = deepcopy(baseline)
            for p in item["parts"]:
                p.pop("download", None)
            if enabled:
                for p in result["parts"]:
                    asset, _ = add_asset(
                        data,
                        import_step(bind(study / p["step"])),
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
                        idx = next(
                            i
                            for i, x in enumerate(item["parts"])
                            if x["name"] == p["name"]
                        )
                        item["parts"][idx] = part
                    else:
                        item["parts"].append(part)
            item["samples"] = []
            for pose in [*result["poses"], *qualification["robots"][name]["poses"]]:
                frames = robot.forward_kinematics(pose["pose_rad"])
                hitnames = sorted(
                    {h["part"] for row in pose["rows"] for h in row["hits"]}
                )
                sources = sorted(
                    {
                        row["part"].replace("_added_spine_stock", "_spine")
                        for row in pose["rows"]
                        if row["hits"]
                    }
                )
                mutual = [
                    row
                    for row in qualification["robots"][name]["mutual"]
                    if row["pose"] == pose["name"] and row["hits"]
                ]
                hitnames += [row["part_b"] for row in mutual]
                sources += [row["part_a"] for row in mutual]
                pose_status = "fail" if hitnames else "pass"
                item["samples"].append(
                    dict(
                        name=pose["name"],
                        kind="pose",
                        phase="",
                        status=pose_status if enabled else "not_run",
                        clearance_status=pose_status if enabled else "not_run",
                        flagged=hitnames + sources if enabled else [],
                        pair=hitnames or ["No added-part overlap at this sample"],
                        gap_note=(
                            "Added parts versus retained assembly and mutual candidate pairs; intentional heat-insert/pilot interference excluded. No continuous or physical qualification."
                            if enabled
                            else "Original assembly reference. No new full-robot qualification."
                        ),
                        frames=[frames[o].flatten().tolist() for o in item["owners"]],
                    )
                )
            item.update(
                name=f"{name.title()} · "
                + ("round shin covers" if name == "pip" else "square shin covers")
                if enabled
                else f"{name.title()} · original",
                source=str(study) if enabled else str(assembly),
                inherited_status="fail",
                stock_summary=(
                    f"Separate uninstalled proposal · {result['polymer_added_g']:.1f} g polymer + {result['nominal_hardware_added_g']:.1f} g hardware by assumed densities · 8 screws + 8 inserts · supplemental checks {qualification['robots'][name]['status'].upper()} · whole robot FAIL"
                    if enabled
                    else "Original retained assembly · whole robot FAIL · physical NOT RUN"
                ),
            )
            candidates.append(item)
    data["robots"] = candidates
    encoded = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode())
    ).decode()
    template = Path("ui/shin-cover.fragment.html")
    (output / "index.html").write_text(
        wrapper(template.read_text().replace("__HEAD_MOUNT_DATA__", encoded))
    )
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
            scope="Display/source identity only; original and candidate whole-robot failures retained",
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in files}),
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file()
            },
            physical_status="not_run",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
