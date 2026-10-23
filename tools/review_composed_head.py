"""Inspect complete composed candidates with their separate planning mass evidence."""

import argparse
import base64
import gzip
import json
from pathlib import Path
from build123d import Compound, import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.review_integral_head_service import embedded_dataset, add_asset
from tools.compose_forearm_enclosure import verify_report
from tools.check_camera_support import copy_source, sha, write
from tools.build_product_review import wrapper


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    files = [
        Path(__file__).relative_to(Path.cwd()),
        Path("tools/review_integral_head_service.py"),
        Path("tools/cap_mesh.py"),
        Path("tools/build_product_review.py"),
        Path("ui/composed-head.fragment.html"),
        Path("ui/product-review.css"),
    ]
    report = {
        "status": "fail",
        "scope": __doc__,
        "source_hashes": copy_source(output, {str(p): p for p in files}),
    }
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    try:
        source = Path("artifacts/integral-head-wings-review-v2")
        previous = verify_report(source)
        bind(source / "report.json")
        original_display = verify_report(Path("artifacts/head-display-v7"))
        bind(Path("artifacts/head-display-v7/report.json"))
        data = embedded_dataset(bind(source / "index.html"))
        for candidate in data["robots"]:
            name = candidate["name"].split(" · ")[0].lower()
            folder = Path(f"artifacts/{name}-integral-head-composed-v2")
            r = verify_report(folder)
            bind(folder / "report.json")
            if r["geometry_build_status"] != "pass":
                raise ValueError("Composed geometry failed")
            for suffix in ["assembly.step", "head_shell.step", "head_rear_cover.step"]:
                producer = original_display if suffix == "assembly.step" else previous
                path = next(
                    k
                    for k in producer["source_bindings"]
                    if name in k and k.endswith(suffix)
                )
                if r["source_bindings"][path] != producer["source_bindings"][path]:
                    raise ValueError("Display source differs")
                bind(Path(path))
            for x in [-5, 5]:
                key = f"head_bridge_bolt_{x}"
                shapes = []
                for suffix in ["screw", "nut", "upper_washer", "lower_washer"]:
                    path = Path(
                        f"artifacts/{name}-head-bridge-service-v3/{key}_{suffix}.step"
                    )
                    if sha(path) != r["source_bindings"][str(path)]:
                        raise ValueError("Bridge mesh differs")
                    shapes.extend(import_step(bind(path)).solids())
                part = next(p for p in candidate["parts"] if p["name"] == key)
                part["asset"] = add_asset(
                    data, Compound(shapes), output / f"{name}-{key}.stl"
                )[0]
            for key in ["head_shell", "head_rear_cover"]:
                part = next(p for p in candidate["parts"] if p["name"] == key)
                part.pop("download", None)
            if {p["name"] for p in candidate["parts"]} != {
                p["name"] for p in r["parts"]
            }:
                raise ValueError("Inventory differs")
            owners = {p["name"]: p["owner"] for p in r["parts"]}
            if any(p["owner"] != owners[p["name"]] for p in candidate["parts"]):
                raise ValueError("Part ownership differs")
            mass_folder = Path(f"artifacts/{name}-integral-head-mass-v1")
            mass = verify_report(mass_folder)
            bind(mass_folder / "report.json")
            if mass["source_hashes"][str(folder / "report.json")] != sha(
                folder / "report.json"
            ):
                raise ValueError("Mass assembly identity differs")
            cfg = RobotConfig.model_validate_json(
                bind(folder / "resolved-config.json").read_text()
            )
            fk = make_robot(cfg).forward_kinematics()
            candidate["source"] = str(folder)
            candidate["inherited_status"] = r["status"]
            candidate["name"] = name.capitalize() + " · composed head"
            totals = [
                mass["profiles"][p]["mass_kg"] for p in ["lower", "nominal", "upper"]
            ]
            candidate["stock_summary"] = (
                f"{r['part_count']} CAD groups · STEP identity PASS · nominal changed-stock screen {r['nominal_composition_status'].upper()} · planning mass lower/nominal/upper {totals[0]:.3f}/{totals[1]:.3f}/{totals[2]:.3f} kg · whole assembly {r['status'].upper()}"
            )
            candidate["samples"] = [
                dict(
                    name="neutral_composed_robot",
                    kind="pose",
                    phase="",
                    status=r["nominal_composition_status"],
                    clearance_status=r["nominal_composition_status"],
                    flagged=[],
                    pair=["No added-stock interference at neutral"],
                    gap_note="Neutral composition only. Prior motion/tool studies remain separate; no new task or free-base dynamics executed.",
                    frames=[fk[o].flatten().tolist() for o in candidate["owners"]],
                )
            ]
        encoded = base64.b64encode(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode())
        ).decode()
        fragment = (
            Path("ui/composed-head.fragment.html")
            .read_text()
            .replace("__HEAD_MOUNT_DATA__", encoded)
        )
        (output / "index.html").write_text(wrapper(fragment))
        (output / "review.css").write_text(Path("ui/product-review.css").read_text())
        report.update(
            status="pass",
            source_bindings=bindings,
            scope="Presentation integrity only; complete robot remains failed and physical/dynamics unqualified",
            presentation_files=["index.html", "review.css"],
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        )
    except Exception:
        import traceback

        report["error"] = traceback.format_exc()
        write(output / "report.json", report)
        raise
    write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
