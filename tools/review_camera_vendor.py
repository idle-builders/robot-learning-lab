"""Compare sealed vendor-camera body studies and their matching workbench views."""

import argparse
import hashlib
import html
import json
import os
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.review_camera_mount import verify_study
from tools.review_camera_support import render


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equivalent(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-7)
    return a == b


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    rows, sections, bindings = [], [], {}
    for name, suffix in [("pip", "v3"), ("scout", "v2")]:
        body = Path(
            f"artifacts/{name}-camera-vendor-" + ("v3" if name == "pip" else "v4")
        )
        workbench = Path(f"artifacts/{name}-camera-vendor-workbench-{suffix}")
        r = verify_study(body)
        w = json.loads((workbench / "report.json").read_text())
        baseline = Path(f"artifacts/{name}-camera-vendor-v2")
        reference = verify_study(baseline)
        for key in [
            "support_self_interference",
            "installed_conflicts",
            "insertion",
            "cassette_removal",
            "driver_access",
            "motion",
            "brow_clearance",
            "contact_stack",
            "contact_witnesses",
            "tool_witnesses",
            "aperture_envelope",
            "face_layout",
        ]:
            if not equivalent(r[key], reference[key]):
                raise ValueError(
                    f"optimized checker differs from original: {name}/{key}"
                )
        bindings[str(baseline / "report.json")] = sha(baseline / "report.json")
        cfg = RobotConfig.model_validate_json(
            (body / "resolved-config.json").read_text()
        )
        if cfg != RobotConfig.model_validate_json(
            (workbench / "resolved-config.json").read_text()
        ):
            raise ValueError("body/workbench configuration mismatch")
        if sha(workbench / "resolved-config.json") != w["resolved_config_sha256"]:
            raise ValueError("workbench configuration attestation changed")
        for key, digest in w["source_hashes"].items():
            if sha(workbench / "source" / key) != digest:
                raise ValueError(f"workbench source changed: {key}")
        for key in [
            "lab/config.py",
            "lab/robot.py",
            "lab/styling.py",
            "lab/mechanical.py",
            "lab/camera_mount.py",
            "lab/camera_support.py",
            "lab/camera_vendor.py",
            "lab/packaging.py",
            "electronics/camera-support-vendor.yaml",
        ]:
            if (
                r["source_hashes"][key] != w["source_hashes"][key]
                or sha(key) != r["source_hashes"][key]
            ):
                raise ValueError(f"body/workbench/current recipe mismatch: {key}")
        robot = make_robot(cfg)
        render(body / "assembled", output / f"{name}-head.png", robot, 145)
        render(body / "assembled", output / f"{name}-front.png", robot, 180)
        shutil.copyfile(workbench / "robot.png", output / f"{name}-body.png")
        clearance = r.get("brow_clearance", [])
        row = dict(
            robot=name,
            mechanical_status=r["status"],
            workbench_status=w["status"],
            workbench_failures=[
                k for k, v in w["checks"].items() if v["status"] == "fail"
            ],
            camera_profile=r["interface"]["source"]["profile"],
            contact_stack=r["contact_stack"]["status"],
            tool_witnesses=r["tool_witnesses"]["status"],
            annular_bearing=r["annular_bearing_witnesses"]["status"],
            original_checker_equivalence="pass",
            motion_conflicts=sum(len(p["conflicts"]) for p in r["motion"]),
            brow_minimum_gap_mm=min(
                (p["minimum_gap_mm"] for p in clearance), default=None
            ),
            not_run=r["not_run"],
        )
        rows.append(row)
        for folder in [body, workbench]:
            bindings[str(folder / "report.json")] = sha(folder / "report.json")
        rel = os.path.relpath(body, output)
        gap = (
            f"; minimum sampled brow/body gap {row['brow_minimum_gap_mm']:.2f} mm"
            if clearance
            else ""
        )
        sections.append(
            f'''<section><h2>{name.title()}</h2><p>Nominal camera/body study: <b>{r["status"]}</b>. {row["motion_conflicts"]} reported intersections over 15 neck poses{gap}.</p><div class="images"><img src="{name}-front.png" alt="Front head view"><img src="{name}-head.png" alt="Three-quarter head view"><img src="{name}-body.png" alt="Whole robot envelope view"></div><p>Contact stack: {row["contact_stack"]}; full bearing rings: {row["annular_bearing"]}; tool witnesses: {row["tool_witnesses"]}. Workbench failures: {html.escape(", ".join(row["workbench_failures"]))}.</p><p><a href="{rel}/camera-support.step">Populated camera/head STEP</a> · <a href="{rel}/report.json">Mechanical findings</a> · <a href="{os.path.relpath(workbench, output)}/review.html">Workbench report</a></p></section>'''
        )
    for negative in [
        "artifacts/pip-camera-vendor-v1",
        "artifacts/camera-vendor-fit-v4",
        "artifacts/scout-camera-support-v3",
    ]:
        old = verify_study(Path(negative))
        if old["status"] != "fail":
            raise ValueError("retained negative study unexpectedly passed")
        bindings[negative + "/report.json"] = sha(Path(negative) / "report.json")
    report = dict(
        status="pass"
        if all(x["mechanical_status"] == "pass" for x in rows)
        else "fail",
        scope="bounded vendor-camera fit, service and sampled brow/body clearance; whole product incomplete",
        robots=rows,
        source_bindings=bindings,
        open=[
            "physical module revision and clamp/tolerance qualification",
            "Scout visor attachment and shell interference",
            "Pi retention and camera flex routing",
            "new mass/dynamics",
            "unified full product BOM and price",
        ],
    )
    (output / "review.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>Vendor camera integration</title><style>body{font:17px/1.5 system-ui;background:#e9eeea;color:#1e3631;max-width:1250px;margin:32px auto;padding:0 24px}h1,h2{line-height:1.15}.images{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}img{width:100%;border-radius:12px;align-self:center}section{padding:24px;margin:24px 0;background:#fff;border-radius:16px}a{color:#186452}.open{padding:18px;background:#fff0da;border-radius:12px}</style><h1>One purchased camera model, two robot candidates</h1><p>The full 631-solid manufacturer model now flows into shared CAD and visual meshes. Printed supports and screw seats use the substrate surfaces at the mounting holes. The original drawing and copper-plane interpretations remain as failed evidence.</p><p class="open">These are engineering candidates. The visor still intersects its shell and lacks a physical attachment. Eye retention, flex/Pi mounting, upper-neck enclosure, measured camera revision, mass/dynamics and full production price remain open. The unchanged legacy power-coverage check still fails.</p>"""
        + "".join(sections)
        + """<section><h2>What the failed attempts taught us</h2><p>The drawing-based posts intersected the vendor substrate by about 8.32 mm³. The next, copper-plane stack cleared the model but left 0.03048 mm between the substrate and each clamping face. The selected substrate stack removes those nominal gaps without intersecting the populated model. This does not resolve tolerances or physical revision.</p><p>Scout's original solid brow produced 40 reported hardware intersections. The forward visor preserves its front edge and top height while removing the rear sweep.</p><p><a href="../pip-camera-vendor-v1/report.json">Failed copper-plane clamping</a> · <a href="../camera-vendor-fit-v4/report.json">Failed drawing stack</a> · <a href="../scout-camera-support-v3/report.json">Original brow interference</a> · <a href="../../docs/design/camera-vendor.md">Design details</a></p></section><p><a href="report.json">Review provenance and limits</a></p>"""
    )
    (output / "source").mkdir()
    shutil.copyfile(__file__, output / "source/review_camera_vendor.py")
    shutil.copyfile(
        "tools/review_camera_support.py", output / "source/review_camera_support.py"
    )
    report["source_hashes"] = {
        str(p.relative_to(output / "source")): sha(p)
        for p in (output / "source").rglob("*")
        if p.is_file()
    }
    report["test_logs"] = {}
    for name, path in [
        ("regression-tests.log", "/tmp/camera-vendor-regressions-v2.log"),
        ("nearest-clearance-test.log", "/tmp/camera-clearance-regression-v2.log"),
    ]:
        shutil.copyfile(path, output / name)
        report["test_logs"][name] = (output / name).read_text().strip().splitlines()[-1]
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
