"""Present sealed Pip/Scout head-service studies beside their workbenches."""

import argparse
import hashlib
import html
import json
import os
import shutil
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.artifact_store import snapshot_file, retain_dependency
from lab.config import RobotConfig
from tools.review_camera_mount import verify_study


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    snapshot_file(src, dst)


def verify_workbench(folder):
    report = json.loads((folder / "report.json").read_text())
    if report.get("resolved_config_sha256") != sha(folder / "resolved-config.json"):
        raise ValueError(f"workbench configuration changed: {folder}")
    for name, digest in report.get("source_hashes", {}).items():
        if sha(folder / "source" / name) != digest:
            raise ValueError(f"workbench source changed: {name}")
    return report


def normalized_config(path):
    return RobotConfig.model_validate_json(Path(path).read_text()).model_dump(
        mode="json"
    )


def conflict_count(report):
    keys = [
        "head_conflicts",
        "new_pair_conflicts",
        "screw_removal",
        "tool_withdrawal",
        "cap_removal",
        "pi_removal",
        "tools_body",
        "tools",
        "service_body",
        "body_motion",
    ]
    count = 0
    for key in keys:
        for row in report.get(key, []):
            if row.get("hits"):
                count += len(row["hits"])
            elif key in {"head_conflicts", "new_pair_conflicts"} and row:
                count += 1
    return count


def failed_workbench_checks(report):
    return [
        name
        for name, row in report.get("checks", {}).items()
        if row.get("status") == "fail"
    ]


def run(pip, scout, pip_workbench, scout_workbench, tests, output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    source_bindings = {}
    dependency_manifests = {}
    drift = {}
    reviewer_copy = output / "source" / "tools" / "review_head_service.py"
    copy_file(Path(__file__), reviewer_copy)
    for label, study, workbench in [
        ("Pip", pip, pip_workbench),
        ("Scout", scout, scout_workbench),
    ]:
        sealed = verify_study(study)
        work = verify_workbench(workbench)
        for name, digest in sealed.get("source_hashes", {}).items():
            if (
                name.startswith("lab/")
                and work.get("source_hashes", {}).get(name) != digest
            ):
                raise ValueError(f"workbench recipe mismatch: {label} {name}")
        study_cfg = normalized_config(study / "resolved-config.json")
        work_cfg = normalized_config(workbench / "resolved-config.json")
        if study_cfg != work_cfg:
            raise ValueError(f"study/workbench configuration differs: {label}")
        current_drift = []
        for name, digest in sealed.get("source_hashes", {}).items():
            if name.startswith(("lab/", "tools/")):
                if not Path(name).is_file() or sha(name) != digest:
                    current_drift.append(name)
        drift[label] = current_drift
        for name in ["report.json", "resolved-config.json"]:
            src = study / name
            dest = output / "source" / label.lower() / "study" / name
            copy_file(src, dest)
        for root_name, root in (("study", study), ("workbench", workbench)):
            dependency_manifests[f"{label.lower()}/{root_name}"] = retain_dependency(root)
        for name in ["report.json", "resolved-config.json"]:
            src = workbench / name
            dest = output / "source" / label.lower() / "workbench" / name
            copy_file(src, dest)
        modes = ["assembled", "exploded"]
        if (study / "carrier").exists():
            modes.append("carrier")
        for mode in modes:
            src = study / mode / "mechanism.png"
            dest = output / f"{label.lower()}-{mode}.png"
            copy_file(src, dest)
        src = workbench / "robot.png"
        dest = output / f"{label.lower()}-robot.png"
        copy_file(src, dest)
        face = sealed.get("face_layout", {})
        pi = sealed.get("interface", {}).get("pi_carrier", {})
        rows.append(
            {
                "robot": label,
                "mechanical_status": sealed.get("status", "fail"),
                "workbench_status": work.get("status", "fail"),
                "board_placement_head_mm": pi.get("position_head_mm"),
                "board_orientation": pi.get("orientation"),
                "new_stock_cm3": sealed.get("new_stock_mm3", 0) / 1000,
                "failed_geometry_rows": conflict_count(sealed),
                "failed_bearing_rows": sum(
                    row.get("status") == "fail" for row in sealed.get("bearing", [])
                ),
                "failed_tool_controls": sum(
                    bool(row.get("hits")) or row.get("negative_mm3", 1) <= 0.02
                    for row in sealed.get("tools", [])
                ),
                "cap_samples": len(sealed.get("cap_removal", [])),
                "pi_samples": len(sealed.get("pi_removal", [])),
                "neck_pose_samples": len(sealed.get("body_motion", [])),
                "face_status": face.get("status", "not_run"),
                "face_note": "No new face-layout optical validation in this study."
                if not face
                else "Face-layout result retained from the sealed study.",
                "workbench_failures": failed_workbench_checks(work),
                "study": str(study),
                "workbench": str(workbench),
            }
        )
        for root in (study, workbench):
            for name in ("report.json", "resolved-config.json"):
                source_bindings[str(root / name)] = sha(root / name)
        source_bindings[str(study / "head.step")] = sha(study / "head.step")
    if tests.exists():
        for log in sorted(tests.glob("*.log")):
            dest = output / "source" / "tests" / log.name
            copy_file(log, dest)
    page_rows = []
    for row in rows:
        failures = ", ".join(row["workbench_failures"]) or "none"
        study_path = pip if row["robot"] == "Pip" else scout
        workbench_path = pip_workbench if row["robot"] == "Pip" else scout_workbench

        def link(path):
            return html.escape(os.path.relpath(path, output))

        page_rows.append(f"""
        <section><h2>{html.escape(row["robot"])}</h2>
        <p class=status>Mechanical study: <b>{html.escape(row["mechanical_status"])}</b> · Workbench: <b>{html.escape(row["workbench_status"])}</b></p>
        <div class=views><figure><img src="{row["robot"].lower()}-assembled.png"><figcaption>Assembled head</figcaption></figure><figure><img src="{row["robot"].lower()}-exploded.png"><figcaption>Exploded service view</figcaption></figure>{f'<figure><img src="{row["robot"].lower()}-carrier.png"><figcaption>Carrier and Pi detail</figcaption></figure>' if ((pip if row["robot"] == "Pip" else scout) / "carrier").exists() else ""}</div>
        <p>Pi placement: {html.escape(str(row["board_placement_head_mm"]))} mm head frame ({html.escape(str(row["board_orientation"]))}). New head stock: {row["new_stock_cm3"]:.2f} cm³. Geometry/control conflict rows: {row["failed_geometry_rows"]}; failed bearing rows: {row["failed_bearing_rows"]}; failed tool controls: {row["failed_tool_controls"]}.</p>
        <p>Service sampling: {row["cap_samples"]} cap-removal rows, {row["pi_samples"]} Pi-removal rows, {row["neck_pose_samples"]} neck poses. Workbench failed checks: {html.escape(failures)}. {html.escape(row["face_note"])}</p>
        <p><a href="{link(study_path / "report.json")}">Study report</a> · <a href="{link(study_path / "head.step")}">Head STEP</a> · <a href="{link(workbench_path / "review.html")}">Workbench review</a> · <a href="{link(Path("docs/design/head-service.md"))}">Head-service limits</a></p>
        <details><summary>Whole-robot workbench view</summary><img src="{row["robot"].lower()}-robot.png"></details></section>""")
    page = f"""<!doctype html><meta charset=utf-8><title>Head service review</title>
    <style>body{{font:16px system-ui;background:#f4f2eb;color:#243238;max-width:1120px;margin:34px auto;padding:0 22px}}h1{{font-size:36px}}section{{background:#fff;padding:22px;border-radius:16px;margin:24px 0}}.status{{padding:12px;background:#e8eee7;border-radius:10px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{max-width:100%;border-radius:12px}}figure{{margin:0}}figcaption{{padding:7px}}@media(max-width:700px){{.views{{grid-template-columns:1fr}}}}</style>
    <h1>Removable head service and Pi carrier</h1>
    <p>Pip and Scout rear-cover candidates. The whole product remains incomplete; these nominal checks do not establish physical retention or durability.</p>
    {"".join(page_rows)}
    <h2>Review boundaries</h2><p>Pi connector, antenna, cable, thermal, material, tolerance, retention, mass, dynamics and production-price work remain open. The images use the study's actual CAD meshes and are evidence for inspection, not manufacturing approval.</p>
    <p>Current recipe drift is recorded separately: {html.escape(json.dumps(drift, sort_keys=True))}.</p>"""
    (output / "review.html").write_text(page)
    result = {
        "status": "pass",
        "scope": __doc__,
        "product_status": "incomplete",
        "robots": rows,
        "current_source_drift": drift,
        "source_hashes": {
            str(p.relative_to(output / "source")): sha(p)
            for p in (output / "source").rglob("*")
            if p.is_file()
        },
        "source_bindings": source_bindings,
        "dependency_manifests": dependency_manifests,
    }
    result["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file()
        and "source" not in p.relative_to(output).parts
        and p.name not in {"report.json", "report.sha256"}
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pip", "scout", "pip-workbench", "scout-workbench", "tests", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.pip,
        args.scout,
        args.pip_workbench,
        args.scout_workbench,
        args.tests,
        args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
