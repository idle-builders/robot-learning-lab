"""Inspect a sealed visor study alongside its matching ordinary workbench."""

import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
from PIL import Image

from lab.config import RobotConfig
from tools.check_camera_support import copy_source, sha, write
from tools.review_camera_mount import verify_study


def run(study, workbench, tests, output):
    output.mkdir(parents=True, exist_ok=False)
    r = verify_study(study)
    if r["status"] != "pass":
        raise ValueError("visor study did not pass")
    w = json.loads((workbench / "report.json").read_text())
    if w.get("resolved_config_sha256") != sha(workbench / "resolved-config.json"):
        raise ValueError("workbench resolved configuration changed")
    for key, digest in w["source_hashes"].items():
        if sha(workbench / "source" / key) != digest:
            raise ValueError(f"workbench source changed: {key}")
    a = RobotConfig.model_validate_json((study / "resolved-config.json").read_text())
    b = RobotConfig.model_validate_json(
        (workbench / "resolved-config.json").read_text()
    )
    if a != b:
        raise ValueError("workbench and mechanical configurations differ")
    bindings = {
        str(study / p): sha(study / p)
        for p in ["report.json", "head.step", "resolved-config.json"]
    }
    for p in ["report.json", "resolved-config.json", "robot.png"]:
        bindings[str(workbench / p)] = sha(workbench / p)
    for key, digest in r["source_hashes"].items():
        if key.startswith("lab/") and w["source_hashes"].get(key) != digest:
            raise ValueError(f"workbench recipe mismatch: {key}")
        if (
            key.startswith(("lab/", "tools/"))
            and Path(key).is_file()
            and sha(key) != digest
        ):
            raise ValueError(f"current recipe differs from study: {key}")
    source_paths = {
        "study/report.json": study / "report.json",
        "workbench/report.json": workbench / "report.json",
        "tools/review_visor_mount.py": Path("tools/review_visor_mount.py"),
    }
    source_paths.update({f"tests/{p.name}": p for p in tests.glob("*.log")})
    source_hashes = copy_source(output, source_paths)
    for mode in ["assembled", "exploded"]:
        model = mujoco.MjModel.from_xml_path(
            str((study / mode / "mechanism.xml").absolute())
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0.01, 0, 0.015]
        camera.distance = 0.31
        camera.azimuth = 155
        camera.elevation = -25
        with mujoco.Renderer(model, height=800, width=960) as renderer:
            renderer.update_scene(data, camera=camera)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            Image.fromarray(renderer.render()).save(output / f"{mode}.png")
        for p in (study / mode).iterdir():
            if p.is_file():
                bindings[str(p)] = sha(p)
    shutil.copy2(workbench / "robot.png", output / "robot.png")
    reduction = 1 - r["changes"]["visor_mm3"] / r["changes"]["old_solid_visor_mm3"]
    checks = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v['status'])}</td></tr>"
        for k, v in w["checks"].items()
    )
    gaps = "".join(f"<li>{html.escape(s)}</li>" for s in r["not_run"])
    failed = [name for name, check in w["checks"].items() if check["status"] == "fail"]
    workbench_detail = (
        ("Failed checks: " + ", ".join(failed))
        if failed
        else "No failed workbench checks"
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>Scout · removable visor</title>
<style>body{{font:17px system-ui;background:#f5f3ed;color:#233238;max-width:1080px;margin:40px auto;padding:0 24px}}h1{{font-size:36px}}.views{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{width:100%;border-radius:16px}}figure{{margin:0}}figcaption{{padding:10px}}table{{border-collapse:collapse;width:100%}}td{{padding:7px;border-bottom:1px solid #ccd2ce}}a{{color:#1b6571}}.status{{padding:16px;background:#dce8dc;border-radius:12px}}@media(max-width:700px){{.views{{grid-template-columns:1fr}}}}</style>
<h1>Scout · a removable visor</h1><p>Nominal attachment study. Two recessed M2.5 screws, insert bosses and an open-bottom cap keep the forward silhouette.</p>
<p class="status">Mechanical checks: <b>{r["status"]}</b>. Whole product: incomplete. Ordinary workbench: <b>{w["status"]}</b>. {html.escape(workbench_detail)}.</p>
<div class="views"><figure><img src="assembled.png"><figcaption>Assembled head · actual CAD meshes</figcaption></figure><figure><img src="exploded.png"><figcaption>Remove screws, then lift the visor</figcaption></figure></div>
<p>Visor plastic volume: {r["changes"]["visor_mm3"] / 1000:.2f} cm³; {reduction:.1%} below the former solid allocation. Added shell material: {r["changes"]["added_shell_mm3"] / 1000:.2f} cm³. These volumes do not update legacy physics mass automatically.</p>
<p>Both mating faces and screw bearing seats have full nominal witness coverage. {len(r["removal"])} visor-removal positions, {len(r["screw_removal"])} screw-removal positions and {len(r["body_motion"])} neck poses have no detected interference. Deliberately displaced visor and tool controls fail as intended. These are sampled nominal checks; insert/plastic strength remains unqualified.</p>
<p><a href="../{study.name}/head.step">Head STEP</a> · <a href="../{study.name}/report.json">Mechanical evidence</a> · <a href="../{workbench.name}/review.html">Workbench</a> · <a href="../../docs/design/visor-mount.md">Interface, cost and limits</a></p>
<details><summary>Ordinary workbench checks</summary><table>{checks}</table></details>
<details><summary>Whole robot appearance</summary><img src="robot.png"><p>Shared visual model; allocated dynamics. The detailed body check uses the separately retained 162-group mechanism baseline.</p></details>
<h2>Still open</h2><ul>{gaps}</ul><p>Camera cable routing must use the host CSI connector. The vendor model's named FPC is internal sensor flex. Pi mounting requires a service opening; the camera aperture is too small for board removal.</p>
"""
    (output / "review.html").write_text(page)
    result = dict(
        status="pass",
        scope=__doc__,
        mechanical_status=r["status"],
        workbench_status=w["status"],
        product_status="incomplete",
        source_hashes=source_hashes,
        source_bindings=bindings,
    )
    result["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file() and "source" not in p.relative_to(output).parts
    }
    write(output / "report.json", result)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["study", "workbench", "tests", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.workbench, args.tests, args.output)
