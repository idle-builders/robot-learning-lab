"""Build a standalone review for the forearm-accounted mass replays."""

import argparse
import hashlib
import html
import json
from pathlib import Path
from lab.recording import _source_inputs
from tools.compose_forearm_enclosure import verify_report

NORMAL_BALANCE = {
    "tilt-recovery",
    "reverse-tilt",
    "drive-return",
    "bus-phase-and-poll-gaps",
    "rotated-mount-corrected",
    "noisy-biased-imu",
    "delayed-command",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path, bindings):
    p = Path(path)
    bindings[str(p)] = sha(p)
    return json.loads(p.read_text())


def verify_archive(folder, report, bindings):
    for name, digest in report.get("source_hashes", {}).items():
        p = Path(folder) / "source" / name
        bindings[str(p)] = sha(p)
        if not p.exists() or sha(p) != digest:
            raise ValueError(f"source archive mismatch: {p}")


def verify_episode(folder, report, bindings):
    e = read(Path(folder) / "episode.json", bindings)
    checks = {
        "report.json": "report_sha256",
        "trajectory.jsonl": "trajectory_sha256",
        "resolved-config.json": "robot_config_sha256",
        "scene.xml": "scene_sha256",
    }
    for filename, key in checks.items():
        p = Path(folder) / filename
        if not p.exists() or sha(p) != e[key]:
            raise ValueError(f"episode seal mismatch: {p}")
    if e.get("controller") != report.get("controller"):
        raise ValueError("controller seal mismatch")
    return e


def bind_recording_inputs(folder, kind, bindings):
    inputs, _, _, _, _ = _source_inputs(Path(folder), kind)
    for path in inputs.values():
        bindings[str(path)] = sha(path)


def run(output):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    bindings = {}
    task_rows = []
    balance_rows = []
    cards = []
    balance_links = []
    mass_rows = []
    for name in ("pip", "scout"):
        mass_folder = Path(f"artifacts/{name}-forearm-accounted-mass-v1")
        verify_report(mass_folder)
        read(mass_folder / "report.json", bindings)
        for case in ("lower", "nominal", "upper", "open"):
            profile = "nominal" if case == "open" else case
            folder = Path(f"artifacts/{name}-forearm-accounted-{case}-task-v1")
            report = read(folder / "report.json", bindings)
            verify_archive(folder, report, bindings)
            verify_episode(folder, report, bindings)
            bind_recording_inputs(folder, "task", bindings)
            mass = Path(f"artifacts/{name}-forearm-accounted-mass-v1/{profile}")
            mi = read(mass / "interfaces.json", bindings)
            ti = read(folder / "interfaces.json", bindings)
            if (
                mi != ti
                or report["mass_model"] != mi["mass_model"]
                or report["mass_model"]["sha256"] != sha(mass / "mass-properties.json")
            ):
                raise ValueError(f"mass/interface mismatch: {folder}")
            massprops = read(mass / "mass-properties.json", bindings)
            row = {
                "robot": name,
                "case": case,
                "profile": profile,
                "status": report["status"],
                "termination": report.get("termination"),
                "controller": report["controller"],
                "duration_s": report.get("duration_s"),
                "mass_kg": massprops["total_mass_kg"],
                "folder": str(folder),
            }
            for asset in (folder / "episode.gif", folder / "review.html"):
                bindings[str(asset)] = sha(asset)
            task_rows.append(row)
            cards.append(
                f'<article><h2>{name.title()} · {case}</h2><p class="result {html.escape(report["status"])}">{html.escape(report["status"].upper())} · {html.escape(str(report.get("termination", "")))}</p><p>{html.escape(report["controller"])} controller · {massprops["total_mass_kg"]:.3f} kg · {report.get("duration_s", 0):.2f} s</p><img loading="lazy" src="../{folder.name}/episode.gif" alt="{name} {case} placement replay"><p><a href="../{folder.name}/review.html">Replay review</a> · <a href="../{folder.name}/report.json">Report</a> · <a href="../{folder.name}/episode.json">Episode seal</a></p></article>'
            )
        for profile in ("lower", "nominal", "upper"):
            folder = Path(f"artifacts/{name}-forearm-accounted-{profile}-balance-v1")
            report = read(folder / "balance-report.json", bindings)
            verify_archive(folder, report, bindings)
            mass = Path(f"artifacts/{name}-forearm-accounted-mass-v1/{profile}")
            mi = read(mass / "interfaces.json", bindings)
            bi = read(folder / "interfaces.json", bindings)
            if (
                mi != bi
                or report["mass_model"] != mi["mass_model"]
                or report["mass_model"]["sha256"] != sha(mass / "mass-properties.json")
            ):
                raise ValueError(f"balance mass/interface mismatch: {folder}")
            cases = report["cases"]
            passed = sum(c["status"] == "pass" for c in cases)
            normal = [c for c in cases if c["case"] in NORMAL_BALANCE]
            row = {
                "robot": name,
                "profile": profile,
                "status": report["status"],
                "passed": passed,
                "total": len(cases),
                "normal_passed": sum(c["status"] == "pass" for c in normal),
                "normal_total": len(normal),
                "folder": str(folder),
            }
            for case in cases:
                bind_recording_inputs(folder / case["case"], "balance", bindings)
            bindings[str(folder / "review.html")] = sha(folder / "review.html")
            balance_rows.append(row)
            balance_links.append(
                f'<tr><td>{name.title()} / {profile}</td><td class="{report["status"]}">{report["status"].upper()}</td><td>{passed}/{len(cases)}</td><td>{row["normal_passed"]}/{len(normal) if normal else 0}</td><td><a href="../{folder.name}/review.html">Suite review</a> · <a href="../{folder.name}/balance-report.json">Report</a></td></tr>'
            )
    for name in ("pip", "scout"):
        for profile in ("lower", "nominal", "upper"):
            p = Path(
                f"artifacts/{name}-forearm-accounted-mass-v1/{profile}/mass-properties.json"
            )
            d = read(p, bindings)
            mass_rows.append(
                {"robot": name, "profile": profile, "mass_kg": d["total_mass_kg"]}
            )
    for linked in (
        Path("docs/design/forearm-mass-accounting.md"),
        Path("artifacts/forearm-accounted-task-launch-v1/report.json"),
        Path("artifacts/forearm-accounted-balance-launch-v1/report.json"),
    ):
        bindings[str(linked)] = sha(linked)
    task_pass = sum(r["status"] == "pass" for r in task_rows if r["case"] != "open")
    content = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forearm-accounted review</title><style>body{font:16px/1.55 system-ui;background:#f4f2e9;color:#243e36;max-width:1250px;margin:auto;padding:36px}h1{font-size:42px;line-height:1.1}h2{margin-top:0}a{color:#216c65}.scope{border-left:4px solid #ad512f;padding:16px 20px;background:#fff8e9}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:20px;margin:22px 0}article{padding:20px;border:1px solid #cad3c7;border-radius:16px;background:#fffef8}.result{font-weight:700}.pass{color:#216c65}.fail{color:#ad512f}img{width:100%;border-radius:8px}table{width:100%;border-collapse:collapse;background:#fffef8;margin:18px 0}td,th{padding:11px;border-bottom:1px solid #d9ded8;text-align:left}.small{font-size:14px;color:#53665e}</style><header><p>ROBOT LEARNING LAB / MASS ACCOUNTING</p><h1>Forearm reserve, replayed.</h1>"""
    content += (
        f'<p>{task_pass}/6 feedback placement trials pass; open controls remain visible as negative controls. The page covers the accounting-only mass change and its retained reference simulations.</p><p class="scope">The derived ledger removes the duplicate forearm owners from <code>arm_covers_and_wire</code> and adds the unmeasured 3/7/12 g residual harness allowance per forearm. Bilateral accounting deltas are −18/−32/−56 g. These are not physical weight savings. Source assembly failures, physical qualification limits and unmeasured actuator/mass assumptions remain.</p><p><a href="../../docs/design/forearm-mass-accounting.md">Accounting note</a> · <a href="report.json">Integrity report</a> · <a href="../forearm-accounted-task-launch-v1/report.json">Task launcher</a> · <a href="../forearm-accounted-balance-launch-v1/report.json">Balance launcher</a></p></header><main><h2>Eight placement replays</h2><section class="cards">'
        ""
        + "".join(cards)
        + f"""</section><h2>Six observed-balance suites</h2><table><thead><tr><th>Suite</th><th>Result</th><th>All cases</th><th>Normal cases</th><th>Evidence</th></tr></thead><tbody>{"".join(balance_links)}</tbody></table><p class="small">Nominal suites report aggregate failure because deliberate negative controls remain in the 16-case suite; lower and upper bound suites contain three cases each. Review links open the producer artifacts.</p><h2>Mass outputs</h2><table><thead><tr><th>Robot</th><th>Lower</th><th>Nominal</th><th>Upper</th></tr></thead><tbody>"""
    )
    for name in ("pip", "scout"):
        vals = [r["mass_kg"] for r in mass_rows if r["robot"] == name]
        content += (
            f"<tr><td>{name.title()}</td>"
            + "".join(f"<td>{v:.6f} kg</td>" for v in vals)
            + "</tr>"
        )
    content += "</tbody></table></main></html>"
    (output / "index.html").write_text(content)
    (output / "builder.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "status": "pass",
        "scope": "Presentation integrity and source/output identity review; accounting and reference simulations remain separate",
        "task_rows": task_rows,
        "balance_rows": balance_rows,
        "mass_rows": mass_rows,
        "checks": [
            "task episode seals",
            "task and balance source archives",
            "mass/interface identity",
            "accounting delta retained",
            "physical qualification not_run",
        ],
        "source_bindings": bindings,
    }
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    print(run(p.parse_args().output)["status"])
