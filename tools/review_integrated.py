"""Review an integrated assembly, planning mass, and its matching trials."""

import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("assembly", "mass", "balance", "task", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def read(folder, name):
        path = folder / name
        bindings[str(path)] = sha(path)
        return json.loads(path.read_text())

    assembly = read(args.assembly, "report.json")
    mass = read(args.mass, "report.json")
    balance = read(args.balance, "balance-report.json")
    task = read(args.task, "report.json")
    if assembly["status"] != "pass" or mass["status"] != "pass":
        raise ValueError("requires passed assembly/mass producers")
    for folder, report in (
        (args.assembly, assembly),
        (args.mass, mass),
        (args.balance, balance),
        (args.task, task),
    ):
        for name, digest in report["source_hashes"].items():
            if sha(folder / "source" / name) != digest:
                raise ValueError(f"retained source changed: {folder}/{name}")
        for name, digest in report.get("output_hashes", {}).items():
            if sha(folder / name) != digest:
                raise ValueError(f"output changed: {folder}/{name}")
    nominal_path = args.mass / "nominal/mass-properties.json"
    identity = sha(nominal_path)
    bindings[str(nominal_path)] = identity
    if any(report["mass_model"]["sha256"] != identity for report in (task, balance)):
        raise ValueError("trial mass identity mismatch")
    if (
        sha(args.assembly / "assembly.step")
        != mass["source_hashes"][str(args.assembly / "assembly.step")]
    ):
        raise ValueError("mass refers to another assembly")

    def link(path, text, *, evidence=True):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"missing linked artifact: {path}")
        if evidence:
            bindings[str(path)] = sha(path)
        return (
            f'<a href="{escape(os.path.relpath(path, args.output))}">{escape(text)}</a>'
        )

    def image(path, alt):
        link(path, alt)
        return f'<img src="{escape(os.path.relpath(path, args.output))}" alt="{escape(alt)}">'

    mass_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{row['mass_kg']:.3f} kg</td></tr>"
        for name, row in mass["profiles"].items()
    )
    case_rows = []
    tracking_names = {
        "tilt-recovery",
        "reverse-tilt",
        "drive-return",
        "bus-phase-and-poll-gaps",
        "rotated-mount-corrected",
        "noisy-biased-imu",
        "delayed-command",
    }
    tracking = [c for c in balance["cases"] if c["case"] in tracking_names]
    tracking_passes = sum(c["status"] == "pass" for c in tracking)
    for case in balance["cases"]:
        name = case["case"]
        saved = read(args.balance / name, "result.json")
        if saved != case:
            raise ValueError(f"case summary changed: {name}")
        final = case.get("final_position_error_m")
        path = case.get("max_path_error_m")

        def value(x):
            return "—" if x is None else f"{x * 1000:.1f} mm"

        case_rows.append(
            f'<tr><td>{escape(name)}</td><td class="{case["status"]}">{case["status"]}</td><td>{value(final)}</td><td>{value(path)}</td></tr>'
        )
    name = escape(task["robot"])
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{name} integration review</title>
<style>body{{margin:0;background:#edf0eb;color:#203934;font:16px/1.5 system-ui}}main{{max-width:1120px;margin:auto;padding:36px}}h1{{font-size:34px}}h2{{margin-top:36px}}.gallery{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}img{{width:100%;border-radius:12px}}section{{background:white;border-radius:16px;padding:24px;margin:24px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#206b61}}.fail{{color:#a34125}}.pass{{color:#287245}}.tag{{background:#e6ead9;padding:5px 10px;border-radius:20px}}@media(max-width:700px){{.gallery{{grid-template-columns:1fr}}main{{padding:16px}}}}</style>
<main><span class="tag">Engineering candidate · product incomplete</span><h1>{name}: integrated head, body and power board</h1>
<p>{assembly["part_count"]} named groups. New board/body/pack and sampled head-motion checks pass. Actual Pi ports, harness, face retention, protective covers and strength remain unfinished. The battery is a separate mass allocation.</p>
<div class="gallery"><div>{image(args.assembly / "mechanism.png", "Full engineering assembly")}<p>Detailed assembly; exposed mechanisms still need covers.</p></div><div>{image(args.assembly / "cutaway/mechanism.png", "Cutaway engineering assembly")}<p>Shells hidden for component inspection.</p></div></div>
<section><h2>Planning mass</h2><table><tr><th>Assumption profile</th><th>Total</th></tr>{mass_rows}</table><p>Materials and component distributions are unmeasured. Nominal is {mass["profiles"]["nominal"]["mass_kg"] / 3 * 100:.1f}% of the provisional 3 kg target. This is not a production-weight guarantee.</p>{link(args.mass / "report.json", "Mass ledger and full tensor evidence")}</section>
<section><h2>Supported placement: {task["status"]}</h2><p>Episode duration {task["duration_s"]:.2f} s; maximum lift {task["max_lift_m"] * 1000:.1f} mm. Fixed torso, privileged object state and a reference controller; no learned-policy or free-base manipulation claim.</p>{image(args.task / "final.png", "Final supported placement")}{link(args.task / "report.json", "Task report")}</section>
<section><h2>Balance with the recorded controller</h2><p>Static pitch reference: {balance.get("pitch_trim", {}).get("pitch_rad", 0) * 180 / 3.141592653589793:.3f}° ({escape(balance.get("pitch_trim", {}).get("mode", "zero"))}). Assumes the recorded zero-joint posture.</p><p>{tracking_passes}/{len(tracking)} standard tracking cases pass. Fault and disabled-control outcomes are also retained below; inspect their individual termination reasons.</p><table><tr><th>Case</th><th>Outcome</th><th>Final error</th><th>Peak path error</th></tr>{"".join(case_rows)}</table>{link(args.balance / "review.html", "Inspect balance trajectories")}</section>
<p>{link(args.assembly / "assembly.step", "Download source assembly STEP")} · {link(args.assembly / "report.json", "Assembly checks")} · {link(Path("docs/design/integrated-assembly.md"), "Current integration notes and next steps", evidence=False)}</p>
<p>These recordings retain the geometry, mass and source identities linked above. The simplified physics contacts do not qualify every detailed CAD interaction. System power qualification and the $1,000 production selling-price target remain open.</p></main></html>"""
    (args.output / "review.html").write_text(content)
    script = Path(__file__).resolve().relative_to(Path.cwd())
    archived_script = args.output / "source" / script
    archived_script.parent.mkdir(parents=True, exist_ok=True)
    archived_script.write_bytes(Path(__file__).read_bytes())
    report = dict(
        status="pass",
        scope="retained-source/output integrity and matching assembly/mass/trial identities",
        product_status="incomplete",
        source_bindings=bindings,
        source_hashes={str(script): sha(archived_script)},
        output_hashes={"review.html": sha(args.output / "review.html")},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status="pass", output=str(args.output))))


if __name__ == "__main__":
    main()
