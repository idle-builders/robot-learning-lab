"""Compare unchanged supported placement tasks before/after detached-object guarding."""

import argparse
import hashlib
import html
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(rows, limit):
    """Recorded control-tick effort, not an inferred continuous torque trace."""
    index = rows[0]["joint_state"]["names"].index("left_shoulder_roll_joint")
    points = " ".join(
        f"{40 + r['joint_state']['time_s'] / 16 * 700:.2f},"
        f"{100 - r['joint_state']['effort_nm'][index] / limit * 65:.2f}"
        for r in rows
    )
    return (
        '<svg viewBox="0 0 780 200" role="img" aria-label="Recorded shoulder roll effort over time">'
        '<path d="M40 35H740M40 165H740" stroke="#c26947" stroke-dasharray="5 4"/>'
        '<path d="M40 100H740" stroke="#c4cbc4"/>'
        f'<polyline points="{points}" fill="none" stroke="#216c65" stroke-width="2"/>'
        f'<g font-size="12" fill="#324941"><text x="43" y="25">±{limit:g} Nm configured limit</text>'
        '<text x="40" y="190">0 s</text><text x="708" y="190">16 s</text></g></svg>'
    )


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings, records, cards = {}, [], []

    def bind(path):
        path = Path(path)
        bindings[str(path)] = sha(path)
        return path

    def load(path):
        return json.loads(bind(path).read_text())

    for name in ("pip", "scout"):
        for case in ("lower", "nominal", "upper", "open"):
            folder = Path(f"artifacts/{name}-release-guard-{case}-v1")
            old = Path(f"artifacts/{name}-composed-{case}-task-v1")
            if case == "open":
                old = Path(f"artifacts/{name}-composed-open-task-v1")
            report, prior = load(folder / "report.json"), load(old / "report.json")
            if "termination" not in report:
                raise ValueError(f"Incomplete current trial: {folder}")
            for path, digest in report["source_hashes"].items():
                if sha(folder / "source" / path) != digest:
                    raise ValueError("Changed retained source")
            record = load(folder / "episode.json")
            for filename, key in (
                ("report.json", "report_sha256"),
                ("trajectory.jsonl", "trajectory_sha256"),
                ("resolved-config.json", "robot_config_sha256"),
                ("scene.xml", "scene_sha256"),
            ):
                if sha(bind(folder / filename)) != record[key]:
                    raise ValueError("Episode differs from sealed outputs")
            for filename in ("task.json", "interfaces.json", "resolved-config.json"):
                if load(folder / filename) != load(old / filename):
                    raise ValueError("Compared task/model changed")
            for path in [Path("lab/tasks.py"), *Path("firmware").glob("*.[ch]")]:
                if sha(bind(folder / "source" / path)) != sha(
                    bind(old / "source" / path)
                ):
                    raise ValueError("Evaluator or firmware changed")
            profile = "nominal" if case == "open" else case
            model = Path(f"artifacts/{name}-integral-head-mass-v1/{profile}")
            if report["mass_model"] != load(model / "interfaces.json")["mass_model"]:
                raise ValueError("Wrong mass profile")
            rows = [
                json.loads(line)
                for line in (folder / "trajectory.jsonl").read_text().splitlines()
            ]
            before = [
                json.loads(line)
                for line in bind(old / "trajectory.jsonl").read_text().splitlines()
            ]
            cfg = load(folder / "resolved-config.json")
            interface = load(folder / "interfaces.json")
            joint = next(
                j
                for j in interface["joints"]
                if j["name"] == "left_shoulder_roll_joint"
            )
            limit = cfg["actuators"][joint["family"]]["effort_limit_nm"]
            index = rows[0]["joint_state"]["names"].index(joint["name"])
            transfer = [r for r in rows if r["phase"] == "transfer"]
            saturated = sum(
                abs(r["joint_state"]["effort_nm"][index]) >= limit - 1e-6
                for r in transfer
            )
            # For completed historical trials, validate their source archive too.
            for path, digest in prior.get("source_hashes", {}).items():
                if sha(old / "source" / path) != digest:
                    raise ValueError("Changed historical source")
            row = dict(
                robot=name,
                case=case,
                old_status=prior["status"],
                old_termination=prior.get("termination", "incomplete_ik_abort"),
                old_completed="termination" in prior,
                old_final_time_s=before[-1]["joint_state"]["time_s"],
                status=report["status"],
                termination=report["termination"],
                duration_s=report["duration_s"],
                release_observed=report["release_observed"],
                grasped=report["grasped"],
                lifted=report["lifted"],
                final=report["final"],
                mass_model=report["mass_model"],
                transfer_saturated_samples=saturated,
                transfer_samples=len(transfer),
                shoulder_effort_limit_nm=limit,
                current_folder=str(folder),
                previous_folder=str(old),
            )
            records.append(row)
            for filename in ("episode.gif", "review.html"):
                bind(folder / filename)
            svg = f"{name}-{case}.svg"
            (output / svg).write_text(plot(rows, limit))
            release = report["release_observed"]
            note = (
                f"Contact-loss guard at {release['time_s']:.2f} s."
                if release
                else "No contact-loss guard triggered."
            )
            prior_label = prior.get("termination", "incomplete IK abort")
            cards.append(
                f'<article data-robot="{name}"><h2>{name.title()} · {case}</h2>'
                f'<p class="result {report["status"]}">{report["status"].upper()} · {report["termination"]}</p>'
                f"<p>Before: {html.escape(prior_label)}. Now: {report['duration_s']:.2f} s. {note}</p>"
                f'<img loading="lazy" src="../{folder.name}/episode.gif" alt="{name} {case} supported placement">'
                f'<img loading="lazy" src="{svg}" alt="Shoulder effort trace">'
                f"<p>Transfer torque at the limit: {saturated}/{len(transfer)} recorded samples.</p>"
                f'<a href="../{folder.name}/review.html">Current trial and phases</a> · '
                f'<a href="../{old.name}/report.json">Prior result</a></article>'
            )
    passed = sum(r["status"] == "pass" for r in records if r["case"] != "open")
    content = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robot Learning Lab · supported placement</title><style>
body{font:16px/1.55 system-ui;background:#f4f2e9;color:#243e36;max-width:1250px;margin:auto;padding:36px}h1{font-size:42px;line-height:1.1}a{color:#216c65}header{max-width:900px}select{padding:9px;font:inherit}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:24px;margin-top:24px}article{padding:22px;border:1px solid #cad3c7;border-radius:16px;background:#fffef8}img{width:100%;height:auto}.result{font-weight:700}.pass{color:#216c65}.fail{color:#ad512f}.scope{border-left:4px solid #ad512f;padding-left:18px}[hidden]{display:none!important}
</style><header><p>ROBOT LEARNING LAB / CONTROL EVIDENCE</p><h1>Track the grasp.<br>Keep the failures visible.</h1>"""
    content += f"<p>{passed}/6 supported mass-profile trials pass after the contact-loss guard. All six finish; the prior Pip nominal run aborted.</p>"
    content += """<p>The controller now stops object-relative motion when contact is lost, including while the block is falling or outside the tray. It holds its last arm target and opens the fingers. Task success remains independently evaluated.</p>
<p class="scope">50 g block · same 16 s deadline, task geometry, 1 mm IK gate and actuator limits. Supported torso, privileged object state, assumed masses and simplified collision shapes. This is reference control, not learned skills or physical validation. Current detailed trajectory clearance is NOT RUN; earlier CAD/table and tray failures remain unresolved.</p>
<p>The upper profiles reach the 0.4 Nm shoulder-roll limit during transfer. A controller guard does not establish adequate actuator torque. Measure the selected actuator and arm assembly before promoting these motions.</p>
<p><a href="../../docs/design/release-aware-placement.md">Diagnosis and bench requirements</a> · <a href="report.json">Bound evidence</a> · <a href="../composed-dynamics-review-v2/index.html">Earlier comparison</a></p>
<label>Show robot <select id="robot"><option value="all">Both robots</option><option value="pip">Pip</option><option value="scout">Scout</option></select></label></header><main class="cards">"""
    content += (
        "".join(cards)
        + """</main><script>document.getElementById('robot').addEventListener('change',e=>{for(const card of document.querySelectorAll('article'))card.hidden=e.target.value!=='all'&&card.dataset.robot!==e.target.value;});</script></html>"""
    )
    (output / "index.html").write_text(content)
    (output / "builder.py").write_bytes(Path(__file__).read_bytes())
    result = dict(
        status="pass",
        scope="Presentation identity/integrity only; engineering results remain separate",
        task_pass=passed,
        task_count=6,
        rows=records,
        source_bindings=bindings,
        current_detailed_trajectory_clearance="not_run",
        physical="not_run",
        output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
    )
    (output / "report.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(run(p.parse_args().output)["status"])
