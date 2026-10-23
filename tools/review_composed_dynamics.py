"""Compare retained composed-mass dynamics without transferring old-model success."""

import argparse
import hashlib
import html
import json
from pathlib import Path

NORMAL = {
    "tilt-recovery",
    "reverse-tilt",
    "drive-return",
    "bus-phase-and-poll-gaps",
    "rotated-mount-corrected",
    "noisy-biased-imu",
    "delayed-command",
}
EXPECTED = {
    "io-fault": "imu_fault",
    "conversion-dropout": "imu_stale",
    "register-saturation": "imu_saturated",
    "excessive-poll-block": "imu_stale",
    "wrong-mount-calibration": "tilt_limit",
    "disabled-controller": "tilt_limit",
    "startup-io-fault": "imu_startup_invalid",
    "startup-dropout": "imu_startup_invalid",
    "startup-saturation": "imu_startup_invalid",
}


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bind(path):
        path = Path(path)
        bindings[str(path)] = sha(path)
        return path

    def load(path):
        return json.loads(bind(path).read_text())

    def verify(folder, filename):
        d = load(folder / filename)
        if d.get("status") == "running":
            raise ValueError(f"Unfinished producer: {folder}")
        for name, digest in d.get("source_hashes", {}).items():
            if sha(folder / "source" / name) != digest:
                raise ValueError(f"Changed retained source: {folder}/{name}")
        for name, digest in d.get("output_hashes", {}).items():
            if sha(folder / name) != digest:
                raise ValueError(f"Changed retained output: {folder}/{name}")
        return d

    def anchor(path, label):
        path = Path(path)
        bind(path)
        import os

        return f'<a href="{html.escape(os.path.relpath(path, output))}">{html.escape(label)}</a>'

    rows, taskrows, suites = [], [], []
    for name in ("pip", "scout"):
        mass_folder = Path(f"artifacts/{name}-integral-head-mass-v1")
        mass_report = verify(mass_folder, "report.json")
        if mass_report["assembly_evidence"]["status"] != "fail":
            raise ValueError("Revise source assembly scope")
        for profile in ("lower", "nominal", "upper"):
            model = mass_folder / profile
            interface = load(model / "interfaces.json")
            identity = interface["mass_model"]
            for label, suffix in (
                ("Zero posture", f"{profile}-balance"),
                ("Parked arms", f"parked-{profile}-balance"),
            ):
                folder = Path(f"artifacts/{name}-composed-{suffix}-v1")
                report = verify(folder, "balance-report.json")
                if (
                    report["mass_model"] != identity
                    or load(folder / "interfaces.json") != interface
                ):
                    raise ValueError("Balance mass/interface mismatch")
                normal = [c for c in report["cases"] if c["case"] in NORMAL]
                faults = [
                    dict(
                        case=c["case"],
                        termination=c.get("termination"),
                        expected=EXPECTED[c["case"]],
                        mechanism_matches=c.get("termination") == EXPECTED[c["case"]],
                    )
                    for c in report["cases"]
                    if c["case"] not in NORMAL
                ]
                initial = []
                for c in report["cases"]:
                    trial = folder / c["case"]
                    if (trial / "trajectory.jsonl").exists():
                        bind(trial / "trajectory.jsonl")
                    if (trial / "initial-state.json").exists():
                        state = load(trial / "initial-state.json")
                        initial.append(state)
                        overlaps = [
                            p
                            for p in state["contacts"]
                            if p["distance_m"] < -1e-6
                            and "wheel" in p["geom1"] + p["geom2"]
                            and "gripper" in p["geom1"] + p["geom2"]
                        ]
                        if overlaps:
                            raise ValueError(
                                "Parked initial wheel/gripper interference remains"
                            )
                record = dict(
                    robot=name,
                    profile=profile,
                    posture=label,
                    mass_kg=identity["total_mass_kg"],
                    mass_sha256=identity["sha256"],
                    folder=str(folder),
                    normal_pass=sum(c["status"] == "pass" for c in normal),
                    normal_count=len(normal),
                    cases=report["cases"],
                    negatives=faults,
                    initial_state_count=len(initial),
                    initial_contacts=initial[0]["contacts"] if initial else None,
                )
                suites.append(record)
                failures = (
                    ", ".join(
                        c["case"] + ": " + (c.get("termination") or "accuracy limit")
                        for c in normal
                        if c["status"] != "pass"
                    )
                    or "None in this suite"
                )
                rows.append(
                    f"<tr><td>{name.title()} / {profile}</td><td>{identity['total_mass_kg']:.3f}</td>"
                    f"<td>{label}</td><td>{record['normal_pass']}/{len(normal)}</td>"
                    f"<td>{html.escape(failures)}</td><td>{anchor(folder / 'review.html', 'Inspect trials')}</td></tr>"
                )
            folder = Path(f"artifacts/{name}-composed-{profile}-task-v1")
            report = verify(folder, "report.json")
            if load(folder / "interfaces.json") != interface:
                raise ValueError("Task interface mismatch")
            if report.get("mass_model", identity) != identity:
                raise ValueError("Task mass mismatch")
            bind(folder / "trajectory.jsonl")
            complete = "source_hashes" in report and "termination" in report
            if complete:
                detail = f"{report['termination']}; peak lift {1000 * report['max_lift_m']:.2f} mm"
            else:
                detail = "Incomplete runner: " + report.get("error", "unknown error")
                bind(folder / "error.log")
            taskrows.append(
                dict(
                    robot=name,
                    profile=profile,
                    status=report["status"],
                    completed=complete,
                    detail=detail,
                    folder=str(folder),
                    producer_attestation="source snapshots checked"
                    if complete
                    else "incomplete producer; interface matched; files bound by this review only",
                )
            )
    # Retain startup-only and tuning failures as selection evidence, not held-out skill validation.
    rejected = []
    rejected_folders = [
        Path(f"artifacts/{n}-composed-startup-balance-v1") for n in ("pip", "scout")
    ]
    rejected_folders += sorted(Path("artifacts").glob("pip-control-screen-*-v[12]"))
    for folder in rejected_folders:
        report = verify(folder, "balance-report.json")
        rejected.append(
            dict(
                folder=str(folder),
                status=report["status"],
                cases=[
                    dict(
                        case=c["case"],
                        status=c["status"],
                        termination=c.get("termination"),
                    )
                    for c in report["cases"]
                ],
            )
        )
    open_controls = []
    for name in ("pip", "scout"):
        folder = Path(f"artifacts/{name}-composed-open-task-v1")
        report = verify(folder, "report.json")
        open_controls.append(
            dict(
                robot=name,
                status=report["status"],
                termination=report["termination"],
                grasped=report["grasped"],
                lifted=report["lifted"],
            )
        )
    nominal = [
        s for s in suites if s["profile"] == "nominal" and s["posture"] == "Parked arms"
    ]
    drive_cards = []
    for s in nominal:
        folder = Path(s["folder"])
        case = next(c for c in s["cases"] if c["case"] == "drive-return")
        image = anchor(folder / "drive-return/episode.gif", "Open animation")
        import os

        src = os.path.relpath(folder / "drive-return/episode.gif", output)
        drive_cards.append(
            f"<article><h3>{s['robot'].title()} drive-return · {case['status']}</h3>"
            f'<img src="{src}" alt="Recorded simulated drive and return">'
            f"<p>Peak path error {case['max_path_error_m'] * 1000:.2f} mm; "
            f"final return error {case['final_position_error_m'] * 1000:.2f} mm.</p>{image}</article>"
        )
    task_table_rows = []
    for t in taskrows:
        folder = Path(t["folder"])
        links = anchor(folder / "report.json", "Report")
        if (folder / "review.html").exists():
            links += " · " + anchor(folder / "review.html", "Inspect recorded task")
        else:
            links += " · " + anchor(
                folder / "phase-04-lower.png", "Last rendered phase"
            )
            links += " · " + anchor(
                folder / "trajectory.jsonl", "Incomplete trajectory"
            )
        task_table_rows.append(
            f"<tr><td>{t['robot'].title()} / {t['profile']}</td>"
            f"<td>{t['status']}</td><td>{html.escape(t['detail'])}</td>"
            f"<td>{links}</td></tr>"
        )
    task_table = "".join(task_table_rows)
    faults = "".join(
        f"<tr><td>{s['robot'].title()} · {s['posture']}</td><td>{n['case']}</td>"
        f"<td>{n['termination']}</td><td>{n['expected']}</td><td>{n['mechanism_matches']}</td></tr>"
        for s in suites
        if s["profile"] == "nominal"
        for n in s["negatives"]
    )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Composed robot · dynamics</title><style>body{font:16px system-ui;background:#f4f2ec;color:#183933;max-width:1240px;margin:32px auto;padding:0 24px;line-height:1.5}h1{font-size:42px}a{color:#006957}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}section{margin:32px 0;overflow:auto}.cards{display:flex;gap:24px}.cards article{flex:1;min-width:0;background:white;padding:20px;border-radius:16px}img{width:100%}.note{padding:20px;background:#fff1cc;border-radius:12px}@media(max-width:700px){.cards{display:block}}</style>
<h1>Same robot. Declared mass. Explicit posture.</h1><p>The complete head composition is now tested with its lower, nominal and upper planning masses. Overall control qualification remains partial; physical evidence is not run.</p>
<div class="note">The all-zero arm pose overlaps wheel and gripper collision proxies by about 7 mm. Parking both elbows at −0.4 rad removes those initial proxy contacts. Startup, held commands and static trim use this same posture. Gains and acceptance limits are unchanged. These are simplified contacts, not complete CAD clearance. Scout retains 2 mm forearm/thigh overlap; Pip has nominal touching surfaces. The posture is not released for hardware.</div>
<section><h2>Inspect the current drive trials</h2><p>Primitive contact simulation with explicit composed-assembly inertia. Rendered meshes omit some detailed mechanisms; use the complete CAD inspector for part fit.</p><div class="cards">"""
        + "".join(drive_cards)
        + """</div></section>
<section><h2>Balance across mass assumptions</h2><p>Pass counts cover the normal cases only: nominal has seven; each bound has three. Separate mass bounds are not a combined robustness qualification.</p><table><tr><th>Robot / mass profile</th><th>kg</th><th>Posture</th><th>Pass</th><th>Remaining failures</th><th>Evidence</th></tr>"""
        + "".join(rows)
        + """</table></section>
<section><h2>Supported block placement</h2><p>50 g block, fixed torso, reference feedback and simulator-provided object state. These runs use a separate one-arm rest pose. No learned policy or free-base manipulation claim.</p><table><tr><th>Robot / profile</th><th>Status</th><th>Outcome</th><th>Evidence</th></tr>"""
        + task_table
        + """</table><p>Both open-controller controls fail without grasp or lift. Pip nominal is an incomplete IK run, not a completed negative trial; its partial trajectory and error are retained.</p></section>
<section><details><summary>Fault tests: did the intended mechanism occur?</summary><p>A failed episode alone is insufficient. Earlier zero-pose trials often fell before the injected fault. Matching the listed mechanism is a simulation check only; physical fault response is unqualified.</p><table><tr><th>Robot / posture</th><th>Case</th><th>Actual termination</th><th>Expected</th><th>Matches</th></tr>"""
        + faults
        + """</table></details></section>
<section><h2>What to build next</h2><p>Print the fit coupons and actuator cradle first, then measure actuator fit, backlash, loaded response and current on a restrained joint bench. Defer a free-standing robot until interference, dynamics, stop/reset and power checks close. Whole-robot CAD failures, harness, fasteners and physical strength remain open.</p><p>Two startup-only suites and ten rejected gain/filter candidates are retained in the data. None is promoted as a learned skill or held-out validation.</p><a href="report.json">Evidence bindings and outcomes</a></section></html>"""
    )
    (output / "index.html").write_text(document)
    bind(
        Path(__file__).relative_to(Path.cwd())
        if Path(__file__).is_absolute()
        else Path(__file__)
    )
    report = dict(
        status="partial",
        scope="retained composed-mass dynamics comparison; not whole-product qualification",
        suites=suites,
        tasks=taskrows,
        open_controls=open_controls,
        rejected=rejected,
        source_bindings=bindings,
        output_hashes={"index.html": sha(output / "index.html")},
        physical_status="not_run",
        detailed_motion_clearance="not_run",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                suites=len(suites),
                tasks=len(taskrows),
                bindings=len(bindings),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
