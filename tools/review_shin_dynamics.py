"""Present source-verified shin-cover inertia probes beside the prior robot."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_forearm_accounted import (
    NORMAL_BALANCE,
    bind_recording_inputs,
    verify_archive,
    verify_episode,
)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def read(path):
        path = Path(path)
        bindings[str(path)] = sha(path)
        return json.loads(path.read_text())

    def link(path, title):
        bindings[str(path)] = sha(path)
        return f'<a href="../../{html.escape(str(path))}">{html.escape(title)}</a>'

    verify_report(Path("artifacts/shin-cover-review-v1"))
    read(Path("artifacts/shin-cover-review-v1/report.json"))
    verify_report(Path("artifacts/shin-hold-pitch-v1"))
    read(Path("artifacts/shin-hold-pitch-v1/report.json"))
    plot = Path("artifacts/shin-hold-pitch-v1/hold-pitch.png")
    bindings[str(plot)] = sha(plot)
    masses, balance, tasks, checks = [], [], [], []
    for name in ("pip", "scout"):
        oldmass = Path(f"artifacts/{name}-forearm-accounted-mass-v1")
        newmass = Path(f"artifacts/{name}-shin-covered-mass-v1")
        assembly = Path(
            f"artifacts/{name}-shin-cover-composed-{'v3' if name == 'pip' else 'v2'}"
        )
        for folder in (oldmass, newmass, assembly):
            verify_report(folder)
            read(folder / "report.json")
        read(Path(f"artifacts/{name}-shin-dynamics-launch-v1/report.json"))
        before_ledger = {
            p["name"]: p for p in read(oldmass / "part-ledger.json")["parts"]
        }
        after_ledger = {
            p["name"]: p for p in read(newmass / "part-ledger.json")["parts"]
        }
        new_names = set(after_ledger) - set(before_ledger)
        if len(new_names) != 18 or set(before_ledger) - set(after_ledger):
            raise ValueError("Unexpected mass inventory delta")
        for key, old in before_ledger.items():
            new = after_ledger[key]
            for field in ("owner", "kind", "role", "module"):
                if old.get(field) != new.get(field):
                    raise ValueError(f"Part identity changed: {key}/{field}")
            if key not in ("left_shin_spine", "right_shin_spine"):
                # Reserve evidence prose changes on thigh owners, never their mass.
                if old["profiles"] != new["profiles"]:
                    for profile in ("lower", "nominal", "upper"):
                        for field in ("mass_kg", "center_m", "tensor_kg_m2"):
                            if not np.allclose(
                                old["profiles"][profile][field],
                                new["profiles"][profile][field],
                                rtol=1e-6,
                                atol=1e-10,
                            ):
                                raise ValueError(
                                    f"Unexpected non-shin mass change: {key}"
                                )
        checks.append(
            dict(
                robot=name,
                new_ledger_entries=sorted(new_names),
                unchanged_parts_and_reserves_mass_identity="pass",
            )
        )
        for profile in ("lower", "nominal", "upper"):
            before = read(oldmass / profile / "mass-properties.json")
            after = read(newmass / profile / "mass-properties.json")
            masses.append(
                dict(
                    robot=name,
                    profile=profile,
                    before_kg=before["total_mass_kg"],
                    after_kg=after["total_mass_kg"],
                    delta_g=1000 * (after["total_mass_kg"] - before["total_mass_kg"]),
                )
            )
            folder = Path(f"artifacts/{name}-shin-covered-{profile}-balance-v1")
            report = read(folder / "balance-report.json")
            prior = read(
                Path(
                    f"artifacts/{name}-forearm-accounted-{profile}-balance-v1/balance-report.json"
                )
            )
            verify_archive(folder, report, bindings)
            for key in (
                "lab/balance.py",
                "lab/io.py",
                "lab/control.py",
                "lab/imu_sim.py",
                "lab/imu.py",
                "lab/session.py",
                "lab/hardware.py",
                "lab/transport.py",
            ):
                if report["source_hashes"][key] != prior["source_hashes"][key]:
                    raise ValueError(f"Balance implementation changed: {key}")
            expected = read(newmass / profile / "interfaces.json")
            actual = read(folder / "interfaces.json")
            if actual != expected or report["mass_model"]["sha256"] != sha(
                newmass / profile / "mass-properties.json"
            ):
                raise ValueError("Balance mass identity differs")
            for current, previous in zip(report["cases"], prior["cases"], strict=True):
                if (
                    current["input"] != previous["input"]
                    or report["controller"] != prior["controller"]
                ):
                    raise ValueError("Control comparison changed conditions")
                bind_recording_inputs(folder / current["case"], "balance", bindings)

                def hold_pitch(path, duration):
                    bindings[str(path)] = sha(path)
                    with path.open() as stream:
                        samples = (json.loads(line) for line in stream if line.strip())
                        return max(
                            (
                                abs(row["pitch_rad"])
                                for row in samples
                                if row["time_s"] > duration - 2
                            ),
                            default=None,
                        )

                peak_pitch = hold_pitch(
                    folder / current["case"] / "trajectory.jsonl",
                    current["input"]["duration_s"],
                )
                prior_peak_pitch = hold_pitch(
                    Path(f"artifacts/{name}-forearm-accounted-{profile}-balance-v1")
                    / current["case"]
                    / "trajectory.jsonl",
                    previous["input"]["duration_s"],
                )
                balance.append(
                    dict(
                        robot=name,
                        profile=profile,
                        case=current["case"],
                        normal=current["case"] in NORMAL_BALANCE,
                        before=previous["status"],
                        after=current["status"],
                        hold_pitch_rad=peak_pitch,
                        prior_hold_pitch_rad=prior_peak_pitch,
                        hold_pitch_limit_rad=current["input"]["final_pitch_limit_rad"],
                        final_position_error_m=current.get("final_position_error_m"),
                        prior_final_position_error_m=previous.get(
                            "final_position_error_m"
                        ),
                        review=str(folder / "review.html"),
                    )
                )
        for case in ("lower", "nominal", "upper", "open"):
            profile = "nominal" if case == "open" else case
            folder = Path(f"artifacts/{name}-shin-covered-{case}-task-v1")
            report = read(folder / "report.json")
            prior = read(
                Path(f"artifacts/{name}-forearm-accounted-{case}-task-v1/report.json")
            )
            verify_archive(folder, report, bindings)
            episode = verify_episode(folder, report, bindings)
            previous_episode = read(
                Path(f"artifacts/{name}-forearm-accounted-{case}-task-v1/episode.json")
            )
            for key in ("request", "seed", "controller"):
                if episode[key] != previous_episode[key]:
                    raise ValueError(f"Task condition changed: {key}")
            for key in ("lab/manipulation.py", "lab/tasks.py", "lab/simulation.py"):
                if report["source_hashes"][key] != prior["source_hashes"][key]:
                    raise ValueError(f"Task implementation changed: {key}")
            bind_recording_inputs(folder, "task", bindings)
            if read(folder / "interfaces.json") != read(
                newmass / profile / "interfaces.json"
            ):
                raise ValueError("Task mass interface differs")
            if (
                report["mass_model"]["sha256"]
                != sha(newmass / profile / "mass-properties.json")
                or report["controller"] != prior["controller"]
            ):
                raise ValueError("Task comparison identity differs")
            tasks.append(
                dict(
                    robot=name,
                    case=case,
                    before=prior["status"],
                    after=report["status"],
                    termination=report.get("termination"),
                    duration_s=report.get("duration_s"),
                    review=str(folder / "review.html"),
                    gif=str(folder / "episode.gif"),
                )
            )
    normal = [r for r in balance if r["normal"]]
    success = sum(r["after"] == "pass" for r in normal)
    taskpass = sum(r["after"] == "pass" for r in tasks if r["case"] != "open")
    style = "body{font:16px/1.6 system-ui;background:#f5f3eb;color:#253f37;margin:35px auto;padding:0 24px;max-width:1250px}h1{font-size:42px;line-height:1.15}table{border-collapse:collapse;width:100%;background:#fffef8;margin:20px 0}td,th{padding:9px;border-bottom:1px solid #cdd5ca;text-align:left}a{color:#237164}.scope{padding:18px;border-left:4px solid #b56b36;background:#fff8e7}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:18px}article{background:#fffef8;padding:18px;border-radius:14px}img{width:100%}.pass{color:#237164}.fail{color:#a74128}"
    page = f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shin covers · mass and control</title><style>{style}</style><p>ROBOT LEARNING LAB / PIP + SCOUT</p><h1>Covered shins, tested inertia.</h1><p>{success}/{len(normal)} normal balance probes and {taskpass}/6 supported placement trials pass.</p><p class="scope">These are simulations of the new mass, center of mass and inertia. Contact geometry and replay visuals retain the earlier model: the new covers cannot collide in these control runs. Whole-robot fit failures, physical verification and production-price qualification remain open. No reserve mass credit or controller retuning is taken.</p>'
    page += (
        link(
            Path("artifacts/shin-cover-review-v1/index.html"),
            "Inspect both covered robots in 3D",
        )
        + " · "
        + link(
            Path("docs/design/shin-cover-dynamics.md"), "Method, limits and next steps"
        )
    )
    page += "<h2>Complete candidate CAD</h2><p>"
    page += (
        " · ".join(
            link(
                Path(
                    f"artifacts/{n}-shin-cover-composed-{'v3' if n == 'pip' else 'v2'}/assembly.step"
                ),
                f"{n.title()} complete STEP",
            )
            for n in ("pip", "scout")
        )
        + '</p><iframe src="../shin-cover-review-v1/index.html" title="Inspect the complete covered robots" style="width:100%;height:820px;border:1px solid #cdd5ca;border-radius:14px"></iframe><h2>Planning mass</h2><table><tr><th>Robot / profile</th><th>Before kg</th><th>Covered kg</th><th>Added g</th></tr>'
    )
    for r in masses:
        page += f"<tr><td>{r['robot'].title()} / {r['profile']}</td><td>{r['before_kg']:.4f}</td><td>{r['after_kg']:.4f}</td><td>{r['delta_g']:.2f}</td></tr>"
    page += '</table><p>Existing leg reserve stays at 15/28/45 g per link for lower/nominal/upper assumptions. It may overstate the remaining material; it does not bound all real-world uncertainty. Material and fastener masses are unmeasured.</p><h2>Supported placement</h2><p>Replay meshes show the earlier geometry. These trials isolate the mass change; failed heavier cases and open-gripper controls remain visible.</p><div class="cards">'
    for r in tasks:
        bindings[r["gif"]] = sha(r["gif"])
        page += f'<article><h3>{r["robot"].title()} · {r["case"]}</h3><p class="{r["after"]}">{r["before"].upper()} → {r["after"].upper()} · {html.escape(str(r["termination"]))}</p><img loading="lazy" src="../../{r["gif"]}" alt="Earlier visual geometry with candidate inertia">{link(Path(r["review"]), "Inspect episode")}</article>'
    page += '</div><h2>Scout loses some upright margin</h2><p>Two previously passing cases exceed the unchanged hold-pitch limit with the cover inertia. The proposed parts remain under development.</p><img src="../shin-hold-pitch-v1/hold-pitch.png" alt="Scout cover inertia raises absolute hold pitch above the unchanged 25 mrad limit in two cases"><h2>Every balance case</h2><p>Same gains, input cases and limits. Normal cases are separated from deliberate fault controls; this is not a new stress qualification. Hold pitch is the largest absolute pitch in the final two seconds. One mrad is 0.001 rad.</p><table><tr><th>Robot / mass</th><th>Case</th><th>Before → covered</th><th>Final error, mm</th><th>Hold pitch / limit, mrad</th><th>Evidence</th></tr>'
    for r in balance:
        err = r["final_position_error_m"]
        errtext = "unavailable" if err is None else f"{1000 * err:.2f}"
        pitchtext = (
            "unavailable"
            if r["hold_pitch_rad"] is None
            else f"{1000 * r['hold_pitch_rad']:.3f} / {1000 * r['hold_pitch_limit_rad']:.3f}"
        )
        page += f'<tr><td>{r["robot"].title()} / {r["profile"]}</td><td>{r["case"]}</td><td class="{r["after"]}">{r["before"]} → {r["after"]}</td><td>{errtext}</td><td>{pitchtext}</td><td>{link(Path(r["review"]), "Suite")}</td></tr>'
    page += '</table><p><a href="report.json">Retained comparison and integrity report</a></p></html>'
    (output / "index.html").write_text(page)
    (output / "builder.py").write_bytes(Path(__file__).read_bytes())
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation and comparison identity only; engineering results retained",
            physical_status="not_run",
            source_bindings=bindings,
            mass_rows=masses,
            balance_rows=balance,
            task_rows=tasks,
            checks=checks,
            normal_balance_passed=success,
            normal_balance_total=len(normal),
            feedback_task_passed=taskpass,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
