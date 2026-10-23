"""Source-bound review of encoder-posture balance and remaining stress limits."""

import argparse
import html
import json
from pathlib import Path
import subprocess
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report

NORMAL = {
    "tilt-recovery",
    "reverse-tilt",
    "drive-return",
    "bus-phase-and-poll-gaps",
    "rotated-mount-corrected",
    "noisy-biased-imu",
    "delayed-command",
}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}
    suites = []
    summary = []
    stressrows = []

    def load(p):
        p = Path(p)
        bindings[str(p)] = sha(p)
        return json.loads(p.read_text())

    def bind(p):
        p = Path(p)
        bindings[str(p)] = sha(p)
        return p

    def read_suite(folder):
        r = load(folder / "balance-report.json")
        if r["status"] == "running":
            raise ValueError("Live result is not reviewable")
        for name, digest in r["source_hashes"].items():
            if sha(folder / "source" / name) != digest:
                raise ValueError("Retained source differs")
        return r

    for name in ("pip", "scout"):
        model = Path(f"artifacts/{name}-integral-head-mass-v1")
        verify_report(model)
        bind(model / "report.json")
        for profile in ("lower", "nominal", "upper"):
            folder = Path(f"artifacts/{name}-composed-observed-{profile}-balance-v1")
            r = read_suite(folder)
            interface = load(model / profile / "interfaces.json")
            if (
                load(folder / "interfaces.json") != interface
                or r["mass_model"] != interface["mass_model"]
            ):
                raise ValueError("Mass/interface mismatch")
            normal = [c for c in r["cases"] if c["case"] in NORMAL]
            drive = next(c for c in normal if c["case"] == "drive-return")
            cases = []
            for c in r["cases"]:
                initial = load(folder / c["case"] / "initial-state.json")
                if initial["contacts"]:
                    raise ValueError("Unexpected initial primitive contact")
                trace = bind(folder / c["case"] / "trajectory.jsonl")
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                active = [
                    x for x in rows if x.get("trim_observation_time_s") is not None
                ]
                if active:
                    if any(
                        abs(
                            x["trim_observation_time_s"]
                            - x["control_timing"]["issued_time_s"]
                        )
                        > 1e-9
                        for x in active
                    ):
                        raise ValueError("Trim is not from command acquisition state")
                cases.append(
                    dict(
                        case=c["case"],
                        status=c["status"],
                        termination=c.get("termination"),
                        trim_range_rad=[
                            min(x["pitch_reference_rad"] for x in active),
                            max(x["pitch_reference_rad"] for x in active),
                        ]
                        if active
                        else None,
                        max_axle_mismatch_m=max(
                            (x["trim_axle_mismatch_m"] for x in active), default=None
                        ),
                    )
                )
            suites.append(
                dict(robot=name, profile=profile, folder=str(folder), cases=cases)
            )
            summary.append(
                f'<tr><td>{name.title()} / {profile}</td><td>{r["mass_model"]["total_mass_kg"]:.3f} kg</td><td>{sum(c["status"] == "pass" for c in normal)}/{len(normal)}</td><td>{drive["final_position_error_m"] * 1000:.3f} mm</td><td><a href="../{folder.name}/review.html">Inspect trials</a></td></tr>'
            )
            bind(folder / "review.html")
        # Controlled ablation for plots: identical rest pose and all gains; only trim mode differs.
        fixed = Path(f"artifacts/{name}-composed-outward-selection-v1")
        old = read_suite(fixed)
        new = read_suite(Path(f"artifacts/{name}-composed-observed-nominal-balance-v1"))
        a = dict(old["controller"])
        b = dict(new["controller"])
        a.pop("pitch_trim")
        b.pop("pitch_trim")
        if a != b or old["mass_model"] != new["mass_model"]:
            raise ValueError("Fixed/observed plot comparison changed other settings")
        for case in ("tilt-recovery", "drive-return"):
            bind(fixed / case / "trajectory.jsonl")
        folder = Path(f"artifacts/{name}-composed-observed-stress-v1")
        r = read_suite(folder)
        if r["mass_model"] != new["mass_model"]:
            raise ValueError("Stress mass differs from nominal comparison")
        stress = []
        for c in r["cases"]:
            trace = bind(folder / c["case"] / "trajectory.jsonl")
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            hold = [x for x in rows if x["time_s"] > c["input"]["duration_s"] - 2]
            maxhold = max((abs(x["pitch_rad"]) for x in hold), default=None)
            reason = c.get("termination")
            if not reason and c["status"] == "fail":
                exceeded = []
                if c["final_position_error_m"] > c["input"]["final_position_limit_m"]:
                    exceeded.append("final position")
                if (
                    maxhold is not None
                    and maxhold > c["input"]["final_pitch_limit_rad"]
                ):
                    exceeded.append(
                        f"hold pitch {maxhold:.6f} > {c['input']['final_pitch_limit_rad']:.3f} rad"
                    )
                if c["max_hold_speed_m_s"] > c["input"]["final_speed_limit_m_s"]:
                    exceeded.append("hold speed")
                reason = "; ".join(exceeded) or "inspect other acceptance limits"
            stress.append(
                dict(
                    case=c["case"],
                    status=c["status"],
                    reason=reason,
                    max_hold_pitch_rad=maxhold,
                )
            )
            stressrows.append(
                f"<tr><td>{name.title()}</td><td>{c['case']}</td><td>{c['status']}</td><td>{html.escape(reason or 'All limits met')}</td></tr>"
            )
        suites.append(
            dict(robot=name, profile="nominal_stress", folder=str(folder), cases=stress)
        )
        bind(folder / "review.html")
    normal_rows = [
        case
        for suite in suites
        if suite["profile"] != "nominal_stress"
        for case in suite["cases"]
        if case["case"] in NORMAL
    ]
    if len(normal_rows) != 26 or any(case["status"] != "pass" for case in normal_rows):
        raise ValueError("Revise normal-case success narrative")
    expected_negative = {
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
    for suite in suites:
        if suite["profile"] == "nominal":
            negatives = [c for c in suite["cases"] if c["case"] not in NORMAL]
            if len(negatives) != 9 or any(
                c["termination"] != expected_negative[c["case"]] for c in negatives
            ):
                raise ValueError("Revise negative-case mechanism narrative")
    thigh = Path("artifacts/arm-thigh-postures-v1")
    cad = verify_report(thigh)
    bind(thigh / "report.json")
    gaps = []
    for r in cad["robots"]:
        for pose in r["poses"]:
            if len(pose["pairs"]) != 270 or pose["status"] != "pass":
                raise ValueError("Revise clearance narrative")
            gaps.append(
                dict(
                    robot=r["robot"],
                    pose=pose["pose"],
                    minimum_mm=min(x["distance_mm"] for x in pose["pairs"]),
                )
            )
    cover_folder = Path("artifacts/cover-thigh-postures-v2")
    covers = verify_report(cover_folder)
    bind(cover_folder / "report.json")
    cover_gaps = []
    for robot in covers["robots"]:
        for pose in robot["pose_rows"]:
            rows = pose["pairs"]
            expected_shells = {
                f"{side}_cover_halves_{half}"
                for side in ("left", "right") for half in ("front", "rear")
            }
            if (len(rows) != 180 or len(pose["cover_ids"]) != 20
                    or not expected_shells.issubset(pose["cover_ids"])
                    or any(p["intersection_volume_mm3"] > 1e-6 or p["distance_mm"] <= 0 for p in rows)):
                raise ValueError("Revise cover clearance narrative")
            cover_gaps.append(dict(robot=robot["robot"], pose=pose["pose"],
                                   minimum_mm=min(p["distance_mm"] for p in rows)))
    view = Path("artifacts/control-postures-review-v1")
    verify_report(view)
    bind(view / "report.json")
    bind(view / "index.html")
    for p in (
        "tools/plot_observed_balance.py",
        str(Path(__file__).relative_to(Path.cwd())),
    ):
        bind(p)
    subprocess.run(
        [
            ".venv-plot/bin/python",
            "tools/plot_observed_balance.py",
            "--output",
            str(output),
        ],
        check=True,
    )
    document = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Encoder-posture balance</title><style>body{font:17px/1.5 system-ui;background:#f4f2ec;color:#183933;max-width:1180px;margin:32px auto;padding:0 24px}h1{font-size:40px}table{width:100%;border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}a{color:#006957}img{width:100%}iframe{width:100%;height:840px;border:0}section{margin:32px 0;overflow:auto}.note{padding:18px;background:#fff1cc;border-radius:12px}</style><h1>Balance follows the robot's measured posture</h1><p>The current controller derives its lean target from encoder joint positions and the declared mass model. It uses no simulator root pose for control. Gains, effort limits and acceptance thresholds are unchanged.</p><div class="note">26/26 normal cases pass across two robots and three planning masses. Stress tests still fail. These are reference-controller simulations with unmeasured hardware assumptions, not learned skills or physical qualification.</div><section><h2>Inspect the control rest postures</h2><iframe src="../control-postures-review-v1/index.html" title="Complete robot CAD at three control rest postures"></iframe><p>The selected arm/thigh CAD pairs already clear at zero shoulder roll: 5.5 mm Pip and 6.5 mm Scout. A separate check covers all 20 named cover groups, including halves, screws and inserts: 180 cover/thigh pairs per pose and robot. At zero shoulder roll the cover gap is 5.672 mm Pip / 0.750 mm Scout; at ±0.04 rad it is 8.868 / 6.040 mm. All selected pairs have positive nominal gaps. Outward rolls +0.04/−0.04 rad increase the selected gaps to 9.032/10.534 mm and remove initial primitive contacts. Full self-clearance and motion between poses remain unqualified.</p></section><section><h2>Normal cases and mass bounds</h2><table><tr><th>Robot / profile</th><th>Planning mass</th><th>Normal passes</th><th>Drive-return error</th><th>Evidence</th></tr>"""
        + "".join(summary)
        + """</table><p>Nominal has seven normal cases plus nine intentional negatives. Negative episodes retain fail status and their termination mechanisms; startup faults have power not_run. Each mass bound has three cases, not the full nominal suite.</p></section><section><h2>Same posture, corrected lean target</h2><p>The plots compare fixed and encoder-derived trim with identical outward arm posture, mass and gains. Position truth is evaluation-only. Lean reference and its encoder acquisition time are retained per command. The dotted position bounds concern final hold; drive path has its separate 40 mm limit.</p><img src="pip-trim.png" alt="Pip fixed and observed trim position and pitch traces"><img src="scout-trim.png" alt="Scout fixed and observed trim position and pitch traces"></section><section><h2>Stress conditions still define limits</h2><table><tr><th>Robot</th><th>Case</th><th>Status</th><th>Failure mechanism</th></tr>"""
        + "".join(stressrows)
        + """</table><p><a href="../pip-composed-observed-stress-v1/review.html">Pip stress traces</a> · <a href="../scout-composed-observed-stress-v1/review.html">Scout stress traces</a>. Existing stress conditions are evaluated with driver-based sensing; no gains were selected from these outcomes. Scout's noisy seed 7 misses the final-hold pitch limit even though its final position passes.</p></section><section><h2>Model and build boundaries</h2><p>Quasi-static trim uses assumed masses and current encoders. It rejects more than 2 mm left/right axle disagreement in the sagittal plane. It does not identify payloads, compensate inertial forces, correct slip or reset the odometry origin. The independent physics test checks gravity-moment cancellation; a physical actuator/IMU bench is still required.</p><p>Detailed arm/thigh pair checks do not validate complete collision geometry. Contact proxies need refinement against the actual CAD before relying on more extensive self-contact behavior. Supported placement remains separately partial, and the production price remains unproven.</p><a href="../cover-thigh-postures-v2/report.json">Cover/thigh pair report</a> · <a href="../arm-thigh-postures-v1/report.json">Mechanism/thigh pair report</a> · <a href="../composed-dynamics-review-v2/index.html">Earlier results and manipulation failures</a> · <a href="report.json">Review source bindings</a></section></html>"""
    )
    (output / "index.html").write_text(document)
    write(
        output / "report.json",
        dict(
            status="partial",
            scope="Observed-posture balance, bounded CAD and stress comparison; not product qualification",
            source_bindings=bindings,
            suites=suites,
            arm_thigh_gaps=gaps,
            cover_thigh_gaps=cover_gaps,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
