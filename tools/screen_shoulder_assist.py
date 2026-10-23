"""Screen passive shoulder mechanisms against source-bound static gravity loads.

A joint-limit grid is NOT a collision-qualified reachable workspace. Neither
spring hardware nor its mass is installed; do not transfer this to task success.
"""

import argparse
import itertools
import json
from pathlib import Path
import mujoco
import numpy as np
import yaml
from lab.recording import _source_inputs
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def assist(q, spec):
    if spec["kind"] == "none":
        return 0.0, 0.0, 0.0
    if spec["kind"] == "torsion":
        return spec["rate_nm_rad"] * (spec["rest_rad"] - q), 0.0, 0.0
    a, b, phi = (
        spec["anchor_radius_m"],
        spec["moving_radius_m"],
        spec["anchor_phase_rad"],
    )
    length = np.sqrt(a * a + b * b + 2 * a * b * np.cos(q + phi))
    force = spec["rate_n_m"] * max(0, length - spec["free_length_m"])
    return float(force * a * b * np.sin(q + phi) / length), float(length), float(force)


def tension_energy(q, spec):
    a, b, phi = (
        spec["anchor_radius_m"],
        spec["moving_radius_m"],
        spec["anchor_phase_rad"],
    )
    length = np.sqrt(a * a + b * b + 2 * a * b * np.cos(q + phi))
    return 0.5 * spec["rate_n_m"] * max(0, length - spec["free_length_m"]) ** 2


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bind(p):
        p = Path(p)
        bindings[str(p)] = sha(p)
        return p

    def read(p):
        return json.loads(bind(p).read_text())

    manifest = Path("experiments/shoulder-assist.yaml")
    settings = yaml.safe_load(bind(manifest).read_text())
    # Reuse only the explicitly recorded hand-local payload hypothesis.
    diagnosis = Path("artifacts/placement-torque-diagnosis-v2")
    diag = read(diagnosis / "report.json")
    if bind(diagnosis / "report.sha256").read_text().split()[0] != sha(
        diagnosis / "report.json"
    ):
        raise ValueError("Diagnosis seal mismatch")
    for p, d in diag["source_hashes"].items():
        if sha(bind(p)) != d:
            raise ValueError("Diagnosis source mismatch")
    cases = []
    independent_errors = []
    for name in settings["robots"]:
        local_payload = np.array(
            next(r for r in diag["robots"] if r["robot"] == name)[
                "rigid_grasp_transport"
            ]["local_point_in_actual_left_forearm_m"]
        )
        mass_folder = Path(f"artifacts/{name}-forearm-accounted-mass-v1")
        verify_report(mass_folder)
        bind(mass_folder / "report.json")
        for profile in settings["profiles"]:
            folder = Path(f"artifacts/{name}-forearm-accounted-{profile}-task-v1")
            inputs, interface, report, _, assurance = _source_inputs(folder, "task")
            for p in inputs.values():
                bind(p)
            for p in report["source_hashes"]:
                bind(folder / "source" / p)
            if interface != read(mass_folder / profile / "interfaces.json") or report[
                "mass_model"
            ]["sha256"] != sha(bind(mass_folder / profile / "mass-properties.json")):
                raise ValueError("Mass/interface mismatch")
            cfg = read(folder / "resolved-config.json")
            model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
            data = mujoco.MjData(model)
            names = [
                "left_shoulder_roll_joint",
                "left_upper_arm_joint",
                "left_forearm_joint",
            ]
            joints = [model.joint(n) for n in names]
            ranges = [list(model.jnt_range[j.id]) for j in joints]
            order = interface["joint_order"]
            qids = [model.joint(n).qposadr[0] for n in order]

            def load_at(command, payload):
                data.qpos[qids] = command
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                j = joints[0]
                axis = data.xaxis[j.id]
                anchor = data.xanchor[j.id]
                body = data.body("left_forearm")
                point = body.xpos + body.xmat.reshape(3, 3) @ local_payload
                load = -np.cross(point - anchor, payload * model.opt.gravity) @ axis
                # Independent point Jacobian virtual work verifies axis/moment sign.
                jp = np.zeros((3, model.nv))
                jr = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jp, jr, point, body.id)
                independent = -jp[:, j.dofadr[0]] @ (payload * model.opt.gravity)
                independent_errors.append(abs(float(independent - load)))
                return float(data.qfrc_bias[j.dofadr[0]] + load)

            trajectory = [
                json.loads(line)
                for line in bind(folder / "trajectory.jsonl").read_text().splitlines()
            ]
            samples = []
            selected = sorted(
                set(range(0, len(trajectory), settings["task_stride"]))
                | {len(trajectory) - 1}
                | {
                    i
                    for i, r in enumerate(trajectory)
                    if i and r["phase"] != trajectory[i - 1]["phase"]
                }
            )
            for i in selected:
                row = trajectory[i]
                if row["joint_state"]["names"] != order:
                    raise ValueError("Joint ordering mismatch")
                # Both payload states at every target avoids assuming a successful grasp.
                for payload in settings["payload_kg"]:
                    cmd = row["action"]
                    q = [cmd[order.index(n)] for n in names]
                    samples.append(
                        dict(
                            domain="recorded_targets",
                            time_s=row["joint_state"]["time_s"],
                            phase=row["phase"],
                            q=q,
                            payload_kg=payload,
                            gravity_nm=load_at(cmd, payload),
                        )
                    )
            grid = [
                np.linspace(*r, settings[key])
                for r, key in zip(
                    ranges, ["roll_samples", "pitch_samples", "elbow_samples"]
                )
            ]
            base = np.zeros(len(order))
            for q in itertools.product(*grid):
                for j, value in zip(names, q):
                    base[order.index(j)] = value
                for payload in settings["payload_kg"]:
                    samples.append(
                        dict(
                            domain="joint_limit_grid",
                            q=list(q),
                            payload_kg=payload,
                            gravity_nm=load_at(base, payload),
                        )
                    )
            limit = cfg["actuators"]["arm"]["effort_limit_nm"]
            summaries = []
            for spec in settings["candidates"]:
                for domain in ["recorded_targets", "joint_limit_grid"]:
                    subset = [s for s in samples if s["domain"] == domain]
                    residuals = np.array(
                        [s["gravity_nm"] - assist(s["q"][0], spec)[0] for s in subset]
                    )
                    worst = int(np.argmax(np.abs(residuals)))
                    summaries.append(
                        dict(
                            candidate=spec["name"],
                            domain=domain,
                            count=len(subset),
                            within_static_limit_count=int(
                                np.sum(np.abs(residuals) <= limit)
                            ),
                            peak_abs_residual_nm=float(abs(residuals[worst])),
                            rms_residual_nm=float(np.sqrt(np.mean(residuals**2))),
                            worst_sample=subset[worst],
                            worst_signed_residual_nm=float(residuals[worst]),
                        )
                    )
            # At each shoulder angle, any q-only assist is bounded by half the load
            # spread caused by other joints/payloads, irrespective of spring design.
            lower_bounds = []
            for roll in grid[0]:
                values = [
                    s["gravity_nm"]
                    for s in samples
                    if s["domain"] == "joint_limit_grid" and s["q"][0] == roll
                ]
                lower_bounds.append(
                    dict(
                        roll_rad=float(roll),
                        min_nm=min(values),
                        max_nm=max(values),
                        best_possible_peak_nm=(max(values) - min(values)) / 2,
                    )
                )
            cases.append(
                dict(
                    robot=name,
                    profile=profile,
                    mass_model=report["mass_model"],
                    episode_assurance=assurance,
                    limit_nm=limit,
                    summaries=summaries,
                    q_only_lower_bounds=lower_bounds,
                    samples=samples,
                )
            )
            print(
                name,
                profile,
                [
                    (s["candidate"], round(s["peak_abs_residual_nm"], 3))
                    for s in summaries
                    if s["domain"] == "recorded_targets"
                ],
                flush=True,
            )
    energy_errors = []
    for spec in settings["candidates"]:
        if spec["kind"] != "tension":
            continue
        for q in np.linspace(-1.3, 1.3, 101):
            derivative = (
                tension_energy(q + 1e-6, spec) - tension_energy(q - 1e-6, spec)
            ) / 2e-6
            energy_errors.append(abs(assist(q, spec)[0] + derivative))
    if max(energy_errors) > 1e-8 or max(independent_errors) > 1e-10:
        raise ValueError("Independent load/sign check failed")
    write(output / "data.json", dict(settings=settings, cases=cases))
    paths = [Path(__file__).relative_to(Path.cwd()), manifest, Path("lab/recording.py")]
    result = dict(
        status="partial",
        scope="Quasistatic passive-mechanism screening, not dynamics or hardware qualification",
        source_bindings=bindings,
        source_hashes=copy_source(output, {str(p): p for p in paths}),
        output_hashes={"data.json": sha(output / "data.json")},
        point_jacobian_max_error_nm=max(independent_errors),
        energy_gradient_max_error_nm=max(energy_errors),
        candidate_geometry="not_run",
        candidate_dynamics="not_run",
        physical="not_run",
        cases=[{k: v for k, v in c.items() if k != "samples"} for c in cases],
        limitations=[
            "Grid includes colliding configurations and is not qualified reachable workspace.",
            "Task samples are commanded poses, not equilibrium or actual achieved poses; both ideal payload states are applied at each sample.",
            "Only left shoulder roll with supported upright torso is analyzed. Right-arm mirroring and free-base orientation are not qualified.",
            "Spring mass, mounting, friction, hysteresis, preload tolerances and extra hardware are absent. Original clamps remain installed.",
            "Payload hand-local point is a transported hypothesis from the earlier original upper-mass observation, not a newly measured grasp.",
            "Static residual omits inertia, contact forces and actuator speed/voltage/temperature behavior. No production quote or savings claimed.",
        ],
    )
    write(output / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
