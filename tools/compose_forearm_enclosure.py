"""Compose experimental forearm enclosures and replay sampled full-body CAD poses.

Only new material against retained parts and opposite-arm additions is checked.
Known producer failures remain failures; sampled task clearance is separate from
dynamic task success, continuous collision detection and manufacturing evidence.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import Color, Compound, export_step, import_step

from lab.artifact_store import verify_dependency
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.check_forearm_cover import poses


def verify_report(folder):
    seal = folder / "report.sha256"
    if seal.exists() and seal.read_text().strip() != sha(folder / "report.json"):
        raise ValueError(f"report checksum differs: {folder}")
    report = json.loads((folder / "report.json").read_text())
    for name, digest in report.get("source_hashes", {}).items():
        if sha(folder / "source" / name) != digest:
            raise ValueError(f"archived source changed: {folder}/{name}")
    for name, digest in report.get("output_hashes", {}).items():
        if sha(folder / name) != digest:
            raise ValueError(f"output changed: {folder}/{name}")
    for name, digest in report.get("source_bindings", {}).items():
        if sha(Path(name)) != digest:
            raise ValueError(f"bound input changed: {name}")
    for ref in report.get("dependency_manifests", {}).values():
        if verify_dependency(ref)["status"] != "pass":
            raise ValueError(f"archived dependency changed: {folder}")
    return report


def kinematic_contract(contract):
    """Remove only mass overrides; all geometry, frames and command units match."""
    result = deepcopy(contract)
    result.pop("mass_model", None)
    for link in result["links"]:
        link.pop("mass", None)
        link.pop("mass_properties", None)
    return result


def task_samples(rows, contract, interval=0.4):
    """Fixed-time sampling plus both endpoints of every phase, including failures."""
    if not rows or interval <= 0:
        raise ValueError("nonempty trajectory and positive interval required")
    last_time = -float("inf")
    selected = {0, len(rows) - 1}
    next_sample = rows[0]["joint_state"]["time_s"] + interval
    for index, row in enumerate(rows):
        state = row["joint_state"]
        values = state["position_rad"]
        time = state["time_s"]
        if (
            state["names"] != contract["joint_order"]
            or len(values) != len(state["names"])
            or not np.isfinite(values).all()
            or not np.isfinite(time)
            or time <= last_time
        ):
            raise ValueError("invalid trajectory time, joint order or position")
        if time >= next_sample:
            selected.add(index)
            next_sample = time + interval
        if index and row["phase"] != rows[index - 1]["phase"]:
            selected.update([index - 1, index])
        last_time = time
    return [
        dict(
            name=f"task-{i}-{rows[i]['phase']}",
            kind="task",
            source_row=i,
            phase=rows[i]["phase"],
            time_s=rows[i]["joint_state"]["time_s"],
            pose=dict(
                zip(
                    contract["joint_order"],
                    rows[i]["joint_state"]["position_rad"],
                    strict=True,
                )
            ),
        )
        for i in sorted(selected)
    ]


def probe_targets(source, retained, probes, baseline_spines):
    # Replacing a spine must not erase its old material as a target for the
    # other arm. Added-stock probes cover only its increment. Same-owner seats
    # are covered by the separately retained producer installation checks.
    return (
        list(retained)
        + [p for p in baseline_spines if p.owner != source.owner]
        + [p for p in probes if p.owner != source.owner and p.name > source.name]
    )


def run(assembly, cover, preview, episode, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail", builder="experimental_forearm_composition", scope=__doc__.strip()
    )
    try:
        # Snapshot before doing any CAD work, not after a potentially long run.
        sources = {
            str(p): p for p in [*Path("lab").glob("*.py"), *Path("tools").glob("*.py")]
        }
        for label, folder, names in [
            (
                "assembly",
                assembly,
                [
                    "report.json",
                    "assembly.step",
                    "interfaces.json",
                    "resolved-config.json",
                ],
            ),
            ("cover", cover, ["report.json", "resolved-config.json"]),
            (
                "preview",
                preview,
                ["report.json", "part-frames.json", "posed-cover-parts.step"],
            ),
            (
                "episode",
                episode,
                ["report.json", "interfaces.json", "task.json", "trajectory.jsonl"],
            ),
        ]:
            sources.update({f"{label}/{name}": folder / name for name in names})
        report["source_hashes"] = copy_source(output, sources)
        base = verify_report(assembly)
        candidate = verify_report(cover)
        overlay = verify_report(preview)
        if overlay["status"] != "rendered":
            raise ValueError("completed visual preview required")
        if base["status"] != "pass" or not candidate["step_identity_pass"]:
            raise ValueError("passed base and cover STEP identity required")
        if not all(c["status"] == "pass" for c in candidate["checks"]):
            raise ValueError("cover attachment/stock checks do not pass")
        for folder in [assembly, cover]:
            for name in ["report.json", "resolved-config.json"]:
                if overlay["source_bindings"].get(str(folder / name)) != sha(
                    folder / name
                ):
                    raise ValueError("preview not bound to requested assembly/cover")
        config = json.loads((assembly / "resolved-config.json").read_text())
        if config != json.loads((cover / "resolved-config.json").read_text()):
            raise ValueError("cover configuration differs")
        for name in ["lab/config.py", "lab/robot.py", "lab/leg_mechanism.py"]:
            if sha(Path(name)) != candidate["source_hashes"][name]:
                raise ValueError(f"kinematic dependency changed: {name}")
        robot = make_robot(RobotConfig.model_validate(config))
        contract = json.loads((assembly / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != contract:
            raise ValueError("base interface differs from current kinematics")
        task_contract = json.loads((episode / "interfaces.json").read_text())
        if kinematic_contract(task_contract) != kinematic_contract(contract):
            raise ValueError("task joint/frame/geometry contract differs")
        if json.loads((episode / "task.json").read_text())["fixture"] != "supported":
            raise ValueError("only supported-torso task replay implemented")
        neutral = robot.forward_kinematics()
        inverse = {name: np.linalg.inv(frame) for name, frame in neutral.items()}
        original = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        colors = {p["name"]: tuple(p["color"]) for p in base["parts"]}
        original = [replace(p, color=colors[p.name]) for p in original]
        metadata = json.loads((preview / "part-frames.json").read_text())
        if len(metadata["step_roundtrip"]) != 22 or not all(
            geometry_agrees(c["actual"], c["expected"])
            for c in metadata["step_roundtrip"]
        ):
            raise ValueError("preview STEP identity checks do not pass")
        if (
            metadata["step_coordinates"] != "neutral world"
            or metadata["step_units"] != "mm"
            or metadata["transform_translation_units"] != "m"
        ):
            raise ValueError("unsupported cover frame convention")
        if set(metadata["replaces"]) != {"left_forearm_spine", "right_forearm_spine"}:
            raise ValueError("unexpected replacement set")
        for p in metadata["parts"]:
            if not np.allclose(
                p["neutral_owner_to_world"], neutral[p["owner"]], atol=1e-12, rtol=0
            ):
                raise ValueError("cover placement frame differs")
        records = [dict(p, module="forearm_enclosure") for p in metadata["parts"]]
        additions = parts_from_step(
            list(import_step(preview / "posed-cover-parts.step").children), records
        )
        additions = [
            replace(
                p, color=(0.35, 0.65, 0.58) if p.role == "shell" else (0.5, 0.5, 0.5)
            )
            for p in additions
        ]
        if len(additions) != 22 or len({p.name for p in additions}) != 22:
            raise ValueError("expected 22 unique replacement/addition groups")
        old_by_name = {p.name: p for p in original}
        retained = [p for p in original if p.name not in metadata["replaces"]]
        if len(retained) != len(original) - 2 or {p.name for p in additions} & {
            p.name for p in retained
        }:
            raise ValueError("ambiguous replacement inventory")
        combined = retained + additions
        # A replacement spine includes previously qualified mounting contacts.
        # Probe only new stock against retained parts, never blanket-exempt it.
        probes = [
            replace(p, shape=p.shape - old_by_name[p.name].shape)
            if p.role == "spine"
            else p
            for p in additions
        ]
        baseline_spines = [
            replace(old_by_name[name], name=f"baseline_{name}")
            for name in metadata["replaces"]
        ]
        if any(not p.shape.is_valid or p.shape.volume <= 0 for p in probes):
            raise ValueError("invalid/empty added-material probe")
        for p in combined:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        print("exporting complete assembly", flush=True)
        export_step(
            Compound(children=[p.shape for p in combined]), output / "assembly.step"
        )
        restored = parts_from_step(
            list(import_step(output / "assembly.step").children),
            [
                dict(name=p.name, owner=p.owner, role=p.role, module=p.module)
                for p in combined
            ],
        )
        restore_by_name = {p.name: p.shape for p in restored}
        comparisons = [
            dict(
                name=p.name,
                expected=geometry_record(p.shape),
                actual=geometry_record(restore_by_name[p.name]),
            )
            for p in combined
        ]
        roundtrip = all(
            geometry_agrees(c["actual"], c["expected"]) for c in comparisons
        )
        write(output / "step-roundtrip.json", comparisons)
        if not roundtrip:
            raise ValueError("combined assembly STEP roundtrip failed")
        report.update(
            part_count=len(combined),
            parts=[
                dict(
                    name=p.name,
                    owner=p.owner,
                    role=p.role,
                    module=p.module,
                    color=p.color,
                    volume_mm3=p.shape.volume,
                )
                for p in combined
            ],
            assembly_step_roundtrip="pass",
            cover_producer_status=candidate["status"],
            cover_producer_failed_poses=sorted(
                {r["pose"] for r in candidate["rows"] if r["status"] == "fail"}
            ),
            same_owner_addition_contacts="Inherited seating/insert/split evidence from cover producer; not treated as free-space clearance",
            added_stock_probe_volumes_mm3={
                p.name: p.shape.volume for p in probes if p.role == "spine"
            },
        )
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        samples = [
            dict(name=name, kind="design", pose=pose) for name, pose in poses(robot)
        ]
        samples += task_samples(rows, contract)
        report["task_sampling"] = dict(
            source_rows=len(rows),
            interval_s=0.4,
            selection="first, last, first/last of every phase, then first row at least 0.4s after previous temporal sample",
            selected_rows=[s["source_row"] for s in samples if s["kind"] == "task"],
        )
        results = []
        for sample in samples:
            frames = robot.forward_kinematics(sample["pose"], enforce_limits=False)
            moved = {
                p.name: replace(
                    p,
                    shape=transform_shape(p.shape, frames[p.owner] @ inverse[p.owner]),
                )
                for p in retained + probes + baseline_spines
            }
            query = CameraQueries()
            failures = []
            minimum = (float("inf"), "", "")
            for p in probes:
                source = moved[p.name]
                targets = [
                    moved[t.name]
                    for t in probe_targets(p, retained, probes, baseline_spines)
                ]
                hits = query.collide(source.shape, targets)
                distance, nearest = query.nearest(source.shape, targets)
                minimum = min(minimum, (distance, p.name, nearest))
                if hits or distance < 0.5 - 1e-7:
                    failures.append(
                        dict(
                            part=p.name,
                            hits=hits,
                            distance_mm=distance,
                            nearest=nearest,
                        )
                    )
            excursions = [
                dict(
                    joint=j.name,
                    position_rad=sample["pose"].get(j.name, 0),
                    limits_rad=j.limits,
                )
                for j in robot.joints
                if not j.continuous
                and not j.limits[0] <= sample["pose"].get(j.name, 0) <= j.limits[1]
            ]
            results.append(
                dict(
                    **sample,
                    status="fail" if failures else "pass",
                    failures=failures,
                    minimum_distance_mm=minimum[0],
                    closest_pair=minimum[1:],
                    measured_joint_limit_excursions=excursions,
                )
            )
            write(output / "progress.json", results)
            print(sample["name"], results[-1]["status"], len(failures), flush=True)
        report.update(
            status="pass"
            if candidate["status"] == "pass"
            and all(r["status"] == "pass" for r in results)
            else "fail",
            sampled_clearance_status="pass"
            if all(r["status"] == "pass" for r in results)
            else "fail",
            task_sampled_clearance_status="pass"
            if all(r["status"] == "pass" for r in results if r["kind"] == "task")
            else "fail",
            samples=results,
            clearance_target_mm=0.5,
            overlap_threshold_mm3=0.02,
            joint_limit_scope="Geometric clearance evaluated at actual recorded poses; excursions reported separately",
            episode_source_status=json.loads((episode / "report.json").read_text())[
                "status"
            ],
            not_run=[
                "continuous/swept clearance",
                "old-old assembly self-collision",
                "table/block/tray contact geometry",
                "new mass or dynamics",
                "harness routing",
                "strength and manufacturing qualification",
            ],
        )
        write(output / "resolved-config.json", config)
        write(output / "interfaces.json", contract)
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name not in {"report.json", "report.sha256"}
    }
    write(output / "report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "error",
                    "part_count",
                    "task_sampled_clearance_status",
                ]
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["assembly", "cover", "preview", "episode", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(**vars(args))
    raise SystemExit(int(result["status"] != "pass"))
