"""Build a source-bound local review across assembly, benches, learning and BOM."""

import argparse
from collections import Counter
from datetime import date
import hashlib
import html
import json
from pathlib import Path

from lab.artifact_store import snapshot_file
from tools.check_camera_support import sha, write


ROOT = Path(__file__).resolve().parents[1]


def copy_review_asset(source, target, content):
    """Share immutable review assets; adapted HTML remains a new ordinary file."""
    if target.exists():
        raise FileExistsError(target)
    if content == source.read_bytes():
        snapshot_file(source, target)
    else:
        target.write_bytes(content)
    if sha(target) != hashlib.sha256(content).hexdigest():
        raise ValueError(f"Review asset copy differs: {target}")


def part_ids(entry, robot):
    return entry.get("part_ids_by_robot", {}).get(robot, entry.get("part_ids", []))


def validate_inventory(bom, robots):
    counts = {}
    for robot in robots:
        name = robot["name"].lower()
        inventory = {p["name"] for p in robot["parts"]}
        mapped = Counter(p for e in bom["entries"] for p in part_ids(e, name))
        if set(mapped) != inventory or any(n != 1 for n in mapped.values()):
            raise ValueError(
                f"{name} BOM inventory mismatch: missing={inventory - mapped.keys()}, "
                f"extra={mapped.keys() - inventory}, duplicated="
                f"{[p for p, n in mapped.items() if n != 1]}"
            )
        counts[name] = len(inventory)
    return counts


def wrapper(fragment, *, selection=False):
    bridge = (
        """
<script>
(() => {
 const root=document.getElementById('head-restored-motion-inspector') || document.getElementById('distal-grip-motion-inspector');
 const send=()=>parent.postMessage({type:'robot-part-selection',
   robot:root.querySelector('[data-id=robot]').value,
   part:root.querySelector('[data-id=part]').value},location.origin);
 new MutationObserver(send).observe(root.querySelector('[data-id=detail]'),
   {childList:true,subtree:true,characterData:true});
})();
</script>"""
        if selection
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Robot inspection</title><link rel="stylesheet" href="review.css">'
        '</head><body class="viewer">' + fragment + bridge + "</body></html>"
    )


def run(output, view):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(path):
        p = ROOT / path
        if not p.is_file() or not p.resolve().is_relative_to(ROOT):
            raise ValueError(f"Missing or external review source: {path}")
        data = p.read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return data.decode()

    def load(path):
        return json.loads(read(path))

    bom = load("robot/prototype-bom.json")
    validation = load("robot/validation-map.json")
    verification = load("robot/part-verification.json")
    display = load(str(view / "dataset.json"))
    display_report = load(str(view / "report.json"))
    fragment = read(str(view / "motion.fragment.html"))
    for filename in ("dataset.json", "motion.fragment.html"):
        if sources[str(view / filename)] != display_report["output_hashes"][filename]:
            raise ValueError("Display asset hash mismatch")
    for path, digest in display_report["source_bindings"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Display binding changed: {path}")
    robots = []
    for name in ("pip", "scout"):
        assembly_path = f"artifacts/{name}-distal-grip-integrated-v1/report.json"
        motion_path = f"artifacts/{name}-distal-grip-park-replay-v1/report.json"
        task_path = f"artifacts/{name}-distal-grip-park-task-v1/report.json"
        assembly, motion, task = map(load, (assembly_path, motion_path, task_path))
        if motion["source_hashes"][assembly_path] != sources[assembly_path]:
            raise ValueError("Motion assembly binding mismatch")
        if motion["source_hashes"][task_path] != sources[task_path]:
            raise ValueError("Motion task binding mismatch")
        # The review's stated improvement must agree with both retained runs.
        if motion["added_material_self_status"] != "pass" or task["status"] != "pass":
            raise ValueError("Review narrative requires updating for changed outcome")
        rendered = next(r for r in display["robots"] if r["name"].lower() == name)
        if rendered["source"] != str(Path(motion_path).parent):
            raise ValueError("Viewer is showing a different experiment")
        flagged = Counter(p for s in rendered["samples"] for p in s["flagged"])
        robots.append(
            dict(
                name=name.capitalize(),
                parts=assembly["parts"],
                task=task,
                task_settings=load(f"artifacts/distal-grip-park-tasks-v1/{name}.json"),
                motion_path=motion_path,
                flagged_parts=dict(flagged),
            )
        )
    coverage = validate_inventory(bom, robots)
    all_ids = {p["name"] for r in robots for p in r["parts"]}
    for group in verification["groups"]:
        ids = set(group.get("part_ids", []))
        ids.update(
            p
            for p in all_ids
            if any(p.startswith(prefix) for prefix in group.get("part_prefixes", []))
        )
        if ids - all_ids:
            raise ValueError(f"Unknown verification parts: {ids - all_ids}")
        group["part_ids"] = sorted(ids)
        group["title"] = group.get("title", group.get("role", group["id"]))
    assembly_links = [
        (
            f"artifacts/{n}-distal-grip-park-replay-v1/report.json",
            f"{n.capitalize()} exact motion report",
        )
        for n in ("pip", "scout")
    ] + [
        ("docs/design/distal-grip-rest-pose.md", "Parked-arm clearance study"),
        ("docs/design/forearm-integration.md", "Retained whole-assembly limitations"),
        (
            "docs/design/composed-dynamics.md",
            "Current mass and posture control results",
        ),
    ]
    assembly_links.append(
        (
            "docs/design/balance-wheel-clearance.md",
            "Exact wheel and arm clearance at rest",
        )
    )
    if (view / "head-error.json").is_file():
        assembly_links.extend(
            [
                (str(view / "head-error.json"), "Head display fidelity measurements"),
                (
                    "docs/design/head-display.md",
                    "Display accuracy and remaining head protection",
                ),
            ]
        )
    assembly_links.extend(
        [
            (
                "docs/design/scout-eye-shell.md",
                "Scout rear-open eye shells: appearance, material and limits",
            ),
            (
                "artifacts/scout-eye-shell-v1/report.json",
                "Eye-shell geometry and conditional cost comparison",
            ),
        ]
    )
    eye_report = load("artifacts/scout-eye-shell-v1/report.json")
    if (
        eye_report["geometry_status"] != "pass"
        or abs(eye_report["removed_bilateral_mm3"] - 58188.57912979015) > 1e-5
        or abs(eye_report["assumed_bilateral_mass_reduction_g"] - 73.89949549483349)
        > 1e-5
    ):
        raise ValueError("Eye-shell presentation differs from retained study")
    for eye_asset in ("eye-shells-v4.rrd", "sections-v4.png", "wall-2mm/eyes.step"):
        path = "artifacts/scout-eye-shell-v1/" + eye_asset
        sources[path] = sha(ROOT / path)
    assembly_links.extend(
        [
            (
                "docs/design/scout-eye-x1.md",
                "Paired X1 eye slices: native estimates and physical comparison",
            ),
            (
                "artifacts/scout-eye-x1-v1/audit.json",
                "X1 eye geometry, placement and settings audit",
            ),
        ]
    )
    eye_slice = load("artifacts/scout-eye-x1-v1/audit.json")
    if (
        eye_slice["numerical_checks"] != "pass"
        or round(eye_slice["comparison"]["filament_reduction_percent"], 2) != 17.86
        or round(eye_slice["comparison"]["job_time_reduction_percent"], 2) != 19.93
    ):
        raise ValueError("Eye slice presentation differs from retained native evidence")
    for asset in (
        "solid-slice/solid.sliced.3mf",
        "shell-slice/shell.sliced.3mf",
        "solid-preview.jpg",
        "shell-preview.jpg",
    ):
        path = "artifacts/scout-eye-x1-v1/" + asset
        sources[path] = sha(ROOT / path)
    assembly_links.extend(
        [
            (
                "docs/design/scout-integral-face.md",
                "One-piece Scout face: geometry, process and retained camera failure",
            ),
            (
                "artifacts/scout-integral-face-v1/assessment.json",
                "Face integration checks and carrier-path diagnostic",
            ),
        ]
    )
    face_integration = load("artifacts/scout-integral-face-v1/assessment.json")
    face_geometry = load("artifacts/scout-integral-face-v1/report.json")
    if (
        face_integration["status"] != "partial"
        or not all(face_integration["checks"].values())
        or face_geometry["candidate_printed_piece_count"] != 1
        or face_geometry["original_group_count"] != 7
        or face_geometry["status"] != "fail"
    ):
        raise ValueError("Integrated-face presentation differs from retained evidence")
    for asset in (
        "integral-face-v2.rrd",
        "integral-face.step",
        "integral-face.stl",
        "native-view-v3.png",
    ):
        path = "artifacts/scout-integral-face-v1/" + asset
        sources[path] = sha(ROOT / path)
    assembly_links.extend(
        [
            (
                "docs/design/scout-face-x1.md",
                "Whole-face X1 orientation, supports and physical comparison",
            ),
            (
                "artifacts/scout-face-x1-v1/audit-v4.json",
                "Whole-face native slice and geometry checks",
            ),
        ]
    )
    face_print = load("artifacts/scout-face-x1-v1/audit-v4.json")
    if (
        face_print["numerical_checks"] != "pass"
        or round(face_print["cases"]["eyes-up"]["estimated_total_filament_g"], 2)
        != 118.70
        or round(face_print["cases"]["eyes-down"]["estimated_total_filament_g"], 2)
        != 109.77
    ):
        raise ValueError("Whole-face print presentation differs from native evidence")
    for asset in (
        "eyes-up-slice/eyes-up.sliced.3mf",
        "eyes-down-slice/eyes-down.sliced.3mf",
        "eyes-up-toolpaths.jpg",
        "eyes-down-toolpaths.jpg",
        "native-preview.json",
    ):
        path = "artifacts/scout-face-x1-v1/" + asset
        sources[path] = sha(ROOT / path)
    learning_links = [
        ("docs/design/learning-platform.md", "Learning architecture"),
        ("docs/design/recordings.md", "Recording & Rerun contract"),
        (
            "docs/design/mcap-recordings.md",
            "MCAP exchange: native tools, replay and limits",
        ),
        (
            "artifacts/mcap-roundtrip-v1/report.json",
            "Eight lossless recording exchanges",
        ),
        (
            "artifacts/mcap-roundtrip-v1/validation.json",
            "Recording tests and viewer evidence",
        ),
        ("skills/definitions/supported-place.yaml", "Placement skill definition"),
        ("skills/definitions/wheel-drive-hold.yaml", "Drive skill definition"),
        (
            "artifacts/skill-factory-v1/validation-summary.json",
            "Skill evaluation results",
        ),
        ("docs/design/external-workers.md", "External GPU worker plan"),
        (
            "docs/design/isaac-pilot.md",
            "Prepared Isaac worker: run instructions and unverified stages",
        ),
        (
            "artifacts/covered-worker-bundles-v1/report.json",
            "Current Pip/Scout worker inputs: static checks only",
        ),
        (
            "artifacts/isaac-pilot-validation-v1/relocation-summary.json",
            "Current worker assets after relocation",
        ),
        (
            "artifacts/isaac-mac-guard-v1/report.json",
            "Mac guard: no native GPU stages launched",
        ),
    ]
    clamp_print = load("artifacts/clamp-x1-v1/audit-v2.json")
    if (
        clamp_print["numerical_checks"] != "pass"
        or round(clamp_print["comparison"]["filament_reduction_percent"], 2) != 6.29
        or round(clamp_print["comparison"]["job_time_reduction_percent"], 2) != -10.98
    ):
        raise ValueError("Clamp print comparison differs from native evidence")
    for name in (
        "original-statistics.png",
        "windowed-statistics.png",
        "windowed-bridge-layer9.png",
        "original-slice/original.sliced.3mf",
        "windowed-slice/windowed.sliced.3mf",
    ):
        path = "artifacts/clamp-x1-v1/" + name
        sources[path] = sha(ROOT / path)
    learning_links.extend(
        [
            (
                "docs/design/target-power-sequence.md",
                "Motor startup: corrected request/reply sequence and physical bench",
            ),
            (
                "artifacts/target-power-sequence-v1/report.json",
                "Nine scenarios executing the actual target adapter with synthetic peripherals",
            ),
            (
                "docs/design/firmware-frame-timeout.md",
                "Timed command frames, telemetry and disarmed bench procedure",
            ),
            (
                "artifacts/firmware-frame-timeout-v1/validation.json",
                "Firmware deadline tests, cross-builds and scoped workbench",
            ),
        ]
    )
    interlock = load("artifacts/hardware-arm-rearm-v4/report.json")
    if (
        interlock["status"] != "pass"
        or interlock["physical_status"] != "not_run"
        or len(interlock["cases"]) != 99
        or any(c["status"] != "pass" for c in interlock["cases"])
        or interlock["negative_control"]["status"] != "fail"
        or sum(p["quantity_per_candidate_board"] for p in interlock["component_bom"])
        != 16
    ):
        raise ValueError("Arm interlock presentation differs from retained evidence")
    for path in (
        "artifacts/hardware-arm-rearm-v1/held-stop.png",
        "tools/plot_arm_interlock.py",
    ):
        sources[path] = sha(ROOT / path)
    startup = load("artifacts/target-power-sequence-v1/report.json")
    if (
        startup["status"] != "pass"
        or startup["physical_hardware"] != "not_run"
        or len(startup["cases"]) != 9
    ):
        raise ValueError("Target startup sequence narrative differs")
    for name in ("pip", "scout"):
        build = load(f"artifacts/{name}-power-sequence-v1/build-report.json")
        mapping = load(f"artifacts/{name}-power-sequence-v1/hardware-map.json")
        if (
            build["status"] != "pass"
            or mapping["calibrated"]
            or mapping["board"]["motor_power_settle_ms"] != 100
        ):
            raise ValueError("Target startup build or calibration narrative differs")
    frame_validation = load("artifacts/firmware-frame-timeout-v1/validation.json")
    if (
        frame_validation["software_status"] != "pass"
        or frame_validation["physical_status"] != "not_run"
    ):
        raise ValueError("Frame timeout presentation differs from retained validation")
    for folder in (
        "pip-frame-timeout-firmware-v1",
        "scout-frame-timeout-firmware-v2",
        "moss-frame-timeout-firmware-v1",
    ):
        for asset in ("robot_controller.uf2", "hardware-map.json", "build-report.json"):
            path = "artifacts/" + folder + "/" + asset
            sources[path] = sha(ROOT / path)
    spacer_slice = load("artifacts/spacer-x1-v1/audit-v2.json")
    spacer_clouds = load("artifacts/spacer-x1-v1/cloud-distance.json")
    if (
        spacer_slice["numerical_status"] != "pass"
        or spacer_clouds["status"] != "pass"
        or len(spacer_slice["objects"]) != 30
        or round(spacer_slice["estimated_filament_g"], 2) != 2.14
    ):
        raise ValueError("Spacer print presentation differs from native evidence")
    for asset in (
        "slice/comparison.sliced.3mf",
        "native-first-layer.png",
        "native-statistics.png",
    ):
        path = "artifacts/spacer-x1-v1/" + asset
        sources[path] = sha(ROOT / path)
    for robot_name in ("pip", "scout"):
        folder = "artifacts/" + robot_name + "-configured-carriers-v2"
        configured = load(folder + "/report.json")
        if (
            configured["geometry_and_mass_status"] != "pass"
            or configured["physical_status"] != "not_run"
            or configured["spacer_solids"] != 40
        ):
            raise ValueError("Configured carrier evidence differs from presentation")
        for asset in (
            "nominal/mass-properties.json",
            "nominal/robot.xml",
            "nominal/robot.urdf",
        ):
            path = folder + "/" + asset
            sources[path] = sha(ROOT / path)
    battery_fit = load("artifacts/battery-packaging-v2/report.json")
    battery_power = load("artifacts/battery-current-budget-v1/report.json")
    if (
        battery_fit["execution_status"] != "pass"
        or battery_power["execution_status"] != "pass"
    ):
        raise ValueError("Battery study execution incomplete")
    for robot in ("pip", "scout"):
        states = {c["case"]: c["status"] for c in battery_fit["robots"][robot]["cases"]}
        if (
            states["rrc2040-crosswise"] != "pass"
            or states["rrc2040-lengthwise"] != "fail"
        ):
            raise ValueError("Battery geometry narrative differs")
    rows = {r["name"]: r for r in battery_power["rows"]}
    for case, value in (
        ("rated-sum-camera-nominal", 15.27),
        ("rated-sum-camera-probe", 15.82),
    ):
        if (
            rows[case]["electrical_status"] != "fail"
            or abs(rows[case]["simulated_peak_pack_current_a"] - value) > 0.005
        ):
            raise ValueError("Battery current narrative differs")
    shared_power = load("artifacts/shared-battery-v1/report.json")
    shared_checks = load("artifacts/shared-battery-validation-v1/report.json")
    if shared_power["execution_status"] != "pass" or shared_checks["status"] != "pass":
        raise ValueError("Shared battery evidence incomplete")
    shared_rows = {r["name"]: r for r in shared_power["rows"]}
    for case, value in (("rated-camera-nominal", 15.35), ("rated-camera-probe", 15.95)):
        row = shared_rows[case]
        if (
            row["electrical_status"] != "fail"
            or abs(row["peak_battery_a"] - value) > 0.005
        ):
            raise ValueError("Shared battery current narrative differs")
    if abs(shared_rows["low-eight-amp"]["minimum_motor_v"] - 8.76) > 0.005:
        raise ValueError("Shared battery voltage narrative differs")
    shared_plot = "artifacts/shared-battery-validation-v1/shared-battery.png"
    sources[shared_plot] = sha(ROOT / shared_plot)
    saddle_reports = [
        load(f"artifacts/capacitor-saddle-v2/{name}/report.json")
        for name in ("pip", "scout")
    ]
    for result in saddle_reports:
        if result["execution_status"] != "pass" or result["neutral_status"] != "pass":
            raise ValueError("Capacitor saddle fit narrative differs")
        for evidence, samples in (
            (result["capacitor_removal"], 105),
            (result["board_and_tray_removal"]["module"], 139),
            (result["board_and_tray_removal"]["tray"], 75),
        ):
            if evidence["status"] != "pass" or evidence["samples"] != samples:
                raise ValueError("Capacitor saddle service narrative differs")
    for path in (
        "artifacts/capacitor-saddle-v2/detail.png",
        "artifacts/capacitor-saddle-v2/context.jpg",
    ):
        sources[path] = sha(ROOT / path)
    terminals = load("artifacts/regen-terminal-v2/report.json")
    if (
        terminals["execution_status"] != "pass"
        or terminals["physical_status"] != "not_run"
    ):
        raise ValueError("Terminal study execution or physical status differs")
    if (
        terminals["local"]["side-forward"]["body"]
        or not terminals["local"]["side-forward"]["allowance"]
    ):
        raise ValueError("Terminal nominal/reserved clearance narrative differs")
    for name in ("pip", "scout"):
        result = terminals["robots"][name]
        if result["neutral_status"] != "pass":
            raise ValueError("Terminal nominal robot fit narrative differs")
        if not all(
            result["driver_probes"]["rear_open_carrier_battery_removed"].values()
        ):
            raise ValueError("Terminal driver obstruction narrative differs")
        for key, count in (("capacitor", 105), ("module_with_terminal", 139)):
            if (
                result["removal"][key]["status"] != "pass"
                or result["removal"][key]["samples"] != count
            ):
                raise ValueError("Terminal removal narrative differs")
    for path in (
        "artifacts/regen-terminal-v2/comparison.jpg",
        "artifacts/regen-terminal-v2/tool-access.jpg",
    ):
        sources[path] = sha(ROOT / path)
    disconnect = load("artifacts/motor-disconnect-v2/report.json")
    if (
        disconnect["status"] != "partial"
        or disconnect["execution_status"] != "pass"
        or disconnect["physical_status"] != "not_run"
        or disconnect["pin_check"]["status"] != "pass"
        or disconnect["divider_model_agreement"] != "pass"
        or len(disconnect["divider_corners"]) != 64
    ):
        raise ValueError("Motor disconnect scope or static-check narrative differs")
    if (
        sum(row["quantity_per_candidate_board"] for row in disconnect["component_bom"])
        != 14
    ):
        raise ValueError("Motor disconnect component count differs")
    if any(
        abs(a - b) > 0.0001
        for a, b in zip(disconnect["ov_range_v"]["rising"], [12.6577, 13.8487])
    ):
        raise ValueError("Motor disconnect cutoff narrative differs")
    bom_links = [
        (
            "artifacts/hardware-arm-rearm-v4/report.json",
            "Independent arm: compiled circuit and 99 native functional cases",
        ),
        (
            "docs/design/arm-interlock.md",
            "Independent arm: candidate semantics, alternatives and unresolved electrical behavior",
        ),
        (
            "docs/design/arm-interlock-bench-template.json",
            "Independent arm: blank physical measurement record",
        ),
        (
            "artifacts/motor-disconnect-v2/report.json",
            "Motor disconnect: compiled circuit, static checks and candidate component BOM",
        ),
        (
            "docs/design/motor-disconnect-bench-template.json",
            "Motor disconnect: blank physical measurement record",
        ),
        (
            "docs/design/regen-terminal.md",
            "Braking-board connection: clearance and tool-access failures retained",
        ),
        (
            "electronics/regen-terminal-bom.json",
            "Connector alternatives: small-pack references, production pricing unresolved",
        ),
        (
            "artifacts/regen-terminal-v2/report.json",
            "Terminal envelopes, both robots and sampled service checks",
        ),
        (
            "docs/design/capacitor-saddle.md",
            "Capacitor saddle: retained obstruction, revised clearance and physical bench",
        ),
        (
            "electronics/capacitor-saddle-bom.json",
            "Capacitor saddle and exact cable-tie volume tiers",
        ),
        (
            "artifacts/capacitor-saddle-v2/pip/report.json",
            "Pip saddle: nominal fit and three sampled removal routes",
        ),
        (
            "artifacts/capacitor-saddle-v2/scout/report.json",
            "Scout saddle: nominal fit and three sampled removal routes",
        ),
        (
            "docs/design/regen-nut-pockets-x1.md",
            "Crossbeam nut-pocket X1 fit job: three widths, physical fit untested",
        ),
        (
            "docs/design/regen-nut-pockets.md",
            "Crossbeam nut-pocket geometry and remaining strength limits",
        ),
        ("docs/design/regen-mount.md", "Braking-board tray and service sequence"),
        (
            "docs/design/motor-capacitor.md",
            "Motor capacitor electrical sensitivity and bench selection",
        ),
        (
            "docs/design/shared-battery.md",
            "Shared battery: camera-induced sag, current limits and physical benches",
        ),
        (
            "artifacts/shared-battery-v1/report.json",
            "Eight native shared-source cases with electrical failures retained",
        ),
        (
            "artifacts/shared-battery-validation-v1/report.json",
            "Independent current/energy fault detection and operating boundaries",
        ),
        (
            "docs/design/battery-packaging.md",
            "Battery packaging and camera-inclusive current deficit",
        ),
        (
            "artifacts/battery-packaging-v2/report.json",
            "Battery body clearance: crosswise passes, lengthwise fails",
        ),
        (
            "artifacts/battery-current-budget-v1/report.json",
            "Native power replay: camera load exceeds the rated-sum allowance",
        ),
        (
            "docs/design/cradle-x1.md",
            "Corrected cradle X1 slice: native preview, estimates and physical fit record",
        ),
        (
            "artifacts/cradle-x1-v1/audit.json",
            "Corrected cradle native geometry and settings audit",
        ),
        (
            "docs/design/cradle-fasteners.md",
            "Corrected cradle: bolt paths, nominal hardware and fixture limits",
        ),
        (
            "artifacts/cradle-stack-v1/report.json",
            "Cradle hardware envelope checks and remaining fixture limits",
        ),
        (
            "artifacts/cradle-fasteners-v1/geometry-delta.json",
            "Cradle correction and unchanged robot clamp geometry",
        ),
        (
            "docs/design/clamp-x1.md",
            "X1 clamp comparison: lower filament, longer print time",
        ),
        (
            "artifacts/clamp-x1-v1/audit-v2.json",
            "Twelve clamp specimens: native slice and geometry checks",
        ),
        (
            "artifacts/clamp-x1-v1/geometry-distance.json",
            "Clamp geometry identity and shifted controls",
        ),
        (
            "docs/design/covered-carriers.md",
            "Covered robots with carriers: inventory, control and production limits",
        ),
        (
            "artifacts/covered-carriers-clearance-v2/report.json",
            "Carriers versus shin parts at ten fixed poses",
        ),
        (
            "artifacts/covered-carriers-integration-v1/control-comparison-all-profiles.json",
            "All three mass profiles: unchanged control inputs and outcomes",
        ),
        (
            "artifacts/covered-carriers-cost-v2/report.json",
            "Combined candidate inventory and production estimate",
        ),
        (
            "artifacts/pip-covered-carriers-v2/report.json",
            "Pip covered-carrier geometry and mass",
        ),
        (
            "artifacts/scout-covered-carriers-v2/report.json",
            "Scout covered-carrier geometry and mass",
        ),
        (
            "artifacts/pip-covered-carriers-balance-v2/balance-report.json",
            "Pip nominal carrier balance and fault results",
        ),
        (
            "artifacts/scout-covered-carriers-balance-v2/balance-report.json",
            "Scout nominal carrier balance and fault results",
        ),
        (
            "docs/design/configured-carriers.md",
            "Configured spacer CAD, matching planning mass and model boundaries",
        ),
        ("variants/pip-carriers.yaml", "Pip experimental spacer configuration"),
        ("variants/scout-carriers.yaml", "Scout experimental spacer configuration"),
        (
            "docs/design/spacer-x1.md",
            "X1 spacer comparison: native slice and physical handling trial",
        ),
        (
            "artifacts/spacer-x1-v1/audit-v2.json",
            "Thirty spacer specimens: geometry, placement and native settings",
        ),
        ("docs/design/first-fit-bench.md", "First fit bench and print quantities"),
        (
            "docs/design/first-fit-slicing.md",
            "Native slicer jobs, previews and provisional print estimates",
        ),
        (
            "docs/design/x1-first-fit.md",
            "Historical X1 slice: old cradle requires replacement",
        ),
        (
            "docs/research/logic-capacitor-refresh.md",
            "Logic-board capacitor selection and remaining release checks",
        ),
        ("docs/design/forearm-fastener-bom.md", "Exact forearm fastener allocation"),
        ("docs/design/prototype-procurement.md", "Procurement plan"),
        ("robot/prototype-bom.json", "Machine-readable BOM"),
        ("docs/research/integrated-production-bom.md", "Production cost assumptions"),
        (
            "docs/research/actuator-volume-evidence-2026-09-20.md",
            "Actuator retail tiers and missing production quotes",
        ),
        ("docs/design/validation-and-bench-plan.md", "Bench plan"),
    ]
    skirt_folder = Path("artifacts/neck-skirt-review-v3")
    skirt_report = load(str(skirt_folder / "report.json"))
    if skirt_report["status"] != "pass":
        raise ValueError("Neck study presentation failed")
    for path in sorted((ROOT / skirt_folder).rglob("*")):
        if not path.is_file() or path.name not in (
            "index.html",
            "pose.png",
            "front.stl",
            "rear.stl",
        ):
            continue
        relative = path.relative_to(ROOT / skirt_folder)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != skirt_report["output_hashes"].get(str(relative)):
            raise ValueError("Neck study presentation changed")
        sources[str(skirt_folder / relative)] = digest
        target = output / "neck-study" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "index.html":
            content = content.replace(
                b"Nominal collision/service selection: None mm.",
                b"No candidate meets the motion and clearance criteria.",
            )
        copy_review_asset(path, target, content)
    for cap_folder, destination in [
        (Path("artifacts/head-cap-review-v4"), "head-cap-study"),
        (Path("artifacts/head-cap-inspector-v4"), "cap-inspector"),
        (Path("artifacts/head-cap-mount-review-v4"), "mount-inspector"),
        (Path("artifacts/integral-head-wings-review-v2"), "integral-head-inspector"),
        (Path("artifacts/integral-head-service-review-v2"), "head-service-inspector"),
        (Path("artifacts/head-bridge-service-review-v3"), "head-bridge-inspector"),
        (Path("artifacts/head-bridge-carrier-review-v5"), "head-carrier-inspector"),
    ]:
        cap_report = load(str(cap_folder / "report.json"))
        if cap_report["status"] != "pass":
            raise ValueError("Head cap presentation failed")
        for relative in cap_report["presentation_files"]:
            rel = Path(relative)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Head cap presentation path escapes package")
            path = ROOT / cap_folder / rel
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            if digest != cap_report["output_hashes"].get(relative):
                raise ValueError("Head cap presentation changed")
            sources[str(cap_folder / rel)] = digest
            target = output / destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_review_asset(path, target, content)
    combined_view = load("artifacts/covered-carriers-composed-review-v1/report.json")
    guard = load("artifacts/upper-arm-guard-v4/report.json")
    guard_view = load("artifacts/upper-arm-guard-motion-review-v3/report.json")
    guard_range = load("artifacts/upper-arm-guard-range-v1/report.json")
    for path in ("lab/upper_arm_guard.py", "robot/upper-arm-guard-fold-study.yaml"):
        if guard_range["source_hashes"].get(path) != guard["source_hashes"][path]:
            raise ValueError("Upper-arm range check uses different guard geometry inputs")
    if (
        guard_view["status"] != "pass"
        or guard_view["source_bindings"].get("artifacts/upper-arm-guard-v4/report.json")
        != sha(ROOT / "artifacts/upper-arm-guard-v4/report.json")
        or set(guard["robots"]) != {"pip", "scout"}
    ):
        raise ValueError("Upper-arm guard presentation differs from checked study")
    for name in ("pip", "scout"):
        path = f"artifacts/{name}-covered-carriers-composed-v2/report.json"
        if guard["source_bindings"].get(path) != sha(ROOT / path):
            raise ValueError("Upper-arm guard used a different whole assembly")
        motion_path = f"artifacts/{name}-upper-guard-motion-v3/report.json"
        motion = load(motion_path)
        if (
            motion["robot"] != name
            or motion["source_bindings"].get("artifacts/upper-arm-guard-v4/report.json")
            != sha(ROOT / "artifacts/upper-arm-guard-v4/report.json")
            or guard_view["source_bindings"].get(motion_path) != sha(ROOT / motion_path)
        ):
            raise ValueError("Upper-arm guard motion presentation source mismatch")
    for path in (
        "artifacts/upper-arm-guard-motion-review-v3/index.html",
        "artifacts/upper-arm-guard-range-v1/termination.json",
        "artifacts/upper-arm-guard-range-baseline-v1/report.json",
        "docs/design/upper-arm-guard.md",
    ):
        read(path)
    combined_cost = load("artifacts/covered-carriers-production-review-v1/report.json")
    if combined_view["status"] != "pass" or combined_cost["status"] != "pass":
        raise ValueError("Combined candidate presentation did not pass")
    for name, groups, pieces in (("pip", 234, 116), ("scout", 240, 118)):
        path = f"artifacts/{name}-covered-carriers-composed-v2/report.json"
        complete = load(path)
        if (
            complete["geometry_build_status"] != "pass"
            or complete["assembly_step_roundtrip"] != "pass"
            or complete["part_count"] != groups
            or complete["piece_counts"]["candidate_printed_solid_count"] != pieces
            or complete["cost_inventory_agreement"]["status"] != "pass"
            or combined_view["source_bindings"].get(path) != sha(ROOT / path)
            or combined_cost["source_bindings"].get(path) != sha(ROOT / path)
        ):
            raise ValueError("Combined candidate CAD/display/cost identity differs")
    for path in (
        "artifacts/covered-carriers-composed-review-v1/index.html",
        "artifacts/covered-carriers-production-review-v1/index.html",
    ):
        read(path)
    composed = Path("artifacts/integral-head-composed-review-v2")
    composed_report = load(str(composed / "report.json"))
    if composed_report["status"] != "pass":
        raise ValueError("Composed robot presentation failed")
    for name in ["index.html", "review.css"]:
        read(str(composed / name))
        if sources[str(composed / name)] != composed_report["output_hashes"][name]:
            raise ValueError("Composed robot presentation changed")
    for path, expected in composed_report["source_bindings"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Composed robot presentation source changed: {path}")
    current_robots = []
    for name in ("pip", "scout"):
        path = f"artifacts/{name}-integral-head-composed-v2/report.json"
        if composed_report["source_bindings"].get(path) != sha(ROOT / path):
            raise ValueError("Current part evidence differs from displayed assembly")
        assembly = load(path)
        current_robots.append(
            dict(
                name=name.capitalize(),
                parts=assembly["parts"],
                report_path=path,
                status=assembly["status"],
            )
        )
    validate_inventory(bom, current_robots)
    # Adapt the retained viewer without changing its archived geometry or results.
    bridge = """<script>
(() => {
 const root=document.getElementById('composed-head-inspector');
 const send=()=>parent.postMessage({type:'robot-part-selection',
   robot:root.querySelector('[data-id=robot]').value,
   part:root.querySelector('[data-id=part]').value},location.origin);
 new MutationObserver(send).observe(root.querySelector('[data-id=detail]'),
   {childList:true,subtree:true,characterData:true});
})();
</script>"""
    current_html = read(str(composed / "index.html"))
    context_toggle = 'data-id="context" type="checkbox" checked'
    if current_html.count(context_toggle) != 1:
        raise ValueError("Current viewer context toggle changed")
    current_html = current_html.replace(
        context_toggle, 'data-id="context" type="checkbox"'
    )
    if current_html.count("</body>") != 1:
        raise ValueError("Current viewer wrapper changed")
    (output / "current-viewer.html").write_text(
        current_html.replace("</body>", bridge + "</body>")
    )
    carrier_view = Path("artifacts/carrier-composed-review-v4")
    carrier_display = load(str(carrier_view / "report.json"))
    if carrier_display["status"] != "pass":
        raise ValueError("Carrier composition display failed")
    for filename, digest in carrier_display["output_hashes"].items():
        if sha(ROOT / carrier_view / filename) != digest:
            raise ValueError("Carrier composition display changed")
    for path, digest in carrier_display["source_bindings"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Carrier composition source changed: {path}")
    boards = []
    component_tables = {}
    bench_folder = Path("artifacts/bench-fit-v4")
    bench_report = load(str(bench_folder / "report.json"))
    if bench_report["status"] != "pass" or bench_report["physical_status"] != "not_run":
        raise ValueError("First bench narrative needs review")
    (output / "bench").mkdir()
    for path in sorted((ROOT / bench_folder).iterdir()):
        if not path.is_file():
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        expected = bench_report["output_hashes"].get(path.name)
        if path.name != "report.json" and digest != expected:
            raise ValueError(f"Bench output binding mismatch: {path.name}")
        sources[str(bench_folder / path.name)] = digest
        copy_review_asset(path, output / "bench" / path.name, content)
    (output / "media").mkdir()
    for folder, stem, label, bom_folder in [
        ("carrier-v13", "carrier", "Controller carrier", "carrier-native-bom-v1"),
        (
            "logic-board-capacitor-refresh-v1",
            "logic-supply",
            "5 V logic supply · XH connector · refreshed capacitors",
            "logic-native-bom-v1",
        ),
        ("imu-board-v2", "imu", "IMU board", "imu-native-bom-v1"),
    ]:
        image_path = f"artifacts/{folder}/{stem}.png"
        image_bytes = (ROOT / image_path).read_bytes()
        sources[image_path] = hashlib.sha256(image_bytes).hexdigest()
        target = f"media/{stem}.png"
        (output / target).write_bytes(image_bytes)
        report_path = f"artifacts/{folder}/build-report.json"
        report = load(report_path)
        bom_root = Path("artifacts") / bom_folder
        bom_report_path = str(bom_root / "report.json")
        bom_report = load(bom_report_path)
        if bom_report["status"] != "pass" or bom_report["component_count"] <= 0:
            raise ValueError("Native board BOM missing or failed")
        for path, digest in bom_report["source_hashes"].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"Native BOM source changed: {path}")
        for filename, digest in bom_report["output_hashes"].items():
            path = str(bom_root / filename)
            if sha(ROOT / path) != digest:
                raise ValueError(f"Native BOM output changed: {path}")
            sources[path] = digest
        sources[bom_report_path] = sha(ROOT / bom_report_path)
        component_path = str(bom_root / "components.md")
        component_tables[component_path] = load(str(bom_root / "components.json"))
        boards.append(
            dict(
                label=label,
                image=target,
                report=report_path,
                components=component_path,
                status=report["status"],
                scope=report.get("scope", "Retained candidate board build only"),
            )
        )
    startup_path = "artifacts/logic-supply-startup-v1/report.json"
    startup = load(startup_path)
    startup_case = next(c for c in startup["cases"] if c["name"] == "startup")
    plot_startup(output / "media/startup.png", sources, startup_case)
    from tools.compose_forearm_enclosure import verify_report

    spacer_folder = Path("artifacts/yoke-spacer-review-v2")
    spacer_report = verify_report(spacer_folder)
    for filename in ("report.json", *spacer_report["output_hashes"]):
        path = spacer_folder / filename
        sources[str(path)] = sha(ROOT / path)
    cost_folder = Path("artifacts/production-cost-review-v1")
    cost_report = verify_report(cost_folder)
    for filename in ("report.json", *cost_report["output_hashes"]):
        path = cost_folder / filename
        sources[str(path)] = sha(ROOT / path)
    bom_links.append(
        (
            "docs/design/production-print-budget.md",
            "Earlier baseline solid counts and print budget",
        )
    )
    observed_folder = Path("artifacts/observed-balance-review-v2")
    observed_report = verify_report(observed_folder)
    for filename in ("report.json", *observed_report["output_hashes"]):
        path = observed_folder / filename
        sources[str(path)] = sha(ROOT / path)
    assembly_links.append(
        (
            "docs/design/observed-posture-balance.md",
            "Encoder-posture balance and bounded CAD checks",
        )
    )
    release_folder = Path("artifacts/release-guard-review-v1")
    release_report = verify_report(release_folder)
    if release_report["status"] != "pass":
        raise ValueError("Placement comparison lacks presentation integrity")
    for filename in ("report.json", *release_report["output_hashes"]):
        path = release_folder / filename
        sources[str(path)] = sha(ROOT / path)
    assembly_links.append(
        (
            "docs/design/release-aware-placement.md",
            "Release-aware placement and actuator bench",
        )
    )
    for folder_name in (
        "arm-material-review-v3",
        "forearm-accounted-review-v3",
        "shoulder-assist-review-v2",
        "shoulder-spring-review-v1",
        "joint-bench-demo-v2",
        "shin-cover-review-v1",
        "shin-dynamics-review-v2",
        "scout-posture-review-v1",
        "covered-production-review-v2",
        "hardware-link-review-v2",
        "hardware-observability-review-v2",
    ):
        folder = Path("artifacts") / folder_name
        comparison = verify_report(folder)
        if comparison["status"] != "pass":
            raise ValueError(f"Comparison integrity failed: {folder}")
        for filename in ("report.json", *comparison["output_hashes"]):
            path = folder / filename
            sources[str(path)] = sha(ROOT / path)
    assembly_links.extend(
        [
            (
                "docs/design/clamp-lightening.md",
                "Lighter clamp geometry and paired physical bench",
            ),
            (
                "docs/design/forearm-mass-accounting.md",
                "Forearm reserve reconciliation and current replays",
            ),
        ]
    )
    assembly_links.append(
        (
            "docs/design/shoulder-assist-study.md",
            "Passive shoulder assistance, space and actuator bench",
        )
    )
    dynamics_folder = Path("artifacts/composed-dynamics-review-v2")
    dynamics_report = load(str(dynamics_folder / "report.json"))
    for path, digest in dynamics_report["source_bindings"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Dynamics evidence changed: {path}")
    for path, digest in dynamics_report["output_hashes"].items():
        read(str(dynamics_folder / path))
        if sha(ROOT / dynamics_folder / path) != digest:
            raise ValueError(f"Dynamics display changed: {path}")
    wheel_review = Path("artifacts/balance-wheel-clearance-review-v2")
    wheel_report = load(str(wheel_review / "report.json"))
    for path, digest in wheel_report["source_bindings"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Wheel clearance evidence changed: {path}")
    for path, digest in wheel_report["output_hashes"].items():
        read(str(wheel_review / path))
        if sha(ROOT / wheel_review / path) != digest:
            raise ValueError(f"Wheel clearance display changed: {path}")
    coverage_folder = Path("artifacts/vision-coverage-audit-v1")
    coverage_report = verify_report(coverage_folder)
    if "error" in coverage_report or len(coverage_report.get("cases", [])) != 5:
        raise ValueError("Camera power coverage review is incomplete")
    for name in ["report.json", "index.html", "waveforms.png"]:
        path = coverage_folder / name
        sources[str(path)] = sha(ROOT / path)
    for path, digest in coverage_report["source_bindings"].items():
        sources[path] = digest
    evidence_paths = {p for p, _ in assembly_links + learning_links + bom_links}
    evidence_paths.add(startup_path)
    evidence_paths.update(p for b in boards for p in (b["report"], b["components"]))
    evidence_paths.update(p for e in bom["entries"] for p in e["evidence_paths"])
    evidence_paths.update(e["path"] for a in validation["areas"] for e in a["evidence"])
    evidence_paths.update(
        e["path"] for g in verification["groups"] for e in g["evidence"]
    )
    evidence_paths.update(r["report_path"] for r in current_robots)
    links = {}
    (output / "evidence").mkdir()
    for path in sorted(evidence_paths):
        content = read(path)
        key = hashlib.sha256(path.encode()).hexdigest()[:16]
        target = f"evidence/{key}.html"
        title = html.escape(path)
        evidence_body = f"<pre>{html.escape(content)}</pre>"
        if path in component_tables:
            table_rows = "".join(
                "<tr>"
                + "".join(
                    f"<td>{html.escape(str(value))}</td>"
                    for value in (
                        ", ".join(row["references"]),
                        row["quantity"],
                        row["manufacturer"] + " · " + row["mpn"],
                        row["identity_basis"],
                    )
                )
                + "</tr>"
                for row in component_tables[path]
            )
            evidence_body = (
                "<style>.native-bom{width:100%;border-collapse:collapse}"
                ".native-bom th,.native-bom td{text-align:left;padding:.65rem .8rem;"
                "border-bottom:1px solid #d9ded6;vertical-align:top}"
                ".native-bom th{background:#e8eee5}.bom-scroll{overflow-x:auto}</style>"
                "<h2>Candidate board components</h2>"
                "<p>Native KiCad quantities checked against the retained parts manifest. "
                "Physical testing and fabrication release remain open.</p>"
                '<div class="bom-scroll"><table class="native-bom"><thead><tr><th>References</th><th>Quantity</th>'
                "<th>Manufacturer / part</th><th>Identity basis</th></tr></thead>"
                f"<tbody>{table_rows}</tbody></table></div>"
                "<p>Mounting-hole footprints are excluded. PCB, assembly, harness, "
                "mounting hardware and purchasing overbuy are additional scope.</p>"
                f"<details><summary>Sources and export boundary</summary>{evidence_body}</details>"
            )
        (output / target).write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><link rel="stylesheet" href="../review.css">'
            f'</head><body><main><p><a href="../index.html">← Product review</a></p>'
            f'<h3>{title}</h3><p class="fine">Retained source · SHA-256 {sources[path]}</p>'
            f"{evidence_body}</main></body></html>"
        )
        links[path] = target
    data = dict(
        bom=bom,
        validation=validation,
        part_verification=verification,
        robots=robots,
        current_robots=current_robots,
        links=links,
        assembly_links=assembly_links,
        learning_links=learning_links,
        bom_links=bom_links,
        boards=boards,
        startup=dict(report=startup_path, case=startup_case),
        power_coverage=dict(
            status=coverage_report["status"],
            cases=coverage_report["cases"],
            viewer="../vision-coverage-audit-v1/index.html",
        ),
    )
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    template = read("ui/product-review.html")
    (output / "index.html").write_text(
        template.replace("__REVIEW_DATA__", payload).replace(
            "__DATE__", date.today().isoformat()
        )
    )
    for source, target in [
        ("ui/product-review.css", "review.css"),
        ("ui/product-review.js", "review.js"),
    ]:
        (output / target).write_text(read(source))
    (output / "viewer.html").write_text(wrapper(fragment, selection=True))
    gym = read("artifacts/robot-inspector-v6/inspector.html")
    gym_report = load("artifacts/robot-inspector-v6/report.json")
    if hashlib.sha256(gym.encode()).hexdigest() != gym_report["fragment_sha256"]:
        raise ValueError("Historical gym display identity mismatch")
    # Preserve the archived viewer, but bound a first-frame browser clock mismatch.
    old_delta = "Math.min((now-last)/1000,.1)"
    if gym.count(old_delta) != 1:
        raise ValueError("Historical playback adapter no longer matches")
    gym = gym.replace(old_delta, "Math.max(0,Math.min((now-last)/1000,.1))")
    (output / "gym.html").write_text(wrapper(gym))
    read("tools/build_product_review.py")
    write(output / "data.json", data)
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation integrity and BOM inventory coverage only; not product readiness",
            source_hashes=sources,
            bom_coverage=coverage,
            evidence_count=len(links),
            display_adapters=[
                "Current composed CAD emits part selection; opaque default; archived geometry unchanged",
                "Historical gym frame delta clamped to [0, 0.1] s; archived source unchanged",
                "Neck comparison null selection rendered as no candidate meets the motion and clearance criteria; archived source unchanged",
            ],
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
            not_run=[
                "physical verification",
                "manufacturing release",
                "NVIDIA training",
                "learned skills",
                "world model",
            ],
        ),
    )
    print(json.dumps(dict(status="pass", output=str(output), bom_coverage=coverage)))


def plot_startup(output, sources, result):
    """Use the existing isolated plotting environment without modifying CAD deps."""
    import subprocess

    for path in (
        "tools/plot_review_startup.py",
        "requirements-plot-macos-arm64.lock",
        "artifacts/logic-supply-startup-v1/startup/trace.dat",
        "artifacts/logic-supply-startup-v1/startup/report.json",
    ):
        sources[path] = sha(ROOT / path)
    actual = json.loads(
        (ROOT / "artifacts/logic-supply-startup-v1/startup/report.json").read_text()
    )
    for key in ("minimum_output_v", "maximum_output_v", "whole_trace_peak_output_v"):
        if actual[key] != result[key]:
            raise ValueError("Startup summary differs from case report")
    subprocess.run(
        [
            str(ROOT / ".venv-plot/bin/python"),
            str(ROOT / "tools/plot_review_startup.py"),
            "--output",
            str(output.resolve()),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", type=Path, default=Path("artifacts/head-display-v7"))
    run(**vars(parser.parse_args()))
