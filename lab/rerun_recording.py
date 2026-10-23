"""Export verified canonical recordings to optional Rerun telemetry replay.

Run with .venv-observe. Optional source-bound geometry uses recorded body poses;
legacy recordings remain telemetry-only when those poses are absent.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .recording import Event, Signal, encoded, read, sha256, verify_recording


def scalar_channels(event, spec):
    """Split mixed-unit commands without ever labeling wheel speed as an angle."""
    if event.validity != "valid" or spec.payload != "numeric":
        return []
    values = event.value if spec.shape else [event.value]
    units = spec.unit if isinstance(spec.unit, list) else [spec.unit] * len(values)
    labels = spec.labels or (
        [str(i) for i in range(len(values))] if spec.shape else ["value"]
    )
    return [
        (unit, f"{spec.role}/{event.signal}/{label}", value)
        for unit, label, value in zip(units, labels, values, strict=True)
    ]


def recording_about(manifest, has_scene=False):
    origin = manifest["source_kind"]
    labels = {
        "physics": "physics simulation",
        "synthetic_transport": "SYNTHETIC transport; no physical measurements",
        "hardware_asserted": "hardware capture (origin asserted, not authenticated); physical qualification not run",
    }
    if origin not in labels:
        raise ValueError("unsupported recording origin")
    if has_scene and origin != "physics":
        raise ValueError(
            "physical/synthetic capture has no qualified scene pose adapter"
        )
    timing = (
        "record_time is host arrival relative to the first retained chunk. Device clocks are preserved in raw events and are NOT synchronized. No sensor age or capture_time is inferred."
        if manifest["clock"] == "host_arrival_relative"
        else "Select record_time for row-aligned telemetry. capture_time only contains samples with known acquisition time; command/estimator acquisition times may be unknown."
    )
    return (
        f"# {manifest['robot']} · {manifest['adapter']}\n\n"
        f"Original result: **{manifest['outcome']['status']}**; {manifest['outcome']['termination']}.\n\n"
        f"{manifest['scope']}. Source: **{labels[origin]}**.\n\n"
        f"{timing}\n\n"
        + (
            "3D poses are privileged simulator truth with source-bound geometry. "
            if has_scene
            else "No 3D scene is attached. "
        )
        + "Camera video is unavailable. Signals retain their unit, frame and provenance. No missing signal is filled with zero. The separate validity tab uses 1=valid, 0=invalid, -1=unavailable; a previous plotted value is not evidence of current validity.\n\n"
        f"Recording: `{manifest['recording_id']}`\n\n"
        f"Provenance: {manifest['provenance_assurance']}.\n\n"
        + "\n".join(f"- {x}" for x in manifest["limitations"])
    )


def export_rerun(source, output, scene=None):
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rb

    source, output = Path(source), Path(output)
    validation = verify_recording(source)
    manifest = read(source / "manifest.json")
    specs = {k: Signal.model_validate(v) for k, v in manifest["signals"].items()}
    recording_about(
        manifest, scene is not None
    )  # Reject false scene/origin claims before export.
    bundle = None
    if scene is not None:
        from .rerun_scene import load_scene, log_body_poses, log_static_scene

        bundle = load_scene(scene)
        if bundle["source_xml_sha256"] != manifest["source_hashes"]["scene.xml"]:
            raise ValueError("scene geometry does not match recording model")
        if "scene/bodies" not in manifest.get("available_signals", []):
            raise ValueError("recording lacks full body poses for scene replay")
    output.mkdir(parents=True, exist_ok=False)
    units = {
        u
        for name, s in specs.items()
        for u in (s.unit if isinstance(s.unit, list) else [s.unit])
        if s.payload == "numeric" and name in manifest.get("available_signals", specs)
    }
    # Unit strings become a single safe path component, while metadata retains SI spelling.
    unit_paths = {
        u: u.replace("/", "_per_").replace("*", "_").replace("^", "pow") for u in units
    }
    blueprint = rb.Blueprint(
        rb.Vertical(
            rb.TextDocumentView(origin="/about", name="Episode and limits"),
            rb.Tabs(
                *(
                    [
                        rb.Spatial3DView(
                            origin="/scene",
                            name="Robot and task",
                            eye_controls=rb.EyeControls3D(
                                position=[0.85, -0.95, 0.7],
                                look_target=[0, 0, 0.25],
                                eye_up=[0, 0, 1],
                            ),
                        )
                    ]
                    if bundle
                    else []
                ),
                *[
                    rb.TimeSeriesView(origin=f"/values/{unit_paths[u]}", name=u)
                    for u in sorted(units, key=lambda u: (u != "V", u))
                ],
                rb.TimeSeriesView(
                    origin="/validity",
                    name="Validity: 1 valid / 0 invalid / -1 unavailable",
                ),
                rb.TimeSeriesView(origin="/timing", name="Sample age (ms)"),
                rb.TextLogView(origin="/events", name="Status and missing data"),
            ),
            row_shares=[0.25, 0.75],
        ),
        rb.TimePanel(timeline="record_time", state="expanded"),
    )
    view_identity = {
        "source_manifest": sha256(source / "manifest.json"),
        "exporter": sha256(__file__),
        "scene": sha256(scene) if scene else None,
        "scene_exporter": sha256(Path(__file__).with_name("rerun_scene.py"))
        if scene
        else None,
        "rerun_version": rr.__version__,
    }
    view_id = "view-" + hashlib.sha256(encoded(view_identity).encode()).hexdigest()[:24]
    recording = rr.RecordingStream("robot-learning-lab-telemetry", recording_id=view_id)
    target = output / "episode.rrd"
    recording.save(target, default_blueprint=blueprint)
    about = recording_about(manifest, bool(bundle))
    recording.log(
        "about",
        rr.TextDocument(about, media_type="text/markdown"),
        static=True,
        strict=True,
    )
    recording.log(
        "metadata", rr.TextDocument(encoded(manifest)), static=True, strict=True
    )
    scalar_samples = 0
    scene_frames = 0
    validity_samples = 0
    try:
        if bundle:
            recording.log("scene", rr.ViewCoordinates.FLU, static=True, strict=True)
            log_static_scene(recording, bundle)
        with (source / "events.jsonl").open() as stream:
            for line in stream:
                event = Event.model_validate_json(line)
                spec = specs[event.signal]
                recording.reset_time()  # Unknown capture time must not inherit a previous sample's clock.
                recording.set_time(
                    "record_time", duration=np.timedelta64(event.record_time_ns, "ns")
                )
                recording.set_time("event_index", sequence=event.sequence)
                if event.capture_time_ns is not None:
                    recording.set_time(
                        "capture_time",
                        duration=np.timedelta64(event.capture_time_ns, "ns"),
                    )
                    recording.log(
                        "timing/" + event.signal,
                        rr.Scalars(
                            (event.record_time_ns - event.capture_time_ns) / 1e6
                        ),
                        strict=True,
                    )
                if spec.payload == "numeric":
                    recording.log(
                        "validity/" + event.signal,
                        rr.Scalars(
                            {"valid": 1, "invalid": 0, "unavailable": -1}[
                                event.validity
                            ]
                        ),
                        strict=True,
                    )
                    validity_samples += 1
                for unit, path, value in scalar_channels(event, spec):
                    recording.log(
                        f"values/{unit_paths[unit]}/{path}",
                        rr.Scalars(value),
                        strict=True,
                    )
                    scalar_samples += 1
                if (
                    bundle
                    and event.signal == "scene/bodies"
                    and event.validity == "valid"
                ):
                    required = {g["body"] for g in bundle["geoms"]}
                    if not required.issubset(event.value):
                        raise ValueError("recorded poses omit scene geometry bodies")
                    log_body_poses(recording, event.value)
                    scene_frames += 1
                elif (
                    spec.payload != "numeric" or event.validity != "valid" or event.note
                ):
                    # Keep validity and raw invalid values inspectable instead of drawing them as normal measurements.
                    recording.log(
                        "events/" + event.signal,
                        rr.TextLog(
                            encoded(
                                {
                                    "validity": event.validity,
                                    "value": event.value,
                                    "capture_time_ns": event.capture_time_ns,
                                    "note": event.note,
                                }
                            )
                        ),
                        strict=True,
                    )
        recording.flush(timeout_sec=30)
    finally:
        recording.disconnect()
    report = {
        "status": "exported",
        "rerun_version": rr.__version__,
        "source_recording_id": manifest["recording_id"],
        "viewer_recording_id": view_id,
        "view_identity": view_identity,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_kind": manifest["source_kind"],
        "clock": manifest["clock"],
        "exporter_sha256": sha256(__file__),
        "rrd_sha256": sha256(target),
        "source_validation": validation,
        "scalar_samples": scalar_samples,
        "validity_samples": validity_samples,
        "scene_frames": scene_frames,
        "scene_bundle_sha256": sha256(scene) if scene else None,
        "scope": "offline telemetry and recorded full-body scene"
        if bundle
        else "offline telemetry charts and diagnostics",
        "limitations": [
            "no synchronized camera stream or live hardware adapter",
            "software control tick is not measured mechanical actuation",
        ],
        "visual_inspection": "not_run",
    }
    (output / "rerun_recording.py").write_bytes(Path(__file__).read_bytes())
    if scene:
        (output / "scene.json").write_bytes(Path(scene).read_bytes())
        (output / "scene.json.sha256").write_text(sha256(scene) + "\n")
        (output / "rerun_scene.py").write_bytes(
            Path(__file__).with_name("rerun_scene.py").read_bytes()
        )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scene",
        type=Path,
        help="Source-bound scene JSON from lab.rerun_scene.export_scene",
    )
    args = parser.parse_args()
    print(json.dumps(export_rerun(args.source, args.output, args.scene), indent=2))


if __name__ == "__main__":
    main()
