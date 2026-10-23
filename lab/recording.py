"""Versioned observation records and loss-aware import of retained robot trials.

No simulator or viewer dependency. Original producer files remain the evidence;
this is a sealed, portable projection for inspection and later dataset adapters.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .config import Strict


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def read(path):
    value = json.loads(Path(path).read_text())
    encoded(value)  # Reject JSON's nonstandard NaN/Infinity extension recursively.
    return value


def ns(seconds, *, allow_negative=False):
    if seconds < 0 and not allow_negative:
        raise ValueError("negative episode time")
    return round(seconds * 1_000_000_000)


class Signal(Strict):
    payload: Literal[
        "numeric",
        "label",
        "task_flags",
        "imu_health",
        "scene_bodies",
        "control_timing",
        "unavailable",
    ] = "numeric"
    role: Literal[
        "simulated_sensor",
        "measured",
        "estimated",
        "truth",
        "command",
        "reference",
        "predicted",
        "outcome",
        "diagnostic",
    ]
    unit: str | list[str]
    frame: str | None = None
    labels: list[str] | None = None
    shape: list[int] | None = None
    description: str

    @model_validator(mode="after")
    def dimensions(self):
        if self.payload == "numeric" and (
            self.shape is None or len(self.shape) > 1 or any(n < 1 for n in self.shape)
        ):
            raise ValueError("numeric signals need scalar [] or vector [length] shape")
        for values in (self.labels, self.unit if isinstance(self.unit, list) else None):
            if values is not None and self.shape != [len(values)]:
                raise ValueError("labels/units must match vector shape")
        return self


class TaskFlags(Strict):
    success: bool
    terminated: bool
    truncated: bool
    reason: str | None
    grasped: bool
    lifted: bool


class ImuHealth(Strict):
    status: int
    error: int
    valid: bool
    sequence: int = Field(ge=0)
    poll_age_us: int = Field(ge=0)
    update_age_bound_us: int = Field(ge=0)
    saturated: bool
    read_end_us: int = Field(ge=0)
    max_poll_gap_us: int = Field(ge=0)


class Event(Strict):
    sequence: int = Field(ge=0)
    signal: str
    # When the producer logged the row, not necessarily when it acquired the value.
    record_time_ns: int = Field(ge=0)
    # Signed capture time permits samples acquired during held pre-motion startup.
    capture_time_ns: int | None = None
    clock: str = "episode_simulation"
    validity: Literal["valid", "invalid", "unavailable"] = "valid"
    value: JsonValue
    note: str | None = None

    @model_validator(mode="after")
    def complete(self):
        encoded(self.value)
        if (
            self.capture_time_ns is not None
            and self.capture_time_ns > self.record_time_ns
        ):
            raise ValueError("capture time is in the future of this same-clock record")
        if (self.validity == "unavailable") != (self.value is None):
            raise ValueError("unavailable requires null; other samples retain a value")
        if (self.capture_time_ns is None or self.validity != "valid") and not self.note:
            raise ValueError("unknown timing or invalid/missing data requires a note")
        return self


def validate_event(event, signals):
    spec = signals[event.signal]
    if event.validity == "unavailable":
        return
    if spec.payload == "unavailable":
        raise ValueError("this adapter does not implement this payload")
    if spec.payload == "label" and not isinstance(event.value, str):
        raise ValueError("expected a text label")
    if spec.payload == "task_flags":
        TaskFlags.model_validate(event.value, strict=True)
    if spec.payload == "imu_health":
        ImuHealth.model_validate(event.value, strict=True)
    if spec.payload == "scene_bodies":
        if not isinstance(event.value, dict) or "torso" not in event.value:
            raise ValueError("scene body snapshot requires torso")
        for pose in event.value.values():
            BodyPose.model_validate(pose, strict=True)
    if spec.payload == "control_timing":
        timing = ControlTiming.model_validate(event.value, strict=True)
        times = [timing.issued_time_s]
        if timing.estimator_update_time_s is not None:
            times.append(timing.estimator_update_time_s)
        for receipt in timing.receipts:
            times.append(receipt.received_time_s)
            if receipt.first_control_tick_s is not None:
                times.append(receipt.first_control_tick_s)
        if any(ns(t) > event.record_time_ns for t in times):
            raise ValueError("future command/estimator time")
    if spec.shape is not None:
        values = event.value
        if spec.shape == []:
            values = [values]
        elif (
            len(spec.shape) != 1
            or not isinstance(values, list)
            or len(values) != spec.shape[0]
        ):
            raise ValueError(f"wrong shape for {event.signal}")
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
            raise ValueError(f"non-numeric sample for {event.signal}")


class EventWriter:
    """Sequential recorder with strict stream identity and no implicit resampling."""

    def __init__(self, path, signals, clock="episode_simulation"):
        self.signals = signals
        self.clock = clock
        self.stream = Path(path).open("x")
        self.count = 0
        self.last_time = -1
        self.capture_times = {}
        self.available_signals = set()

    def append(self, event):
        event = Event.model_validate(event)
        if event.sequence != self.count or event.record_time_ns < self.last_time:
            raise ValueError("event sequence/time regression")
        if event.clock != self.clock:
            raise ValueError("clock mismatch; map clock domains explicitly")
        validate_event(event, self.signals)
        validate_capture_order(event, self.capture_times)
        self.stream.write(encoded(event.model_dump()) + "\n")
        self.count += 1
        self.last_time = event.record_time_ns
        if event.validity == "valid":
            self.available_signals.add(event.signal)

    def close(self):
        self.stream.close()


def validate_capture_order(event, capture_times):
    if event.capture_time_ns is not None:
        if (
            event.signal in capture_times
            and event.capture_time_ns < capture_times[event.signal]
        ):
            raise ValueError("signal acquisition clock regressed")
        capture_times[event.signal] = event.capture_time_ns


class BodyPose(Strict):
    position_m: list[float] = Field(min_length=3, max_length=3)
    quaternion_xyzw: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def unit_quaternion(self):
        if abs(sum(x * x for x in self.quaternion_xyzw) - 1) > 1e-5:
            raise ValueError("body quaternion must have unit norm")
        return self


class CommandReceipt(Strict):
    sequence: int = Field(ge=0)
    received_time_s: float = Field(ge=0)
    accepted: bool
    targets: list[float]
    first_control_tick_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def tick_order(self):
        if self.first_control_tick_s is not None and (
            not self.accepted or self.first_control_tick_s < self.received_time_s
        ):
            raise ValueError("control tick precedes accepted command")
        return self


class ControlTiming(Strict):
    issued_time_s: float = Field(ge=0)
    sequence: int = Field(ge=0)
    estimator_update_time_s: float | None = Field(default=None, ge=0)
    receipts: list[CommandReceipt]


def signal_catalog(interfaces, kind):
    names = interfaces["joint_order"]
    joints = interfaces["joints"]
    if names != [j["name"] for j in joints] or len(set(names)) != len(names):
        raise ValueError("joint ordering is inconsistent or duplicated")
    units = ["rad/s" if j["continuous"] else "rad" for j in joints]
    if interfaces["mixed_command"]["units_by_joint"] != units:
        raise ValueError("mixed command units differ from the joint modes")
    signals = {}

    def add(
        name,
        role,
        unit,
        description,
        frame=None,
        shape=None,
        labels=None,
        payload="numeric",
    ):
        signals[name] = Signal(
            payload=payload,
            role=role,
            unit=unit,
            description=description,
            frame=frame,
            shape=shape,
            labels=labels,
        )

    for name, unit in [
        ("position_rad", "rad"),
        ("velocity_rad_s", "rad/s"),
        ("effort_nm", "N*m"),
    ]:
        add(
            "joint/" + name,
            "simulated_sensor" if name != "effort_nm" else "truth",
            unit,
            "Retained simulator joint state; not physical servo feedback",
            "joint_local",
            [len(names)],
            names,
        )
    add(
        "command/target",
        "command",
        units,
        "Issued mixed targets; optional control/timing distinguishes received targets and first controller use",
        "joint_local",
        [len(names)],
        names,
    )
    add(
        "control/timing",
        "diagnostic",
        "s",
        "Issue, receipt and first software control tick; not measured mechanical actuation",
        payload="control_timing",
    )
    add(
        "scene/bodies",
        "truth",
        "m and XYZW",
        "Named body-to-world poses at the retained state time",
        frame="world",
        payload="scene_bodies",
    )
    add(
        "supervisor/status",
        "diagnostic",
        "code",
        "Producer supervisor status",
        shape=[],
    )
    add(
        "base/position_m",
        "truth",
        "m",
        "Full root position, when retained",
        "world",
        [3],
    )
    add(
        "base/quaternion_xyzw",
        "truth",
        "1",
        "Full root rotation, when retained",
        "world",
        [4],
    )
    add(
        "camera/rgb",
        "simulated_sensor",
        "pixel",
        "Synchronized image stream; endpoint PNGs are not video",
        payload="unavailable",
    )
    for name, unit in [("gyro_rad_s", "rad/s"), ("specific_force_m_s2", "m/s^2")]:
        add(
            "imu/" + name,
            "simulated_sensor",
            unit,
            "Simulated torso-frame IMU; balance records the noise/calibration-processed sample consumed by its estimator",
            "torso",
            [3],
        )
    if kind == "task":
        add(
            "task/block_position_m",
            "truth",
            "m",
            "Privileged object pose",
            "world",
            [3],
        )
        add(
            "task/block_quaternion_xyzw",
            "truth",
            "1",
            "Privileged object rotation",
            "world",
            [4],
        )
        add(
            "task/reward",
            "outcome",
            "1",
            "Producer training reward, not a replacement success metric",
            shape=[],
        )
        add(
            "task/flags",
            "outcome",
            "1",
            "Original success, termination and truncation flags",
            payload="task_flags",
        )
        add(
            "task/phase",
            "diagnostic",
            "label",
            "Reference-controller phase",
            payload="label",
        )
    else:
        add(
            "imu/poll_start_proxy_s",
            "diagnostic",
            "s",
            "Driver poll start relative to motion release; not physical conversion time",
            shape=[],
        )
        add(
            "imu/health",
            "diagnostic",
            "structured",
            "Driver health; read_end_us is in driver time, not episode time",
            payload="imu_health",
        )
        for key, unit, role in [
            ("estimated_pitch_rad", "rad", "estimated"),
            ("estimated_position_m", "m", "estimated"),
            ("estimated_velocity_m_s", "m/s", "estimated"),
            ("pitch_rad", "rad", "truth"),
            ("tilt_rad", "rad", "truth"),
            ("truth_x_m", "m", "truth"),
            ("truth_velocity_m_s", "m/s", "truth"),
            ("base_height_m", "m", "truth"),
            ("reference_x_m", "m", "reference"),
            ("reference_velocity_m_s", "m/s", "reference"),
            ("estimated_bus_current_a", "A", "estimated"),
        ]:
            add(
                "balance/" + key,
                role,
                unit,
                "Retained balance value; see timing notes on each event",
                shape=[],
            )
    return signals


def quaternion_xyzw(wxyz):
    if len(wxyz) != 4 or abs(sum(v * v for v in wxyz) - 1) > 1e-5:
        raise ValueError("expected a normalized WXYZ quaternion")
    return [*wxyz[1:], wxyz[0]]


def convert_rows(rows, interfaces, kind, writer):
    names = interfaces["joint_order"]
    previous_time = -1
    row_count = 0
    for row in rows:
        state = row["joint_state"] if kind == "task" else row
        t = ns(state["time_s"])
        if t <= previous_time:
            raise ValueError(
                "producer rows must have strictly increasing simulation time"
            )
        previous_time = t
        row_count += 1
        if kind == "task" and state["names"] != names:
            raise ValueError("trajectory joint order differs from retained interfaces")

        def emit(name, value, capture=t, validity="valid", note=None):
            writer.append(
                dict(
                    sequence=writer.count,
                    signal=name,
                    record_time_ns=t,
                    capture_time_ns=capture,
                    value=value,
                    validity=validity,
                    note=note,
                )
            )

        for field in ("position_rad", "velocity_rad_s", "effort_nm"):
            emit("joint/" + field, state[field])
        emit("supervisor/status", state["supervisor_status"])
        timing = row.get("control_timing")
        if timing is not None:
            ControlTiming.model_validate(timing, strict=True)
            if any(len(r["targets"]) != len(names) for r in timing["receipts"]):
                raise ValueError("receipt target shape differs from joint order")
            emit(
                "control/timing",
                timing,
                None,
                note="Nested timestamps mark issue, receipt and first software control tick",
            )
        else:
            emit(
                "control/timing",
                None,
                None,
                "unavailable",
                "Legacy producer did not retain command timing",
            )
        emit(
            "command/target",
            row["action" if kind == "task" else "targets"],
            ns(timing["issued_time_s"]) if timing else None,
            note="Issued target; see control/timing for separately received commands"
            if timing
            else "Issued before this post-step row; exact issue/receipt/apply times not retained",
        )
        scene = row.get("scene_state")
        if scene is not None:
            if (
                scene["source"] != "privileged_simulator_truth"
                or ns(scene["time_s"]) != t
            ):
                raise ValueError(
                    "scene source or timestamp differs from retained state"
                )
            emit("scene/bodies", scene["bodies"])
            emit("base/position_m", scene["bodies"]["torso"]["position_m"])
            emit("base/quaternion_xyzw", scene["bodies"]["torso"]["quaternion_xyzw"])
        else:
            for field in ("scene/bodies", "base/position_m", "base/quaternion_xyzw"):
                emit(
                    field,
                    None,
                    None,
                    "unavailable",
                    "Legacy trajectory lacks full scene poses; no reconstruction",
                )
        for field in ("camera/rgb",):
            emit(
                field,
                None,
                None,
                "unavailable",
                "Not retained in this trajectory; no reconstruction or zero fill",
            )
        for field in ("gyro_rad_s", "specific_force_m_s2"):
            sample = row.get("imu") if kind == "task" else row.get("controller_imu")
            if sample is not None:
                proxy = sample.get("timestamp_basis") == "poll_start_proxy"
                emit(
                    "imu/" + field,
                    sample[field],
                    None if proxy else ns(sample["time_s"], allow_negative=True),
                    note="Driver poll start is only a timestamp proxy; conversion/acquisition time is unknown"
                    if proxy
                    else None,
                )
            else:
                emit(
                    "imu/" + field,
                    None,
                    None,
                    "unavailable",
                    "Balance row retains health, not raw IMU vectors",
                )
        if kind == "task":
            emit("task/block_position_m", row["task_state"]["block_position_m"])
            emit(
                "task/block_quaternion_xyzw",
                quaternion_xyzw(row["task_state"]["block_quaternion_wxyz"]),
            )
            emit("task/reward", row["reward"])
            emit(
                "task/flags",
                {
                    k: row[k]
                    for k in (
                        "success",
                        "terminated",
                        "truncated",
                        "reason",
                        "grasped",
                        "lifted",
                    )
                },
            )
            emit(
                "task/phase",
                row["phase"],
                ns(timing["issued_time_s"]) if timing else None,
                note="Controller phase before the post-step observation",
            )
        else:
            sample = row.get("controller_imu")
            if sample and sample.get("timestamp_basis") == "poll_start_proxy":
                emit(
                    "imu/poll_start_proxy_s",
                    sample["time_s"],
                    None,
                    note="Read timestamp proxy, not sensor conversion time",
                )
            else:
                emit(
                    "imu/poll_start_proxy_s",
                    None,
                    None,
                    "unavailable",
                    "No driver poll-start proxy retained",
                )
            health = row.get("imu")
            emit(
                "imu/health",
                health,
                None,
                "valid" if health is not None else "unavailable",
                "Driver clock offset is not normalized; health.valid describes the sensor, not this diagnostic record",
            )
            for key, spec in writer.signals.items():
                if key.startswith("balance/"):
                    source_key = key.split("/")[1]
                    known_time = (
                        spec.role == "truth" or source_key == "estimated_bus_current_a"
                    )
                    capture = t if known_time else None
                    if not known_time and timing is not None:
                        capture = ns(timing["estimator_update_time_s"])
                    emit(
                        key,
                        row[source_key],
                        capture,
                        note=None
                        if known_time
                        else "Pre-step estimate/reference; IMU acquisition is separately timestamped"
                        if timing
                        else "Pre-step estimate/reference; acquisition time not retained",
                    )
    return row_count


def _source_inputs(folder, kind):
    root = folder if kind == "task" else folder.parent
    report_name = "report.json" if kind == "task" else "balance-report.json"
    report = read(root / report_name)
    outcome = report if kind == "task" else read(folder / "result.json")
    interfaces = read(root / "interfaces.json")
    if report.get("mass_model") != interfaces.get("mass_model"):
        raise ValueError("report/interface mass identity mismatch")
    inputs = {
        "producer-report.json": root / report_name,
        "interfaces.json": root / "interfaces.json",
        "resolved-config.json": root / "resolved-config.json",
        "trajectory.jsonl": folder / "trajectory.jsonl",
        "scene.xml": folder / ("scene.xml" if kind == "task" else "model.xml"),
    }
    # Producer source snapshots, not today's mutable workspace.
    upstream_interfaces = []
    upstream_configs = []
    for name, expected in report["source_hashes"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe producer source path")
        from .artifact_paths import checked_path
        snapshot = checked_path(root / "source", name)
        if sha256(snapshot) != expected:
            raise ValueError(f"producer source snapshot changed: {name}")
        if name.endswith("mass-properties.json") and expected == (
            report.get("mass_model") or {}
        ).get("sha256"):
            inputs["mass-properties.json"] = snapshot
        if name.endswith("interfaces.json"):
            upstream_interfaces.append(read(snapshot))
        if name.endswith("resolved-config.json"):
            upstream_configs.append(read(snapshot))
    if interfaces not in upstream_interfaces:
        raise ValueError("interfaces differ from the producer's retained model input")
    # New optional fields can be explicit defaults in a fresh producer and absent
    # in its retained model input. Preserve original hashes; compare effective values.
    from .config import RobotConfig
    from pydantic import ValidationError

    def effective_config(value):
        try:
            return RobotConfig.model_validate(value).model_dump(mode="json")
        except ValidationError:
            return value

    if effective_config(read(inputs["resolved-config.json"])) not in [
        effective_config(c) for c in upstream_configs
    ]:
        raise ValueError(
            "configuration differs from the producer's retained model input"
        )
    if report.get("mass_model") and "mass-properties.json" not in inputs:
        raise ValueError("selected full-tensor mass profile is missing")
    config_hash = sha256(inputs["resolved-config.json"])
    if report.get("resolved_config_sha256", config_hash) != config_hash:
        raise ValueError("producer configuration hash mismatch")
    if kind == "task":
        episode = read(folder / "episode.json")
        unsigned = {k: v for k, v in episode.items() if k != "episode_id"}
        identity = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True).encode()
        ).hexdigest()
        if episode["episode_id"] != "episode-" + identity[:24]:
            raise ValueError("original episode identity changed")
        for name, key in [
            ("trajectory.jsonl", "trajectory_sha256"),
            ("scene.xml", "scene_sha256"),
            ("producer-report.json", "report_sha256"),
            ("resolved-config.json", "robot_config_sha256"),
        ]:
            if sha256(inputs[name]) != episode[key]:
                raise ValueError(f"original episode artifact changed: {name}")
        inputs["episode.json"] = folder / "episode.json"
        assurance = "producer episode seal and source snapshots verified"
    else:
        if (
            outcome.get("joint_order", interfaces["joint_order"])
            != interfaces["joint_order"]
        ):
            raise ValueError("balance joint order mismatch")
        inputs["result.json"] = folder / "result.json"
        inputs["case.json"] = folder / "case.json"
        case = read(inputs["case.json"])
        if case != outcome["input"] or case["name"] != outcome["case"]:
            raise ValueError("balance case/result mismatch")
        assurance = "source snapshots verified; legacy trajectory/model/result first sealed at import"
    return inputs, interfaces, report, outcome, assurance


def export_recording(folder, output, kind):
    folder, output = Path(folder), Path(output)
    if kind not in ("task", "balance"):
        raise ValueError("only retained task and balance adapters are implemented")
    inputs, interfaces, producer, outcome, assurance = _source_inputs(folder, kind)
    signals = signal_catalog(interfaces, kind)
    output.mkdir(parents=True, exist_ok=False)
    (output / "source").mkdir()
    input_hashes = {}
    for name, path in inputs.items():
        data = path.read_bytes()
        (output / "source" / name).write_bytes(data)
        input_hashes[name] = hashlib.sha256(data).hexdigest()
    (output / "source" / "recording.py").write_bytes(Path(__file__).read_bytes())
    input_hashes["recording.py"] = sha256(output / "source" / "recording.py")
    writer = EventWriter(output / "events.jsonl", signals)
    try:
        with (output / "source" / "trajectory.jsonl").open() as stream:
            count = convert_rows(
                (json.loads(line) for line in stream if line.strip()),
                interfaces,
                kind,
                writer,
            )
    finally:
        writer.close()
    if count and abs(writer.last_time - ns(outcome["duration_s"])) > 1:
        raise ValueError("trajectory endpoint differs from reported duration")
    if not count and outcome["status"] != "fail":
        raise ValueError("empty successful trajectory")
    identity = hashlib.sha256(
        encoded({"kind": kind, "inputs": input_hashes}).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 2,
        "recording_id": "recording-" + identity[:24],
        "source_kind": "physics",
        "adapter": kind,
        "source_location": str(folder),
        "backend": {
            "name": "mujoco",
            "version": producer.get("versions", {}).get("mujoco"),
        },
        "source_hashes": input_hashes,
        "provenance_assurance": assurance,
        "robot": interfaces["robot"],
        "joint_order": interfaces["joint_order"],
        "mass_model": interfaces.get("mass_model"),
        "scope": outcome["scope"],
        "seed": outcome.get("seed", outcome.get("input", {}).get("seed")),
        "scenario": outcome.get("input"),
        "partition": "unassigned",
        "clock": "episode_simulation",
        "clock_note": "row times rounded to nanoseconds; no claim of nanosecond measurement accuracy",
        "outcome": {
            "status": outcome["status"],
            "termination": outcome.get("termination"),
            "duration_s": outcome["duration_s"],
        },
        "signals": {k: v.model_dump() for k, v in signals.items()},
        "rows": count,
        "events": writer.count,
        "events_sha256": sha256(output / "events.jsonl"),
        "available_signals": sorted(writer.available_signals),
        "limitations": [
            "offline projection, not a new simulation, learned skill, physical capture or deployment",
            "original row data retained in source/trajectory.jsonl; projection selects named channels",
            "legacy command/estimator capture times may be unknown; no implicit time alignment or resampling",
            "camera stream unavailable; full scene pose only available when retained by producer",
            "source report outcomes preserved; import validation is not robot task success",
            "retained scene XML references original assets; this is not a standalone simulation bundle",
        ],
    }
    (output / "manifest.json").write_text(encoded(manifest) + "\n")
    (output / "manifest.sha256").write_text(sha256(output / "manifest.json") + "\n")
    return verify_recording(output)


def verify_recording(folder):
    folder = Path(folder)
    if (
        sha256(folder / "manifest.json")
        != (folder / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("recording manifest changed")
    manifest = read(folder / "manifest.json")
    if manifest["schema_version"] not in (1, 2):
        raise ValueError("unsupported recording schema")
    for name, expected in manifest["source_hashes"].items():
        if Path(name).name != name or sha256(folder / "source" / name) != expected:
            raise ValueError(f"recording source changed or unsafe name: {name}")
    identity = hashlib.sha256(
        encoded(
            {"kind": manifest["adapter"], "inputs": manifest["source_hashes"]}
        ).encode()
    ).hexdigest()
    if manifest["recording_id"] != "recording-" + identity[:24]:
        raise ValueError("recording identity mismatch")
    if sha256(folder / "events.jsonl") != manifest["events_sha256"]:
        raise ValueError("recording events changed")
    signals = {k: Signal.model_validate(v) for k, v in manifest["signals"].items()}
    previous = -1
    capture_times = {}
    count = 0
    unavailable = set()
    with (folder / "events.jsonl").open() as stream:
        for line in stream:
            event = Event.model_validate_json(line)
            validate_event(event, signals)
            validate_capture_order(event, capture_times)
            if (
                event.sequence != count
                or event.record_time_ns < previous
                or event.clock != manifest["clock"]
            ):
                raise ValueError("recording sequence/time/clock mismatch")
            previous = event.record_time_ns
            count += 1
            if event.validity == "unavailable":
                unavailable.add(event.signal)
    if count != manifest["events"]:
        raise ValueError("recording event count mismatch")
    return {
        "status": "pass",
        "recording_id": manifest["recording_id"],
        "events": count,
        "rows": manifest["rows"],
        "source_outcome": manifest["outcome"],
        "unavailable_signals": sorted(unavailable),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", choices=["task", "balance"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify_recording(args.source)
    else:
        if args.kind is None or args.output is None:
            parser.error("export requires --kind and --output")
        result = export_recording(args.source, args.output, args.kind)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
