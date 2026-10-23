"""Lossless offline MCAP transport for existing canonical robot recordings.

MCAP owns the container, CRCs, compression and indexes. Robot signal semantics,
origin, independent clocks and source outcomes remain in the existing contract.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from mcap.records import Attachment, Channel, Header, Message, Metadata, Schema
from mcap.stream_reader import StreamReader
from mcap.writer import Writer

from .recording import Event, encoded, read, sha256, verify_recording

PROFILE = "robot-learning-lab.recording.v1"
MAX_PACKAGE_BYTES = 128 * 1024 * 1024


def attachments_for(manifest):
    names = set()
    for name in manifest["source_hashes"]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("unsafe source attachment name")
        names.add("source/" + name)
    return {"manifest.json", "manifest.sha256"} | names


def pack(source, output):
    source, output = Path(source), Path(output)
    verification = verify_recording(source)
    manifest = read(source / "manifest.json")
    names = attachments_for(manifest)
    if sum((source / name).stat().st_size for name in names | {"events.jsonl"}) > MAX_PACKAGE_BYTES:
        raise ValueError("recording exceeds this bounded offline conversion limit")
    schema = encoded(Event.model_json_schema()).encode()
    with output.open("xb") as stream:
        writer = Writer(stream, enable_data_crcs=True)
        writer.start(profile=PROFILE)
        schema_id = writer.register_schema("RobotRecordingEvent", "jsonschema", schema)
        channels = {
            name: writer.register_channel(
                topic=name, message_encoding="json", schema_id=schema_id,
                metadata={"clock": manifest["clock"], "source_kind": manifest["source_kind"],
                          "signal": encoded(spec), "publish_time_semantics": "unknown; log_time fallback",
                          "capture_time_semantics": "retained inside event; not MCAP publish_time"},
            ) for name, spec in manifest["signals"].items()
        }
        writer.add_metadata("robot_recording", {
            "recording_id": manifest["recording_id"], "clock": manifest["clock"],
            "source_kind": manifest["source_kind"], "outcome": encoded(manifest["outcome"]),
        })
        for name in sorted(names):
            writer.add_attachment(0, 0, name, "application/octet-stream", (source / name).read_bytes())
        with (source / "events.jsonl").open("rb") as events:
            for line in events:
                event = Event.model_validate_json(line)
                if event.sequence > 2**32 - 1 or event.record_time_ns > 2**64 - 1:
                    raise ValueError("event exceeds native MCAP sequence/time range")
                writer.add_message(channels[event.signal], event.record_time_ns, line,
                                   event.record_time_ns, event.sequence)
        writer.finish()
    return {**verification, "mcap_sha256": sha256(output), "mcap_bytes": output.stat().st_size,
            "scope": "container conversion; original robot result preserved"}


def unpack(source, output, expected_sha256):
    source, output = Path(source), Path(output)
    if sha256(source) != expected_sha256:
        raise ValueError("MCAP differs from expected digest")
    attachments, channels, schemas, messages, metadata = {}, {}, {}, [], {}
    header_seen = False
    total = 0
    with source.open("rb") as stream:
        for record in StreamReader(stream, validate_crcs=True,
                                   record_size_limit=MAX_PACKAGE_BYTES).records:
            if isinstance(record, Header):
                if header_seen or record.profile != PROFILE:
                    raise ValueError("unsupported MCAP recording profile")
                header_seen = True
            if isinstance(record, (Schema, Channel)):
                mapping = schemas if isinstance(record, Schema) else channels
                if record.id in mapping and record != mapping[record.id]:
                    raise ValueError("conflicting MCAP schema/channel identity")
                mapping[record.id] = record
            elif isinstance(record, Attachment):
                if record.name in attachments:
                    raise ValueError("duplicate MCAP attachment")
                attachments[record.name] = record.data
                total += len(record.data)
            elif isinstance(record, Metadata):
                if record.name in metadata:
                    raise ValueError("duplicate recording metadata")
                metadata[record.name] = record.metadata
            elif isinstance(record, Message):
                channel = channels[record.channel_id]
                schema = schemas[channel.schema_id]
                if channel.message_encoding != "json" or schema.encoding != "jsonschema":
                    raise ValueError("unexpected event encoding")
                if schema.data != encoded(Event.model_json_schema()).encode():
                    raise ValueError("unexpected event schema")
                event = Event.model_validate_json(record.data)
                if (record.sequence != len(messages) or event.sequence != record.sequence
                        or event.record_time_ns != record.log_time
                        or record.publish_time != record.log_time
                        or event.signal != channel.topic):
                    raise ValueError("MCAP time, sequence or topic differs from event")
                messages.append((record.data, channel))
                total += len(record.data)
            if total > MAX_PACKAGE_BYTES:
                raise ValueError("expanded recording exceeds offline conversion limit")
        if stream.read(1):
            raise ValueError("unexpected bytes after MCAP footer")
    manifest = json.loads(attachments["manifest.json"])
    expected_metadata = {"robot_recording": {
        "recording_id": manifest["recording_id"], "clock": manifest["clock"],
        "source_kind": manifest["source_kind"], "outcome": encoded(manifest["outcome"]),
    }}
    if not header_seen or metadata != expected_metadata:
        raise ValueError("MCAP origin, outcome or clock metadata differs")
    topics = [channel.topic for channel in channels.values()]
    if len(topics) != len(set(topics)) or set(topics) != set(manifest["signals"]):
        raise ValueError("MCAP channel catalog differs from recording")
    if set(attachments) != attachments_for(manifest):
        raise ValueError("missing or unexpected MCAP attachment")
    if hashlib.sha256(attachments["manifest.json"]).hexdigest() != attachments["manifest.sha256"].decode().strip():
        raise ValueError("archived manifest digest differs")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256(attachments["source/" + name]).hexdigest() != expected:
            raise ValueError("archived source digest differs")
    events = b"".join(line for line, _ in messages)
    if hashlib.sha256(events).hexdigest() != manifest["events_sha256"]:
        raise ValueError("event bytes differ from original recording")
    for line, channel in messages:
        event = Event.model_validate_json(line)
        if (channel.metadata.get("clock") != manifest["clock"]
                or channel.metadata.get("source_kind") != manifest["source_kind"]
                or channel.metadata.get("signal") != encoded(manifest["signals"][event.signal])
                or channel.metadata.get("publish_time_semantics") != "unknown; log_time fallback"
                or channel.metadata.get("capture_time_semantics") != "retained inside event; not MCAP publish_time"):
            raise ValueError("MCAP channel semantics differ from recording")
    if output.exists():
        raise FileExistsError(output)
    # Validate the reconstructed canonical contract before publishing the output.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".mcap-restore-") as temp:
        temp = Path(temp)
        for name, data in attachments.items():
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (temp / "events.jsonl").write_bytes(events)
        verification = verify_recording(temp)
        output.mkdir(exist_ok=False)
        for path in temp.iterdir():
            path.rename(output / path.name)
    return {**verification, "mcap_sha256": expected_sha256,
            "scope": "CRC, byte identity and canonical semantics; not physical qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("pack", "unpack"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    if args.operation == "unpack" and not args.expected_sha256:
        parser.error("unpack requires --expected-sha256 from the export result")
    result = (pack(args.source, args.output) if args.operation == "pack" else
              unpack(args.source, args.output, args.expected_sha256))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
