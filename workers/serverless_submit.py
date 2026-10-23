"""Submit one job through the official Flash dev API and retain local evidence."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
from workers.s3_artifacts import (Artifacts, CONFIG, INPUT_BYTES, INPUT_FILES,
                                 client_from_auth, profile_auth, redact, safe_name)

TRANSFER = ROOT / "artifacts/isaac-pilot-transfer-v2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def payload_for(operation, bundle=None):
    alternate = bundle is not None
    bundle = Path(bundle) if alternate else TRANSFER / "bundles/pip"
    if operation == "probe" and not alternate:
        xml = (bundle / "model/supported.xml").read_text()
        return {"xml": xml, "sha256": digest(xml.encode())}
    seal = (bundle / "manifest.json").read_bytes()
    expected = digest(seal) if alternate else "c7b36cb135970a4d29be3366279c4505fefc59ba1cebdb288a48d5168f4c054e"
    if alternate and (bundle / "manifest.sha256").exists() and (bundle / "manifest.sha256").read_text().strip() != expected:
        raise ValueError("Alternate bundle seal changed")
    if digest(seal) != expected:
        raise ValueError("Pip bundle seal changed")
    files = {n: (TRANSFER / n).read_bytes() for n in ["isaac_pilot.py", "isaac_convert.py", "isaac-pilot.json", "asset_probe.py"]}
    for name, sha in json.loads(seal)["files"].items():
        safe_name(name)
        if not (bundle / name).parent.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Bundle path escapes its directory")
        if (bundle / name).is_symlink():
            from lab.artifact_store import validate_file
            validate_file(bundle / name, repo_root=ROOT)
        data = (bundle / name).read_bytes()
        if digest(data) != sha:
            raise ValueError("Bundle file changed: " + name)
        files["bundles/pip/" + name] = data
    files["bundles/pip/manifest.json"] = seal
    if operation == "probe":
        data = files["bundles/pip/model/supported.xml"]
        return {"xml": data.decode(), "sha256": digest(data), "bundle_sha256": expected}
    if operation == "control":
        for path in (ROOT / "workers/control_loop").rglob("*.py"):
            files["control/" + str(path.relative_to(ROOT / "workers/control_loop"))] = path.read_bytes()
        files["control/gaussian-constraints.txt"] = (ROOT / "workers/control_loop/gaussian-constraints.txt").read_bytes()
        files["control/isaac_convert.py"] = (ROOT / "workers/isaac_convert.py").read_bytes()
        files["control/settings.json"] = (ROOT / "learning/pip-control-v1.json").read_bytes()
        files["control/electromechanical_reference.py"] = (ROOT / "lab/electromechanical.py").read_bytes()
    return {"files": {n: {"sha256": digest(v), "base64": base64.b64encode(v).decode()} for n, v in files.items()}, "bundle_sha256": expected}


def add_control_request(payload, stage, extra=None, motor_checkpoint=None, selector_checkpoint=None):
    if extra is not None and not isinstance(extra, dict):
        raise ValueError("Control request must be a JSON object")
    request = {**(extra or {}), "stage": stage}
    if selector_checkpoint is not None:
        if stage != 'skills':raise ValueError('A selector checkpoint requires the skills stage')
        data = selector_checkpoint.read_bytes()
        payload['files']['control/selector.zip'] = {'sha256':digest(data),'base64':base64.b64encode(data).decode()}
        request['selector_checkpoint_sha256'] = digest(data)
        request['selector_checkpoint_origin'] = str(selector_checkpoint)
    request_data = json.dumps(request, sort_keys=True).encode()
    payload["files"]["control/request.json"] = {"sha256": digest(request_data), "base64": base64.b64encode(request_data).decode()}
    if stage in ("camera", "skills"):
        checkpoint = motor_checkpoint or ROOT / "artifacts/pip-control-serverless-v6/evidence/control/logs/rsl_rl/pip_residual_drive/2026-09-21_18-58-10/model_49.pt"
        for name, path in [("control/motor.pt", checkpoint), ("control/perception_control.py", ROOT / "lab/perception_control.py")]:
            data = path.read_bytes()
            payload["files"][name] = {"sha256": digest(data), "base64": base64.b64encode(data).decode()}


def s3_payload(payload, storage):
    if len(payload["files"]) > INPUT_FILES:
        raise ValueError("Too many input files")
    files, total = {}, 0
    for name, item in payload["files"].items():
        safe_name(name)
        data = base64.b64decode(item["base64"], validate=True)
        total += len(data)
        if total > INPUT_BYTES or digest(data) != item["sha256"]:
            raise ValueError("Input size/hash mismatch")
        files[name] = storage.put_bytes(data)
    return {"files": files, "bundle_sha256": payload["bundle_sha256"], "storage": dict(CONFIG)}


def request_envelope(payload, storage_auth=None):
    inputs = {"payload": payload}
    if storage_auth is not None:
        inputs["storage_auth"] = storage_auth
    return {"input": inputs}


def unpack_evidence(output, archive_b64, expected):
    data = base64.b64decode(archive_b64, validate=True)
    if digest(data) != expected:
        raise ValueError("Returned archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        infos = archive.infolist()
        if sum(i.file_size for i in infos) > 100_000_000:
            raise ValueError("Returned archive too large")
        for item in infos:
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe returned archive path")
        output.mkdir(exist_ok=False)
        archive.extractall(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation", choices=["probe", "isaac", "control", "agile"])
    p.add_argument("--stage", choices=["motor", "camera", "skills"], default="motor")
    p.add_argument("--bundle", type=Path, help="Alternate hash-verified sealed bundle")
    p.add_argument("--request-json", type=Path, help="Extra control request fields; CLI stage takes precedence")
    p.add_argument("--motor-checkpoint", type=Path, help="Retained motor checkpoint for camera/skill evaluation; bytes are hashed into the input manifest")
    p.add_argument("--selector-checkpoint", type=Path, help="Re-evaluate a retained skill selector without retraining it")
    p.add_argument("--s3", action="store_true", help="Use authenticated immutable S3 transport for control")
    p.add_argument("--url", default="http://localhost:8898")
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.s3 and args.operation not in {"control", "agile"}:
        p.error("--s3 is supported for control and agile jobs")
    if (args.request_json or args.motor_checkpoint or args.selector_checkpoint) and args.operation != "control":
        p.error("Control request/checkpoint options require a control job")
    if args.operation == "agile" and (not args.s3 or args.bundle is None):
        p.error("AGILE requires --s3 and an explicit --bundle")
    args.output.mkdir(parents=True, exist_ok=False)
    payload = payload_for(args.operation, args.bundle)
    if args.operation == "control":
        add_control_request(payload, args.stage, json.loads(args.request_json.read_text()) if args.request_json else None, args.motor_checkpoint, args.selector_checkpoint)
    identity = {"operation": args.operation, "payload_sha256": digest(json.dumps(payload, sort_keys=True).encode())}
    if "files" in payload:
        identity["files"] = {n: v["sha256"] for n, v in payload["files"].items()}
    (args.output / "inputs.json").write_text(json.dumps(identity, indent=2))
    source = args.output / "submitted-source"
    source.mkdir()
    for name, item in payload.get("files", {}).items():
        dest = source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(item["base64"]))
    (source / "worker.py").write_bytes((ROOT / "workers/serverless" / (args.operation + ".py")).read_bytes())
    if args.s3:
        (source / "s3_artifacts.py").write_bytes((ROOT / "workers/serverless/s3_artifacts.py").read_bytes())
    # Snapshot bytes are immutable before dispatch; use the existing local CAS.
    if source.absolute().is_relative_to(ROOT / "artifacts"):
        from lab.artifact_store import plan_migration, apply_migration
        apply_migration(ROOT, plan_migration(ROOT, [source.absolute()]))
    auth, storage = None, None
    try:
        if args.s3:
            auth = profile_auth("runpod")
            storage = Artifacts(client_from_auth(CONFIG, auth))
            payload = s3_payload(payload, storage)
            # Public transport identity excludes the separate authentication input.
            identity["transport"] = {"kind": "s3", "payload_sha256": digest(json.dumps(payload, sort_keys=True).encode()), "storage": dict(CONFIG)}
            (args.output / "inputs.json").write_text(json.dumps(identity, indent=2))
            (args.output / "transfer.json").write_text(json.dumps(payload, indent=2))
        request = urllib.request.Request(args.url.rstrip("/") + "/" + args.operation + "/runsync", data=json.dumps(request_envelope(payload, auth)).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1950) as response:
            raw = response.read()
        result = redact(json.loads(raw), auth)
        (args.output / "response.json").write_text(json.dumps(result, indent=2))
        worker = result.get("output", {})
        if "s3_evidence" in worker:
            if storage is None:
                raise ValueError("Unexpected S3 output on an inline request")
            verified = storage.download_run(worker["s3_evidence"], args.output / "evidence")
            (args.output / "s3-manifest.json").write_text(json.dumps(verified.pop("manifest"), indent=2))
            (args.output / "download-verification.json").write_text(json.dumps(verified, indent=2))
        if "archive_base64" in worker:
            unpack_evidence(args.output / "evidence", worker["archive_base64"], worker["archive_sha256"])
        # Flash can wrap a worker error in an HTTP-200 COMPLETED response.
        if result.get("status") != "COMPLETED" or worker.get("status") not in ["pass", "partial"]:
            raise RuntimeError("Remote worker failed; inspect response.json")
    except Exception as exc:
        (args.output / "client-error.json").write_text(json.dumps({"error_type": type(exc).__name__, "note": "Client timeout does not cancel the remote job; inspect the endpoint before retrying. Authentication values are omitted."}, indent=2))
        raise SystemExit("Submission or transfer failed; inspect the retained credential-free client-error.json") from None
    finally:
        if args.output.absolute().is_relative_to(ROOT / "artifacts"):
            from lab.artifact_store import plan_migration, apply_migration
            apply_migration(ROOT, plan_migration(ROOT, [args.output.absolute()]))


if __name__ == "__main__":
    main()
