"""Boto3 transport for immutable, hash-addressed robot job files.

Kept beside the endpoint so Flash's native local-module bundler includes it.
Credentials are function arguments only, never manifests or subprocess environment.
"""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid

CONFIG = {"bucket": "8f4psp0z8g", "region": "EU-RO-1",
          "endpoint": "https://s3api-eu-ro-1.runpod.io/", "prefix": "robot-learning-lab/v1"}
INPUT_FILES = 600
INPUT_BYTES = 150_000_000
OUTPUT_FILES = 5000
OUTPUT_BYTES = 500_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name):
    if not isinstance(name, str):
        raise ValueError("unsafe artifact path")
    path = PurePosixPath(name)
    if (not name or not path.parts or path.is_absolute() or
            ".." in path.parts or "\\" in name or "\x00" in name or
            path.as_posix() != name or path.parts.count("source") > 1 or
            "artifacts" in path.parts or ".artifact-store" in path.parts):
        raise ValueError("unsafe artifact path")
    return path


def object_key(sha):
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("invalid artifact SHA-256")
    return f"{CONFIG['prefix']}/objects/sha256/{sha[:2]}/{sha[2:]}"


def public_config(config):
    if config != CONFIG:
        raise ValueError("unrecognized S3 storage destination")
    return dict(CONFIG)


def redact(value, credentials=None):
    """Remove known credential values and credential fields before retaining errors."""
    sensitive = {str(v) for v in (credentials or {}).values() if isinstance(v, str) and v}
    if isinstance(value, str):
        for secret in sensitive:
            value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k in {"storage_auth", "aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
                    else redact(v, credentials)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, credentials) for v in value]
    return value


def client_from_auth(config, auth):
    public_config(config)
    allowed = {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
    if not isinstance(auth, dict) or not {"aws_access_key_id", "aws_secret_access_key"}.issubset(auth) or set(auth) - allowed:
        raise ValueError("S3 credentials are unavailable")
    import boto3
    from botocore.config import Config
    return boto3.client("s3", endpoint_url=config["endpoint"], region_name=config["region"],
                        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                      connect_timeout=10, read_timeout=60, retries={"max_attempts": 2},
                                      request_checksum_calculation="when_required", response_checksum_validation="when_required"), **auth)


def profile_auth(profile="runpod"):
    import boto3
    credentials = boto3.Session(profile_name=profile).get_credentials()
    if credentials is None:
        raise ValueError("S3 profile credentials are unavailable")
    frozen = credentials.get_frozen_credentials()
    result = {"aws_access_key_id": frozen.access_key, "aws_secret_access_key": frozen.secret_key}
    if frozen.token:
        result["aws_session_token"] = frozen.token
    return result


class Artifacts:
    def __init__(self, client, config=None):
        self.config = public_config(config or CONFIG)
        self.client = client
        self.verified = set()

    def _validate(self, ref):
        if (ref.get("key") != object_key(ref.get("sha256")) or
                type(ref.get("size")) is not int or not 0 <= ref["size"] <= OUTPUT_BYTES):
            raise ValueError("invalid S3 object reference")

    def _read(self, ref, target=None):
        self._validate(ref)
        response = self.client.get_object(Bucket=self.config["bucket"], Key=ref["key"])
        body = response["Body"]
        size, sha = 0, hashlib.sha256()
        try:
            while block := body.read(1024 * 1024):
                size += len(block)
                if size > ref["size"]:
                    raise ValueError("S3 object exceeds declared size")
                sha.update(block)
                if target is not None:
                    target.write(block)
        finally:
            body.close()
        if size != ref["size"] or sha.hexdigest() != ref["sha256"]:
            raise ValueError("S3 object size/hash mismatch")
        self.verified.add(ref["sha256"])

    def put_stream(self, stream, sha, size):
        ref = {"sha256": sha, "size": size, "key": object_key(sha)}
        self._validate(ref)
        if sha in self.verified:
            return ref
        try:
            head = self.client.head_object(Bucket=self.config["bucket"], Key=ref["key"])
        except Exception as error:
            code = getattr(error, "response", {}).get("Error", {}).get("Code")
            if str(code) not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            if head["ContentLength"] != size:
                raise ValueError("existing S3 object size mismatch; not overwritten")
            self._read(ref)
            return ref
        # Callers verify source bytes before upload. A concurrent equal-key upload
        # has identical content; no unique run files are overwritten or deleted.
        request = dict(Bucket=self.config["bucket"], Key=ref["key"],
                       Body=stream if size else b"", Metadata={"sha256": sha})
        # Runpod S3 signs an explicit ContentLength=0 differently (403
        # SignatureDoesNotMatch). Let boto3 handle the empty body; it is still
        # a real, hash-verified empty object, not an omitted artifact.
        if size:
            request["ContentLength"] = size
        self.client.put_object(**request)
        self.verified.add(sha)
        return ref

    def put_bytes(self, data):
        return self.put_stream(io.BytesIO(data), digest(data), len(data))

    def put_file(self, path):
        with Path(path).open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(0)
            return self.put_stream(stream, sha, size)

    def materialize(self, files, destination, *, max_files=INPUT_FILES, max_bytes=INPUT_BYTES):
        if not isinstance(files, dict) or len(files) > max_files:
            raise ValueError("too many artifact files")
        for name, ref in files.items():
            safe_name(name)
            self._validate(ref)
        if sum(ref["size"] for ref in files.values()) > max_bytes:
            raise ValueError("artifact transfer exceeds size limit")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        for name, ref in files.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
            with temp.open("xb") as stream:
                self._read(ref, stream)
            temp.rename(target)
        return {"status": "pass", "files": len(files), "bytes": sum(ref["size"] for ref in files.values())}

    def retain_tree(self, directory, run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
            raise ValueError("invalid S3 run id")
        directory = Path(directory)
        files, total = {}, 0
        for folder, dirs, names in os.walk(directory, followlinks=False):
            if any((Path(folder) / name).is_symlink() for name in dirs):
                raise ValueError("output directory symlink forbidden")
            for name in sorted(names):
                path = Path(folder) / name
                relative = path.relative_to(directory).as_posix()
                safe_name(relative)
                if path.is_symlink() or not path.is_file():
                    raise ValueError("output must be an ordinary file")
                total += path.stat().st_size
                if len(files) >= OUTPUT_FILES or total > OUTPUT_BYTES:
                    raise ValueError("output artifact limit exceeded")
                files[relative] = self.put_file(path)
        manifest = {"schema": "robot-learning-lab.s3-run.v1", "run_id": run_id, "files": files}
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ref = self.put_bytes(encoded)
        record_key = f"{self.config['prefix']}/runs/{run_id}/{ref['sha256']}.json"
        self.client.put_object(Bucket=self.config["bucket"], Key=record_key,
                               Body=json.dumps({"manifest": ref}).encode())
        return {"storage": dict(self.config), "manifest": ref, "run_id": run_id,
                "files": len(files), "bytes": total, "record_key": record_key}

    def download_run(self, reference, destination):
        public_config(reference["storage"])
        if reference["manifest"]["size"] > 2_000_000:
            raise ValueError("returned run manifest too large")
        stream = io.BytesIO()
        self._read(reference["manifest"], stream)
        manifest = json.loads(stream.getvalue())
        if manifest.get("schema") != "robot-learning-lab.s3-run.v1" or manifest.get("run_id") != reference["run_id"]:
            raise ValueError("returned run manifest identity mismatch")
        result = self.materialize(manifest["files"], destination, max_files=OUTPUT_FILES, max_bytes=OUTPUT_BYTES)
        return {**result, "manifest": manifest, "manifest_sha256": reference["manifest"]["sha256"]}
