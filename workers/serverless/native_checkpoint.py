"""Validate GR00T checkpoint manifests and downloaded inference files."""
import hashlib
from pathlib import Path
import re

try:
    from .s3_artifacts import object_key, safe_name
except ImportError:  # Flash bundles local worker modules as top-level imports.
    from s3_artifacts import object_key, safe_name


SCHEMA = "robot-learning-lab.native-checkpoint.v2"
RESUME_ONLY = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def validate_hf_checkpoint_reference(reference):
    """Check the inference-only Hugging Face manifest before a paid evaluation."""
    if reference.get("schema") != "robot-learning-lab.hf-checkpoint.v1":
        raise ValueError("unrecognized Hugging Face checkpoint reference")
    if (not isinstance(reference.get("repo_id"), str) or
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", reference["repo_id"]) or
            not isinstance(reference.get("revision"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", reference["revision"])):
        raise ValueError("invalid Hugging Face checkpoint identity")
    files = reference.get("files")
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError("invalid Hugging Face checkpoint file manifest")
    total = 0
    for name, item in files.items():
        safe_name(name)
        if (not isinstance(item, dict) or
                not isinstance(item.get("sha256"), str) or
                not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or
                type(item.get("bytes")) is not int or item["bytes"] < 0):
            raise ValueError("invalid Hugging Face checkpoint file entry")
        total += item["bytes"]
    if total != reference.get("bytes") or len(files) != reference.get("file_count"):
        raise ValueError("Hugging Face checkpoint manifest totals mismatch")
    if any(name in RESUME_ONLY for name in files):
        raise ValueError("resume-only state belongs outside the inference archive")
    required = {"config.json", "model.safetensors.index.json", "statistics.json",
                "processor_config.json"}
    if not required.issubset(files) or not any(
            name.startswith("model-") and name.endswith(".safetensors") for name in files):
        raise ValueError("Hugging Face checkpoint lacks required inference files")
    return files


def verify_hf_checkpoint_tree(reference, destination):
    """Hash every downloaded inference file against its pinned manifest."""
    files = validate_hf_checkpoint_reference(reference)
    destination = Path(destination)
    for name, item in files.items():
        path = destination / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"missing or wrong-size checkpoint file: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                raise ValueError(f"checkpoint SHA-256 mismatch: {name}")
    return {"status": "pass", "files": len(files), "bytes": reference["bytes"],
            "verified": files, "excluded_resume_files": sorted(RESUME_ONLY)}


def validate_checkpoint_reference(reference, config):
    if reference.get("schema") != SCHEMA:
        raise ValueError("unrecognized native checkpoint reference")
    expected_storage = {key: config[key] for key in ("bucket", "region", "endpoint")}
    if reference.get("storage") != expected_storage:
        raise ValueError("native checkpoint storage mismatch")
    files = reference.get("files")
    if not isinstance(files, dict) or not files or len(files) > 100:
        raise ValueError("invalid native checkpoint file manifest")
    total = 0
    for name, item in files.items():
        safe_name(name)
        if not isinstance(item, dict):
            raise ValueError("invalid native checkpoint file entry")
        sha = item.get("sha256")
        size = item.get("bytes")
        if item.get("key") != object_key(sha) or type(size) is not int or size < 0:
            raise ValueError("invalid native checkpoint object reference")
        total += size
    if total != reference.get("bytes") or len(files) != reference.get("file_count"):
        raise ValueError("native checkpoint manifest totals mismatch")
    return files


def inference_files(reference, config):
    files = validate_checkpoint_reference(reference, config)
    selected = {name: item for name, item in files.items() if name not in RESUME_ONLY}
    required = {"config.json", "model.safetensors.index.json", "statistics.json",
                "processor_config.json"}
    if not required.issubset(selected):
        raise ValueError("native checkpoint lacks required inference configuration")
    shards = [name for name in selected if name.startswith("model-") and name.endswith(".safetensors")]
    if not shards:
        raise ValueError("native checkpoint lacks model shards")
    return selected


def materialize_inference_checkpoint(client, config, reference, destination):
    files = inference_files(reference, config)
    total = sum(item["bytes"] for item in files.values())
    if len(files) > 50 or total > 12_000_000_000:
        raise ValueError("inference checkpoint transfer exceeds bounded size")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    verified = {}
    for name, item in sorted(files.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.get_object(Bucket=config["bucket"], Key=item["key"])
        body = response["Body"]
        size = 0
        digest = hashlib.sha256()
        try:
            with target.open("xb") as stream:
                while block := body.read(8 * 1024 * 1024):
                    size += len(block)
                    if size > item["bytes"]:
                        raise ValueError("checkpoint object exceeds declared size")
                    digest.update(block)
                    stream.write(block)
        finally:
            body.close()
        if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("checkpoint object size/hash mismatch")
        verified[name] = {"sha256": item["sha256"], "bytes": size}
    excluded = sorted(set(reference["files"]) - set(files))
    return {"status": "pass", "files": len(files), "bytes": total,
            "verified": verified, "excluded_resume_files": excluded}
