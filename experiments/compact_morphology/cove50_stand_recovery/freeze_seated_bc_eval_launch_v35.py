"""Build the V35 worker transport and review manifests without compute."""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile

from create_seated_bc_eval_guard_v35 import (ART, ATTESTATION_SCHEMA, BUNDLE,
    CAP_S, GPU, IMAGE, MANIFEST, MANIFEST_SCHEMA, MAX_POD_S, MAX_SPEND,
    PLAN, PLAN_SHA256, RATE, READINESS_S, RELAY, ROOT, STOP_RESERVE_S)


APE = ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk"
POLLEN = ROOT / ".venv-control/src/microduck_rl"
POLLEN_COMMIT = "0bf989701f14d4008c866ffd6f09067870f99379"
EAGER = ("reach", "grasp", "grasp_tracking", "place_tracking", "place_stages", "foot_upright")
ASSETS = {
    "upright_endpoint_screen": "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json",
    "upright_left_mirror_endpoint": "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json",
}
SOURCE_MANIFEST = ART / "source-manifest.json"
AUDIT = ART / "import-closure-audit.json"
GATE = ROOT / "experiments/compact_morphology/cove50_stand_recovery/checkout_import_gate_v35.py"
GUARD = ROOT / "experiments/compact_morphology/cove50_stand_recovery/create_seated_bc_eval_guard_v35.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def freeze(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Frozen V35 launch artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def check_pollen() -> None:
    head = subprocess.check_output(["git", "-C", str(POLLEN), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(POLLEN), "status", "--porcelain"], text=True).strip()
    if head != POLLEN_COMMIT or dirty:
        raise RuntimeError("V31d Pollen source is not the clean pinned commit")


def audit_imports(plan: dict) -> dict:
    parsed = ast.parse((APE / "task.py").read_text())
    eager = tuple(alias.name for node in parsed.body
                  if isinstance(node, ast.ImportFrom) and node.level == 1
                  and node.module is None for alias in node.names
                  if alias.name in EAGER)
    foot = (APE / "foot_upright.py").read_text()
    if (eager != EAGER or foot.count("register_mjlab_task(") != 5
            or 'pose_spec = json.loads((LEFT_ENDPOINT if side == "left" else ENDPOINT).read_text())' not in foot):
        raise RuntimeError("V35 eager task registration or endpoint read changed")
    for name, rel in ASSETS.items():
        if Path(rel).name not in foot or sha(ROOT / rel) != plan["source_sha256"]["v31d_material_" + name]:
            raise RuntimeError(f"V35 eager endpoint differs: {name}")
    return {"schema": "robot-learning-lab.cove50-seated-bc-eval.v35.import-audit",
            "status": "local_static_audit_only",
            "plan_sha256": PLAN_SHA256,
            "eager_task_modules": list(eager),
            "eager_module_sha256": {name: sha(APE / f"{name}.py") for name in ("task", *EAGER)},
            "eager_assets": {name: {"path": rel, "sha256": sha(ROOT / rel),
                                    "local_source_is_symlink": (ROOT / rel).is_symlink(),
                                    "packed_as_regular_file": True}
                             for name, rel in ASSETS.items()},
            "worker_pythonpath_required": "/root/robot-learning-lab/experiments/compact_morphology/mjlab_ape/src:/root/robot-learning-lab",
            "claim_limit": "Linux origin, installed runtime, provider and CUDA still require live attestation."}


def members(plan: dict) -> list[str]:
    files = set(plan["source_paths"].values())
    files.update({str((APE / f"{name}.py").relative_to(ROOT)) for name in EAGER})
    files.update(ASSETS.values())
    files.update((str(PLAN.relative_to(ROOT)),
                  str((ART / "attempt-ledger.json").relative_to(ROOT)),
                  str(SOURCE_MANIFEST.relative_to(ROOT)),
                  str(GATE.relative_to(ROOT))))
    roots = (ROOT / "experiments/compact_morphology/mjlab_ape", POLLEN)
    for directory in roots:
        for path in directory.rglob("*"):
            if (path.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".DS_Store")
                                           or part.startswith("._") for part in path.relative_to(directory).parts)):
                files.add(str(path.relative_to(ROOT)))
    return sorted(files)


def bundle(files: list[str]) -> bytes:
    memory = io.BytesIO()
    with gzip.GzipFile(fileobj=memory, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for rel in files:
                source = ROOT / rel
                if not source.is_file():
                    raise RuntimeError(f"V35 bundle member missing: {rel}")
                content = source.read_bytes()  # Dereference local artifact-store symlinks.
                item = tarfile.TarInfo(rel)
                item.mode = 0o755 if os.access(source, os.X_OK) else 0o644
                item.size = len(content)
                item.mtime = item.uid = item.gid = 0
                item.uname = item.gname = ""
                archive.addfile(item, io.BytesIO(content))
    return memory.getvalue()


def main() -> None:
    if os.environ.get("COPYFILE_DISABLE") != "1":
        raise RuntimeError("Set COPYFILE_DISABLE=1 for AppleDouble-free V35 source packaging")
    if sha(PLAN) != PLAN_SHA256:
        raise RuntimeError("Reviewed V35 plan bytes differ")
    plan = json.loads(PLAN.read_text())
    if set(plan["source_paths"]) != set(plan["source_sha256"]):
        raise RuntimeError("V35 plan source key set differs")
    for name, rel in plan["source_paths"].items():
        if sha(ROOT / rel) != plan["source_sha256"][name]:
            raise RuntimeError(f"V35 frozen source differs: {name}")
    check_pollen()
    audit = audit_imports(plan)
    freeze(AUDIT, canonical(audit))
    freeze(SOURCE_MANIFEST, canonical({
        "schema": "robot-learning-lab.cove50-seated-bc-eval.v35.source-manifest",
        "plan_sha256": PLAN_SHA256,
        "v31d_installed_closure_content_sha256": plan["v31d_installed_closure_content_sha256"],
        "pollen_commit": POLLEN_COMMIT,
        "eager_source_sha256": audit["eager_module_sha256"],
        "eager_assets": {name: {"path": row["path"], "sha256": row["sha256"]}
                         for name, row in audit["eager_assets"].items()},
        "checkout_gate_path": str(GATE.relative_to(ROOT)),
        "checkout_gate_sha256": sha(GATE),
    }))
    payload = bundle(members(plan))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for name, rel in plan["source_paths"].items():
            entry = archive.getmember(rel)
            stream = archive.extractfile(entry)
            if not entry.isfile() or stream is None or hashlib.sha256(stream.read()).hexdigest() != plan["source_sha256"][name]:
                raise RuntimeError(f"V35 bundled frozen source differs: {name}")
        for row in audit["eager_assets"].values():
            entry = archive.getmember(row["path"])
            if not entry.isfile() or hashlib.sha256(archive.extractfile(entry).read()).hexdigest() != row["sha256"]:
                raise RuntimeError("V35 bundled eager endpoint differs")
        for local in (SOURCE_MANIFEST, GATE, ART / "attempt-ledger.json"):
            rel = str(local.relative_to(ROOT))
            entry = archive.getmember(rel)
            if not entry.isfile() or hashlib.sha256(archive.extractfile(entry).read()).hexdigest() != sha(local):
                raise RuntimeError(f"V35 bundled launch member differs: {rel}")
    freeze(BUNDLE, payload)
    freeze(MANIFEST, canonical({
        "schema": MANIFEST_SCHEMA,
        "plan_path": str(PLAN.relative_to(ROOT)), "plan_sha256": PLAN_SHA256,
        "guard_source_path": str(GUARD.relative_to(ROOT)), "guard_source_sha256": sha(GUARD),
        "relay_source_path": str(RELAY.relative_to(ROOT)),
        "relay_source_sha256": sha(RELAY),
        "checkout_gate_path": str(GATE.relative_to(ROOT)),
        "checkout_gate_sha256": sha(GATE),
        "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "source_manifest_sha256": sha(SOURCE_MANIFEST),
        "bundle_path": str(BUNDLE.relative_to(ROOT)), "bundle_sha256": sha(BUNDLE),
        "attestation_schema": ATTESTATION_SCHEMA,
        "image": IMAGE, "gpu_id": GPU, "cloud_type": "SECURE", "gpu_count": 1,
        "listed_rate_usd_per_hour": RATE, "max_pod_wall_s": MAX_POD_S,
        "gpu_listed_spend_cap_usd": MAX_SPEND, "effective_cap_s": CAP_S,
        "stop_reserve_s": STOP_RESERVE_S, "max_start_readiness_s": READINESS_S,
        "sentinel_visits_max": 1, "original_visits_max": 24,
        "container_disk_in_gb": 30, "volume_in_gb": 10,
        "ssh": True, "ports": ["22/tcp"],
    }))
    print(json.dumps({"plan_sha256": PLAN_SHA256, "bundle_sha256": sha(BUNDLE),
                      "manifest_sha256": sha(MANIFEST), "source_count": len(plan["source_paths"]),
                      "bundle_member_count": len(members(plan))}, sort_keys=True))


if __name__ == "__main__":
    main()
