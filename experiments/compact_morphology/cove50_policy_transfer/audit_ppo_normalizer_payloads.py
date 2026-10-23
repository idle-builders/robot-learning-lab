"""Inspect retained PPO actor normalizer storage without executing pickle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickletools
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
INTERFACE = ROOT / "artifacts/ape-policy-interface-audit-20261012/report.json"
POINTER = ROOT / "robot/ape-current-design.json"
REGISTRY = ROOT / "skills/ape/registry.json"
OUT = ROOT / "artifacts/ape-policy-ppo-normalizers-20261013"
KEYS = ("obs_normalizer._mean", "obs_normalizer._var",
        "obs_normalizer._std", "obs_normalizer.count")
STRING_OPS = {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_normalizer(path: Path, width: int) -> dict:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Expected PyTorch ZIP checkpoint: {path}")
    with zipfile.ZipFile(path) as archive:
        pickle_name = next((name for name in archive.namelist()
                            if name.endswith("/data.pkl")), None)
        if pickle_name is None:
            raise ValueError(f"Missing checkpoint pickle metadata: {path}")
        prefix = pickle_name.removesuffix("data.pkl")
        if archive.read(prefix + "byteorder") != b"little":
            raise ValueError(f"Unsupported storage byte order: {path}")
        ops = list(pickletools.genops(archive.read(pickle_name)))
        actor_start = next(i for i, (op, value, _) in enumerate(ops)
                           if op.name in STRING_OPS and value == "actor_state_dict")
        critic_start = next(i for i, (op, value, _) in enumerate(ops)
                            if op.name in STRING_OPS and value == "critic_state_dict")
        if actor_start >= critic_start:
            raise ValueError(f"Unexpected actor/critic storage order: {path}")
        payloads = {}
        for i in range(actor_start + 1, critic_start):
            op, key, _ = ops[i]
            if op.name not in STRING_OPS or key not in KEYS:
                continue
            if key in payloads:
                raise ValueError(f"Duplicate actor normalizer key {key}: {path}")
            stop = next((j for j in range(i + 1, critic_start)
                         if ops[j][0].name == "BINPERSID"), None)
            if stop is None:
                raise ValueError(f"Missing tensor storage for {key}: {path}")
            storage_ids = [value for code, value, _ in ops[i + 1:stop]
                           if code.name in STRING_OPS and isinstance(value, str)
                           and value.isdecimal()]
            if len(storage_ids) != 1:
                raise ValueError(f"Unresolved tensor storage for {key}: {path}")
            raw = archive.read(prefix + "data/" + storage_ids[0])
            expected_bytes = 8 if key.endswith("count") else width * 4
            if len(raw) != expected_bytes:
                raise ValueError(f"Normalizer storage width mismatch for {key}: {path}")
            payloads[key] = {
                "storage_sha256": hashlib.sha256(raw).hexdigest(),
                "values": np.frombuffer(raw, dtype="<i8" if key.endswith("count")
                                        else "<f4"),
            }
    if set(payloads) != set(KEYS):
        raise ValueError(f"Missing actor normalizer fields: {path}")
    mean = payloads["obs_normalizer._mean"]["values"]
    var = payloads["obs_normalizer._var"]["values"]
    std = payloads["obs_normalizer._std"]["values"]
    count = int(payloads["obs_normalizer.count"]["values"][0])
    if (not all(np.isfinite(array).all() for array in (mean, var, std)) or
            (var < 0).any() or (std < 0).any() or count <= 0):
        raise ValueError(f"Invalid actor normalizer payload: {path}")
    if np.max(np.abs(std * std - var)) > 1e-4:
        raise ValueError(f"Actor normalizer variance/std disagree: {path}")
    return {
        "actor_input_width": width,
        "normalizer_vector_width": len(mean),
        "normalizer_count": count,
        "zero_std_indices": np.flatnonzero(std == 0).tolist(),
        "zero_std_feature_count": int(np.count_nonzero(std == 0)),
        "smallest_positive_std": float(std[std > 0].min()) if (std > 0).any() else None,
        "largest_std": float(std.max()),
        "storage_sha256": {key: payloads[key]["storage_sha256"] for key in KEYS},
        "qualification": "payload_present_and_width_checked_no_first_observation_or_rollout",
    }


def main() -> None:
    interface = json.loads(INTERFACE.read_text())
    pointer = json.loads(POINTER.read_text())
    if (interface["pointer_sha256"] != sha(POINTER) or
            interface["registry_sha256"] != sha(REGISTRY) or
            pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design"):
        raise ValueError("Selected body or policy inventory changed")
    rows = []
    for policy in interface["policies"]:
        shape = policy["checkpoint_actor_shape"]
        if shape["format"] != "rsl_rl":
            continue
        path = ROOT / policy["checkpoint"]
        if sha(path) != policy["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {policy['id']}")
        rows.append({"id": policy["id"], "checkpoint": policy["checkpoint"],
                     "checkpoint_sha256": policy["checkpoint_sha256"],
                     **actor_normalizer(path, shape["observation_features"])})
    if len(rows) != 24:
        raise ValueError(f"Expected 24 PPO policies, found {len(rows)}")
    report = {
        "schema": "robot-learning-lab.ape-ppo-normalizer-payloads.v1",
        "role": "saved_checkpoint_payload_audit_no_rollout",
        "source_sha256": {"script": sha(Path(__file__)),
                          "interface_report": sha(INTERFACE),
                          "pointer": sha(POINTER), "registry": sha(REGISTRY)},
        "policy_count": len(rows),
        "all_actor_widths_match": all(row["actor_input_width"] ==
                                      row["normalizer_vector_width"] for row in rows),
        "policies_with_zero_std_features": sum(row["zero_std_feature_count"] > 0
                                              for row in rows),
        "cases": rows,
        "interpretation": (
            "All saved PPO actor normalizer mean, variance, standard deviation "
            "and count storages are present and match checkpoint input widths. "
            "Zero standard-deviation entries are recorded, not treated as a "
            "division rule; load the source normalizer implementation exactly. "
            "This static ZIP/pickle-opcode inspection does not execute pickle, "
            "normalize a first observation, load a selected body or replay a policy."),
    }
    matrix = {
        "schema": "robot-learning-lab.ape-ppo-normalizer-matrix.v1",
        "role": report["role"],
        "source_report": str((OUT / "report.json").relative_to(ROOT)),
        "policy_count": len(rows),
        "cases": [{key: row[key] for key in (
            "id", "actor_input_width", "normalizer_vector_width", "normalizer_count",
            "zero_std_feature_count", "zero_std_indices", "smallest_positive_std",
            "largest_std", "qualification")} for row in rows],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    matrix["source_report_sha256"] = sha(OUT / "report.json")
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Verified actor normalizer payloads for {len(rows)} PPO checkpoints")


if __name__ == "__main__":
    main()
