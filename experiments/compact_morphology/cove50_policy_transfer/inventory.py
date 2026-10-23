"""Snapshot the retained ape policy catalog for an explicit transfer audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "skills/ape/registry.json"
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008/policy-inventory.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    rows = []
    for item in registry["policies"]:
        ckpt = ROOT / item["local_checkpoint"]
        if not ckpt.is_file() or sha(ckpt) != item["sha256"]:
            raise ValueError(f"Retained checkpoint missing or changed: {item['id']}")
        rows.append({
            "id": item["id"], "task": item["task"],
            "historical_source_dof": item["source_dof"],
            "checkpoint": item["local_checkpoint"],
            "checkpoint_sha256": item["sha256"],
            "historical_result": item["result"],
            "cove_transfer_status": "not_yet_retrained_or_qualified",
            "interface_change": "Cove has 21 axes; any 22-axis actor must explicitly remove head_pitch from actions and observations",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "registry": str(REGISTRY.relative_to(ROOT)),
        "registry_sha256": sha(REGISTRY),
        "current_design": "cove-50-direct-frame",
        "policy_count": len(rows),
        "policies": rows,
    }, indent=2) + "\n")
    print(f"Verified {len(rows)} retained policy checkpoints; wrote {OUT}")


if __name__ == "__main__":
    main()
