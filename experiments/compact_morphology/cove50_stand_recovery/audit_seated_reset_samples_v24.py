"""Check sampled CUDA seated resets against the exact native MuJoCo body."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").is_file()
    and (parent / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml").is_file()
)
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source_path: Path, out_path: Path) -> dict:
    source = source_path.read_text()
    report = json.loads(source[source.index('{\n  "schema"'):])
    assert report["device"] == "cuda:0"
    model = mujoco.MjModel.from_xml_path(str(BODY))
    floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_geom >= 0, "Bare native body must contain its floor geom"
    sole_geoms = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                   f"{side}_sole") for side in ("left", "right")]
    assert all(geom >= 0 for geom in sole_geoms)
    rows = []
    for bucket, examples in report["accepted_source_state_samples_by_attempt"].items():
        for example in examples:
            data = mujoco.MjData(model)
            data.qpos[:] = example["qpos"]
            data.qvel[:] = example["qvel"]
            mujoco.mj_forward(model, data)
            floor_dists = [float(c.dist) for c in data.contact[:data.ncon]
                           if floor_geom in (c.geom1, c.geom2)]
            self_contacts = [float(c.dist) for c in data.contact[:data.ncon]
                             if floor_geom not in (c.geom1, c.geom2)]
            sole_lowest = []
            for geom in sole_geoms:
                half = model.geom_size[geom, :3]
                corners = np.array([[x * half[0], y * half[1], z * half[2]]
                                    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                rotation = data.geom_xmat[geom].reshape(3, 3)
                sole_lowest.append(float(np.min((corners @ rotation.T)[:, 2]
                                               + data.geom_xpos[geom, 2])))
            rows.append({
                "sample_bucket": bucket, "phase": example["phase"],
                "attempt": int(example["attempt"]), "world": example["world"],
                "root_z_m": float(data.qpos[2]),
                "deepest_floor_contact_m": min(floor_dists) if floor_dists else None,
                "deepest_self_contact_m": min(self_contacts) if self_contacts else None,
                "bare_body_sole_lowest_z_m": sole_lowest,
                "source_predicted_sole_floor_gaps_m": example["predicted_sole_floor_gaps_m"],
                "floor_penetration_gt_1mm": any(x < -0.001 for x in floor_dists),
                "self_penetration_gt_1mm": any(x < -0.001 for x in self_contacts),
            })
    result = {
        "schema": "robot-learning-lab.cove50-seated-reset-native-audit.v24",
        "source_log": str(source_path.relative_to(ROOT)),
        "source_log_sha256": sha(source_path),
        "body_sha256": sha(BODY),
        "sampled_entries": len(rows),
        "failed_entries": sum(r["floor_penetration_gt_1mm"] or r["self_penetration_gt_1mm"]
                              for r in rows),
        "rows": rows,
        "claim_limit": "Stratified saved CUDA qpos checked in the bare native body. The composed MJLab scene is not reconstructed, so this is a geometry cross-check, not exact contact parity or a learned score.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_seated_reset_samples_v24.py CUDA_LOG OUT_JSON")
    print(json.dumps({k: v for k, v in audit(Path(sys.argv[1]), Path(sys.argv[2])).items()
                      if k != "rows"}, indent=2))
