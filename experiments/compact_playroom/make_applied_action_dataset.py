"""Derive GR00T button demonstrations with Berkeley's applied action limits.

The PPO teacher's raw actions were recorded before the simulation's action
term clipped them. This keeps every video/episode unchanged and replaces the
action labels and previous-action state with commands that reached the actuator.
Regenerate stats with NVIDIA's gr00t.data.stats before training.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if output.exists():
    raise FileExistsError(output)
shutil.copytree(source, output)

lower = np.array([-0.5, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
upper = np.array([0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
changed = previous_changed = frames = 0
files = []
for path in sorted((output / "data").rglob("*.parquet")):
    table = pq.read_table(path)
    index = table.schema.get_field_index("action")
    if index < 0:
        raise ValueError(f"Missing action column in {path}")
    raw = np.asarray(table.column(index).to_pylist(), dtype=np.float32)
    applied = np.clip(raw, lower, upper)
    changed += int(np.count_nonzero(np.any(raw != applied, axis=1)))
    frames += len(raw)
    table = table.set_column(
        index, table.schema.field(index),
        pa.array(applied.tolist(), type=table.schema.field(index).type),
    )
    state_index = table.schema.get_field_index("observation.state")
    state = np.asarray(table.column(state_index).to_pylist(), dtype=np.float32)
    if state.shape[1] not in (19, 25):
        raise ValueError(f"Expected 19-D or 25-D button state in {path}, got {state.shape}")
    previous = np.clip(state[:, -8:], lower, upper)
    previous_changed += int(np.count_nonzero(np.any(state[:, -8:] != previous, axis=1)))
    state[:, -8:] = previous
    table = table.set_column(
        state_index, table.schema.field(state_index),
        pa.array(state.tolist(), type=table.schema.field(state_index).type),
    )
    pq.write_table(table, path, compression="zstd")
    files.append({"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

for name in ("stats.json", "relative_stats.json"):
    (output / "meta" / name).unlink(missing_ok=True)
manifest = {
    "source": str(source),
    "change": "8-D action and previous-action state clipped to BerkeleyVelocityLeftArmAction limits; video and other state unchanged",
    "source_modality_sha256": hashlib.sha256((source / "meta" / "modality.json").read_bytes()).hexdigest(),
    "frames": frames,
    "frames_with_changed_action": changed,
    "frames_with_changed_previous_action": previous_changed,
    "limits_min": lower.tolist(),
    "limits_max": upper.tolist(),
    "parquet_files": files,
}
(output / "applied_action_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"output": str(output), "frames": frames, "frames_with_changed_action": changed, "frames_with_changed_previous_action": previous_changed}))
