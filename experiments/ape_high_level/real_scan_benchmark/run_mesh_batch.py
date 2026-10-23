"""Apply the same provisional closed visual-route probe to frozen real scans."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from prepare_arkit_corpus import ASSETS


HERE = Path(__file__).resolve().parent
SCREEN = HERE / "splat_recorded_view_screen.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    ids = json.loads(SCREEN.read_text())["qualified_appearance_video_ids"]
    if args.limit is not None:
        ids = ids[:args.limit]
    rows = []
    for index, scene_id in enumerate(ids, 1):
        output = ASSETS / scene_id / "mesh_probe"
        completed = subprocess.run(
            [sys.executable, str(HERE / "run_mesh_probe.py"),
             "--scene", scene_id, "--steps", str(args.steps),
             "--output", str(output)], capture_output=True, text=True,
            timeout=240, check=False)
        trace_path = output / "trace.json"
        if completed.returncode == 0 and trace_path.exists():
            result = json.loads(trace_path.read_text())
            row = {key: value for key, value in result.items() if key != "trace"}
        else:
            row = {"scene_id": scene_id, "status": "probe_failed",
                   "reason": completed.stderr[-500:]}
        rows.append(row)
        print(f"{index}/{len(ids)} {scene_id} {row['status']} "
              f"stop={row.get('actor_declared_standoff')} "
              f"truth={row.get('truth_standoff_within_band')}", flush=True)
        (ASSETS / "mesh_probe_batch.json").write_text(json.dumps({
            "schema": "ape.real-scan-mesh-probe-batch.v1",
            "candidate_count": len(ids), "completed_count": len(rows),
            "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
