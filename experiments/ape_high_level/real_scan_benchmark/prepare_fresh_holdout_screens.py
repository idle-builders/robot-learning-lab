"""Screen the prospectively frozen, capture-disjoint indoor holdout before policy runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from screen_new_capture_owner_routes import mesh_bounds


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRECOMMIT = ROOT / "artifacts/ape-real-room-next-captures-20261014/fresh_eight_capture_holdout_precommit.json"
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/fresh_holdout_20261014"
GLBS = ROOT / "assets/real_indoor_scenes/scansceneverse/glbs"
REPORT = ROOT / "artifacts/ape-real-room-next-captures-20261014/fresh_eight_capture_screen_report.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def save_rows(rows: list[dict], *, appearance: bool) -> None:
    REPORT.write_text(json.dumps({"schema": "ape.fresh-eight-capture-screen.v1",
                                  "precommit": relative(PRECOMMIT),
                                  "appearance_screened": appearance,
                                  "rows": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--appearance", action="store_true")
    args = parser.parse_args()
    frozen = json.loads(PRECOMMIT.read_text())
    for path, expected in frozen["frozen_code_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Frozen source code changed before holdout: {path}")
    rows = []
    for entry in frozen["capture_groups"]:
        scene_id = entry["id"]
        row = {"capture_group": f"scansceneverse/{scene_id}", "name": entry["name"],
               "purpose": entry["purpose"]}
        source_index = GLBS / f"{scene_id}.source.json"
        source = json.loads(source_index.read_text())
        mesh = GLBS / f"{scene_id}.glb"
        if digest(source_index) != frozen["source_index_sha256"][scene_id] or \
                digest(mesh) != source["sha256"]:
            raise ValueError(f"Source identity/hash mismatch: {scene_id}")
        scale = frozen["provisional_scale_to_metres_by_scene_id"][scene_id]
        room = BASE / scene_id
        room.mkdir(parents=True, exist_ok=True)
        geometry_path = room / "geometry_screen.json"
        source_path = room / "owner_capture_source.json"
        if not geometry_path.is_file():
            lower, upper, _ = mesh_bounds(mesh, scale)
            area = float((upper[0] - lower[0]) * (upper[2] - lower[2]))
            step = 0.8 if area > 50 else 0.4
            command = [sys.executable, str(HERE / "screen_new_capture_owner_routes.py"),
                       "--mesh", str(mesh), "--scale", str(scale),
                       "--grid-step", str(step), "--output", str(geometry_path)]
            try:
                process = subprocess.run(command, cwd=ROOT, capture_output=True,
                                         text=True, timeout=600)
            except subprocess.TimeoutExpired:
                row.update(geometry_status="screen_timeout")
                rows.append(row)
                save_rows(rows, appearance=args.appearance)
                print(scene_id, row["geometry_status"], flush=True)
                continue
            if process.returncode:
                row.update(geometry_status="screen_failed", error=process.stderr[-1200:])
                rows.append(row)
                save_rows(rows, appearance=args.appearance)
                print(scene_id, row["geometry_status"], flush=True)
                continue
        geometry = json.loads(geometry_path.read_text())
        row.update(geometry_status="screened",
                   geometry_candidate_count=geometry["geometry_candidate_count"],
                   geometry_screen=relative(geometry_path),
                   scale_to_metres=scale,
                   grid_step_m=geometry["grid_step_m"])
        record = {
            "schema": "ape.textured-real-capture-source.v1",
            "scene_id": f"scansceneverse/{scene_id}", "name": entry["name"],
            "source_type": "textured_mesh", "source_url": source["source_url"],
            "source_mesh": relative(mesh), "source_sha256": source["sha256"],
            "source_provenance": relative(source_index), "source_z_up": False,
            "scale_to_metres": scale,
            "floor_y_1pct_m": geometry["floor_y_1pct_m"],
            "geometry_screen": relative(geometry_path),
            "qualification": "Prospectively frozen holdout source; metric scale provisional and visual/task qualification pending."
        }
        source_path.write_text(json.dumps(record, indent=2) + "\n")
        row["source_capture"] = relative(source_path)
        if args.appearance:
            screen_path = room / "owner_placement_screen/screen.json"
            if not screen_path.is_file():
                command = [sys.executable, str(HERE / "screen_textured_capture_owner.py"),
                           "--source", str(source_path),
                           "--output", str(screen_path.parent),
                           "--precommit", relative(PRECOMMIT)]
                try:
                    process = subprocess.run(command, cwd=ROOT, capture_output=True,
                                             text=True, timeout=600)
                except subprocess.TimeoutExpired:
                    row.update(appearance_status="screen_timeout")
                    rows.append(row)
                    save_rows(rows, appearance=args.appearance)
                    print(scene_id, row["appearance_status"], flush=True)
                    continue
                if process.returncode:
                    row.update(appearance_status="screen_failed", error=process.stderr[-1200:])
                    rows.append(row)
                    save_rows(rows, appearance=args.appearance)
                    print(scene_id, row["appearance_status"], flush=True)
                    continue
            appearance = json.loads(screen_path.read_text())
            row.update(appearance_status=appearance["status"],
                       appearance_screen=relative(screen_path),
                       owner_candidate_count=appearance["candidate_count"],
                       owner_eligible_count=appearance["eligible_count"])
        rows.append(row)
        save_rows(rows, appearance=args.appearance)
        print(scene_id, row["geometry_candidate_count"],
              row.get("appearance_status", "geometry_only"), flush=True)


if __name__ == "__main__":
    main()
