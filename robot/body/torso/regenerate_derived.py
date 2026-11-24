"""Regenerate or clean disposable scratch STL files for the APE0 Center-Body Torso FDM Carrier Study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def clean_scratch():
    removed = []
    scratch_dir = common.ARTIFACT_DIR / "scratch_stl"
    if scratch_dir.exists():
        for p in sorted(scratch_dir.rglob("*.stl")):
            removed.append(str(p.relative_to(common.ROOT)))
            p.unlink()
        shutil.rmtree(scratch_dir, ignore_errors=True)
    cand_dir = common.ARTIFACT_DIR / "candidates"
    if cand_dir.exists():
        for p in sorted(cand_dir.rglob("*.stl")):
            removed.append(str(p.relative_to(common.ROOT)))
            p.unlink()
    exe_bin = common.ARTIFACT_DIR / "zbuffer_renderer_bin"
    if exe_bin.exists():
        removed.append(str(exe_bin.relative_to(common.ROOT)))
        exe_bin.unlink()

    reg_record = {
        "schema": "cove50.scratch_regeneration.v1",
        "study": "robot/body/torso",
        "regenerate_command": ".venv/bin/python robot/body/torso/regenerate_derived.py --regenerate",
        "clean_command": ".venv/bin/python robot/body/torso/regenerate_derived.py --clean",
        "cleaned_disposable_files": removed,
    }
    (common.STUDY_DIR / "scratch-regeneration.json").write_text(json.dumps(reg_record, indent=2) + "\n")
    print(f"Cleaned {len(removed)} disposable scratch files and recorded scratch-regeneration.json")


def regenerate():
    py = str(common.ROOT / ".venv/bin/python")
    subprocess.run([py, str(common.STUDY_DIR / "build_torso_carriers.py")], check=True)
    subprocess.run([py, str(common.STUDY_DIR / "package_and_render.py")], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()
    if args.regenerate:
        regenerate()
    if args.clean:
        clean_scratch()


if __name__ == "__main__":
    main()
