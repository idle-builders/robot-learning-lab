"""Run and archive a frozen whole-task room breadth screen."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import artifact_archive as archive

HERE = Path(__file__).resolve().parent
SIM_PY = ROOT / ".venv-learning/bin/python"
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
OUT = ROOT / "assets/real_indoor_scenes/learner_option_pilot_20261016"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_episode(folder: Path, s3) -> dict:
    name = folder.parent.name
    label = ("v4_breadth20" if folder.name == "v4_breadth20_standard"
             else folder.name)
    if label == "v4_breadth20" and name == "capture_cabin_east_1m":
        name = "cabin"  # Original first-room archive name.
    manifest = ART / f"learner_option_{label}_{name}_archive.json"
    receipt = ART / f"learner_option_{label}_{name}_prune_receipt.json"
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        assert saved["remote_verified_manifest_sha256"] == digest(manifest)
        return saved
    relative = folder.relative_to(ROOT).as_posix()
    if not manifest.is_file():
        archive.make_plan(ROOT, [relative], manifest)
    rows, encoded = archive.read_manifest(manifest)
    missing = [row for row in rows["files"] if not (ROOT / row["path"]).is_file()]
    if missing:
        # Recover a prior interruption after remote verification but during
        # local pruning. Only captured media may already be absent.
        assert all(Path(row["path"]).suffix in (".png", ".mp4") and
                   Path(row["path"]).name != "whole_episode_review.mp4"
                   for row in missing)
        manifest_sha = digest(manifest)
        manifest_key = (f"{archive.PREFIX}/manifests/sha256/"
                        f"{manifest_sha[:2]}/{manifest_sha[2:]}.json")
        assert archive.remote_hash(s3, "robot-learning-lab-archive",
                                   manifest_key, len(encoded)) == manifest_sha
        for row in missing:
            assert archive.remote_hash(s3, "robot-learning-lab-archive",
                archive.object_key(row["sha256"]), row["size"]) == row["sha256"]
        remote_sha = manifest_sha
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            remote = archive.upload(ROOT, manifest, s3,
                                    "robot-learning-lab-archive")
        remote_sha = remote["manifest_sha256"]
    assert remote_sha == digest(manifest)
    removed = []
    for row in rows["files"]:
        path = ROOT / row["path"]
        assert path.is_relative_to(folder)
        if path.is_file():
            assert digest(path) == row["sha256"]
        else:
            assert row in missing
        if path.suffix in (".png", ".mp4") and path.name != \
                "whole_episode_review.mp4":
            removed.append(row)
    saved = {"schema": "ape.verified-media-prune.v1",
             "remote_verified_manifest_sha256": remote_sha,
             "removed": removed,
             "retained_whole_video": relative + "/whole_episode_review.mp4"}
    for row in removed:
        (ROOT / row["path"]).unlink(missing_ok=True)
    receipt.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


def compact_packets(folder: Path) -> None:
    source = folder / "independent_range/packets.json"
    if not source.is_file():
        return
    target = source.with_name(source.name + ".gz")
    assert not target.exists()
    original = digest(source)
    with source.open("rb") as raw, gzip.open(target, "wb", compresslevel=9) as packed:
        shutil.copyfileobj(raw, packed)
    with gzip.open(target, "rb") as packed:
        assert hashlib.sha256(packed.read()).hexdigest() == original
    source.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=ART /
                        "learner_option_v4_breadth20_precommit.json")
    parser.add_argument("--run", default="v4_breadth20_standard")
    parser.add_argument("--max-new", type=int, default=0)
    parser.add_argument("--min-free-mb", type=int, default=100)
    args = parser.parse_args()
    precommit = args.precommit.resolve()
    report = OUT / f"{args.run}_sweep.json"
    freeze = json.loads(precommit.read_text())
    policy = ROOT / freeze["policy"]
    runner = ROOT / freeze["runner"]
    assert digest(policy) == freeze["policy_sha256"]
    assert digest(runner) == freeze["runner_sha256"]
    assert digest(ROOT / freeze["source_manifest"]) == freeze[
        "source_manifest_sha256"]
    s3 = archive.client(None, "https://s3.pub2.infomaniak.cloud",
                        "us-east-1")
    rows = []
    launched = 0
    for case in freeze["scenes"]:
        key = case["scene_key"]
        folder = OUT / key / args.run
        summary = folder / "summary.json"
        if not summary.is_file():
            if args.max_new and launched >= args.max_new:
                break
            if shutil.disk_usage(ROOT).free < args.min_free_mb * 1_000_000:
                print("paused_low_disk", key, flush=True)
                break
            command = [str(SIM_PY), str(runner), "--scene", key,
                       "--run", args.run, "--policy",
                       str(policy), "--fault", "none",
                       "--route-sensor", "separate_range",
                       "--monitor-coverage", "all360",
                       "--scenario-variant", "standard"]
            if freeze.get("generic_recovery"):
                command.append("--recover")
            trial = subprocess.run(command, cwd=ROOT, text=True,
                                   capture_output=True)
            launched += 1
            if trial.returncode or not summary.is_file():
                print("runner_failed", key, trial.stderr[-1500:], flush=True)
                break
        episode = json.loads(summary.read_text())
        reel = folder / "whole_episode_review.mp4"
        archive_name = folder.parent.name
        archive_label = ("v4_breadth20" if folder.name ==
                         "v4_breadth20_standard" else folder.name)
        if archive_label == "v4_breadth20" and archive_name == \
                "capture_cabin_east_1m":
            archive_name = "cabin"
        receipt = ART / (f"learner_option_{archive_label}_{archive_name}"
                         "_prune_receipt.json")
        if not reel.is_file() and not receipt.is_file():
            subprocess.run([str(SIM_PY), str(HERE / "make_episode_reel.py"),
                            str(folder)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
        saved = archive_episode(folder, s3)
        compact_packets(folder)
        rows.append({"scene_key": key, "result": episode["result"],
                     "owner_status": episode["owner_status"],
                     "wrong_candidate_questions": episode.get(
                         "wrong_candidate_questions", 0),
                     "options": [e["executed"] for e in episode["events"]],
                     "video": (str(reel.relative_to(ROOT))
                               if reel.is_file() else None),
                     "archived_video": str(reel.relative_to(ROOT)),
                     "summary": str(summary.relative_to(ROOT)),
                     "archive_manifest_sha256": saved[
                         "remote_verified_manifest_sha256"]})
        report.write_text(json.dumps({
            "schema": "ape.learner-option-breadth-sweep.v1",
            "precommit": str(precommit.relative_to(ROOT)),
            "precommit_sha256": digest(precommit),
            "policy_sha256": freeze["policy_sha256"],
            "generic_recovery": bool(freeze.get("generic_recovery")),
            "processed": len(rows), "total": len(freeze["scenes"]),
            "completed": sum(r["result"] == "completed" for r in rows),
            "safe_unresolved": sum(r["result"] != "completed" for r in rows),
            "rows": rows}, indent=2) + "\n")
        print(len(rows), key, episode["result"], flush=True)
    print("sweep", len(rows), "/", len(freeze["scenes"]),
          "new", launched, flush=True)


if __name__ == "__main__":
    main()
