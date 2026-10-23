"""V382: apply V380's frozen 'held above floor' cue to raised replicas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_direct_structural_entry_v271 import V149, V150
from audit_held_item_candidate_v380 import (MIN_HELD_CENTER_Z_M, choose)
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v381/independent-hand-show.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v382"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(INPUT.read_text())
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(Path(V150).read_text())[
        "owner_color"]["hue_histogram"])
    rows = []
    for original in source["rows"]:
        samples = [choose(vision, exemplar, owner_hist, sample)
                   for sample in original["samples"]]
        row = {
            "case": original["case"],
            "source_video_path": original["video_path"],
            "source_video_sha256": original["video_sha256"],
            "baseline_held_correct_frames": original[
                "evaluator_correct_frames"],
            "baseline_wrong_accepted_frames": original[
                "evaluator_wrong_accepted_frames"],
            "height_gate_held_correct_frames": sum(
                sample["evaluator_selected_held_can"]
                for sample in samples),
            "height_gate_wrong_accepted_frames": sum(
                sample["actor_selected_box"] is not None and not
                sample["evaluator_selected_held_can"]
                for sample in samples),
            "samples": samples,
        }
        rows.append(row)
        print(row["case"], "baseline", row[
            "baseline_held_correct_frames"], row[
                "baseline_wrong_accepted_frames"], "height", row[
                "height_gate_held_correct_frames"], row[
                "height_gate_wrong_accepted_frames"], flush=True)
    report = {
        "scope": "Apply V380's frozen >=0.12m center-height condition and otherwise unchanged all-proposals DINO/hue/robust-depth selection to V381's new supported-raised identical replica scene. This reveals whether floor/held height is a general relation cue. Same source selected frame, scripted owner and synthetic optics limits as V381; evaluator truth scores after choice.",
        "source_sha256": sha(SOURCE),
        "v380_source_sha256": sha(Path(choose.__code__.co_filename)),
        "input_report_sha256": sha(INPUT),
        "held_height_min_m": MIN_HELD_CENTER_Z_M,
        "rows": rows,
    }
    (OUT / "raised-replica-height-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
