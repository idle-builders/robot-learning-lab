"""V260: pair V259's final uncertain can with owner yes/no alternatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import run_terminal_owner_clarification_v249 as clarification
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v260"
V259_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v259"
V259 = V259_ROOT / "expiring-map-valid-stool-live-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision("generic")
    owner = np.load(clarification.V149)
    owner_hist = np.asarray(json.loads(clarification.V150.read_text())[
        "owner_color"]["hue_histogram"])
    branch = V259_ROOT / "scene107/seed107"
    events = json.loads((branch / "online-color-bank-events.json").read_text())
    crops = [event for event in events
             if event["source"] == "original_owner_accepted_crop"][-2:]
    if len(crops) != 2:
        raise AssertionError("Missing final owner-anchored view bank")
    features = []
    sources = []
    for event in crops:
        frame = branch / f"frame-{event['call_index']}.npz"
        with np.load(frame) as saved:
            rgb = Image.fromarray(saved["rgb"])
        feature = np.asarray(vision.feature([rgb.crop(event["crop_box"])])[0],
                             dtype=float)
        digest = hashlib.sha256(feature.tobytes()).hexdigest()
        if digest != event["feature_sha256"]:
            raise AssertionError("Reconstructed prototype differs from run")
        features.append(feature)
        sources.append({"frame_sha256": sha(frame),
                        "crop_box": event["crop_box"],
                        "feature_sha256": digest})
    prior = (clarification.OUT, clarification.V247_ROOT)
    clarification.OUT = OUT
    clarification.V247_ROOT = V259_ROOT
    try:
        rows = [clarification.episode(107, case, vision, owner,
                                      owner_hist, features)
                for case in ("present", "ordinary_absent",
                             "red_lookalike_absent")]
    finally:
        clarification.OUT, clarification.V247_ROOT = prior
    report = {"scope": "V260 revealed V259 seed107 terminal owner question from the expiring-map route-valid moving-stool stop. Reconstruct exact final two owner-confirmed RGB prototypes from saved hashed frames. V249 guarded lower-head candidate question, scripted owner yes/no and fresh post-answer recheck; paired true can, ordinary target-absent and red-material distinct-mesh target-absent lookalike. Visual-only outcome, no post-answer base move, physical sensor/native skill or real owner validation.",
              "source_sha256": sha(SOURCE), "v259_sha256": sha(V259),
              "prototype_sources": sources, "rows": rows}
    (OUT / "seed107-owner-clarification-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], row["status"], row["owner_questions"])


if __name__ == "__main__":
    main()
