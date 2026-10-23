"""V197: replay V196 crops with owner-anchored accepted-view feature bank."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v197"
V196 = ROOT / "artifacts/ape-closed-loop-cove50-v196/seed103-head-view-audit.json"
V196_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v196"
V195 = ROOT / "artifacts/ape-closed-loop-cove50-v195/scene103/seed103/trace.json"
V195_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v195/scene103/seed103"
OWNER = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"
SIMILARITY_MIN = .35
MEMORY_MAX_M = .12


def main() -> None:
    vision = Vision("generic")
    owner = np.load(OWNER)
    trace = json.loads(V195.read_text())
    source_paths = {sha(path): path for path in V195_ROOT.glob("frame-*.npz")}
    accepted = [event for event in trace["events"]
                if event.get("accepted_box") is not None and
                event.get("accepted_similarity") is not None and
                event["accepted_similarity"] >= SIMILARITY_MIN and
                event.get("actor_input_sha256") in source_paths]
    assert len(accepted) >= 2
    bank_events = accepted[-2:]
    bank = [owner]
    bank_sources = []
    for event in bank_events:
        source = source_paths[event["actor_input_sha256"]]
        with np.load(source) as saved:
            rgb = Image.fromarray(saved["rgb"])
        feature = vision.feature([rgb.crop(event["accepted_box"])])[0]
        bank.append(feature)
        bank_sources.append({"time_s": event["time_s"],
                             "frame_sha256": sha(source),
                             "owner_similarity_at_accept": event["accepted_similarity"],
                             "crop_box": event["accepted_box"],
                             "bank_feature_sha256": __import__("hashlib").sha256(
                                 feature.tobytes()).hexdigest()})
    prior = json.loads(V196.read_text())
    rows = []
    max_reproduction_error = 0.
    for old in prior["rows"]:
        yaw = old["yaw_rad"]
        label = old["label"]
        path = V196_ROOT / f"{label}-yaw{yaw:.1f}.npz"
        with np.load(path) as saved:
            rgb = Image.fromarray(saved["rgb"])
        proposals = old["proposals"]
        if proposals:
            features = vision.feature([rgb.crop(p["box"]) for p in proposals])
            scores = features @ np.asarray(bank).T
            max_reproduction_error = max(max_reproduction_error, max(
                abs(float(row[0]) - proposal["similarity"])
                for row, proposal in zip(scores, proposals)))
        else:
            scores = np.zeros((0, len(bank)))
        candidates = []
        scored = []
        for proposal, row in zip(proposals, scores):
            owner_score = float(row[0])
            bank_score = float(np.max(row))
            distance = proposal["distance_to_memory_m"]
            passed = bank_score >= SIMILARITY_MIN and distance is not None and distance <= MEMORY_MAX_M
            scored.append({"box": proposal["box"],
                           "similarities": [round(float(v), 4) for v in row],
                           "distance_to_memory_m": distance,
                           "evaluator_truth_iou": proposal["evaluator_truth_iou"],
                           "owner_pass": owner_score >= SIMILARITY_MIN and
                           distance is not None and distance <= MEMORY_MAX_M,
                           "bank_pass": passed})
            if passed:
                candidates.append((bank_score, len(scored) - 1))
        selected = scored[max(candidates)[1]] if candidates else None
        rows.append({"yaw_rad": yaw, "label": label,
                     "accepted_box": None if selected is None else selected["box"],
                     "accepted_correct_evaluator_only": bool(
                         selected is not None and selected["evaluator_truth_iou"] >= .25),
                     "accepted_wrong_evaluator_only": bool(
                         selected is not None and selected["evaluator_truth_iou"] < .25),
                     "proposals": scored,
                     "raw_sha256": sha(path)})
    assert max_reproduction_error < .005, max_reproduction_error
    present = [row for row in rows if row["label"] == "present"]
    absent = [row for row in rows if row["label"] == "absent"]
    correct = sum(row["accepted_correct_evaluator_only"] for row in present)
    wrong = sum(row["accepted_wrong_evaluator_only"] for row in present)
    absent_accepted = sum(row["accepted_box"] is not None for row in absent)
    result = {"scope": "V197 revealed seed103 offline V196 saved-box/crop replay. Bank=owner exemplar plus last2 V195 original-owner>=0.35 accepted crops, no evaluator selection; max prototype similarity>=0.35 and depth-memory<=12cm. Evaluator IoU only post-choice.",
              "source_sha256": sha(SOURCE),
              "v195_trace_sha256": sha(V195), "v196_sha256": sha(V196),
              "owner_sha256": sha(OWNER),
              "bank_sources": bank_sources,
              "max_original_score_reproduction_error": round(max_reproduction_error, 5),
              "correct_present_views": correct,
              "wrong_present_views": wrong,
              "accepted_absent_views": absent_accepted,
              "screen_pass": correct >= 2 and wrong == 0 and absent_accepted == 0,
              "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "view-bank-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
