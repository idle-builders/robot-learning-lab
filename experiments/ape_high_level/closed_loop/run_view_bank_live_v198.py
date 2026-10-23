"""V198: online owner-anchored view bank in revealed seed-103 V195 loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import run_third_split_live_v195 as baseline
from audit_side_look_target_v81 import SIMILARITY_MIN, TARGET_DISTANCE_MAX_M
from run import ROOT, Vision, iou, locate_world, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v198"
V192 = ROOT / "artifacts/ape-closed-loop-cove50-v192/third-split-geometry-audit.json"
V193 = ROOT / "artifacts/ape-closed-loop-cove50-v193/third-split-origin-audit.json"
V194 = ROOT / "artifacts/ape-closed-loop-cove50-v194/third-split-route-audit.json"
V197 = ROOT / "artifacts/ape-closed-loop-cove50-v197/view-bank-audit.json"
EXEMPLAR = ROOT / "artifacts/ape-closed-loop-cove50-v149/owner-taught-exemplar.npy"


class OwnerAnchoredViewBank:
    def __init__(self, owner: np.ndarray):
        self.owner = np.asarray(owner, dtype=float)
        self.recent: list[np.ndarray] = []
        self.calls = 0
        self.events: list[dict] = []

    def infer(self, vision: Vision, exemplar: np.ndarray,
              rgb: Image.Image, depth: np.ndarray,
              camera_pose: tuple[np.ndarray, np.ndarray],
              expected_xy: np.ndarray, evaluator_box: list[int] | None,
              annotated_path: Path) -> dict:
        assert np.allclose(exemplar, self.owner)
        self.calls += 1
        proposals = vision.detect(rgb)
        features = (vision.feature([rgb.crop(row["box"]) for row in proposals])
                    if proposals else np.zeros((0, len(self.owner))))
        bank = np.stack([self.owner, *self.recent])
        scores = features @ bank.T
        rows = []
        candidate_indices = []
        for index, (proposal, feature, row_scores) in enumerate(
                zip(proposals, features, scores)):
            world = locate_world(proposal["box"], camera_pose, depth)
            distance = (None if world is None else float(np.linalg.norm(
                np.asarray(world[:2]) - expected_xy)))
            winner = int(np.argmax(row_scores))
            score = float(row_scores[winner])
            accepted = (score >= SIMILARITY_MIN and distance is not None and
                        distance <= TARGET_DISTANCE_MAX_M)
            rows.append({**proposal,
                         "similarity": round(score, 4),
                         "owner_similarity": round(float(row_scores[0]), 4),
                         "winning_prototype_index": winner,
                         "depth_world_xyz_m": world,
                         "distance_to_memory_m": (None if distance is None else
                                                  round(distance, 4)),
                         "actor_gate_accept": accepted,
                         "evaluator_target_iou": round(
                             iou(proposal["box"], evaluator_box), 4)})
            if accepted:
                candidate_indices.append(index)
        selected_index = (max(candidate_indices,
                              key=lambda index: rows[index]["similarity"])
                          if candidate_indices else None)
        selected = rows[selected_index] if selected_index is not None else None
        annotated = rgb.copy()
        draw = ImageDraw.Draw(annotated)
        for row in rows:
            draw.rectangle(row["box"],
                           outline=(0, 255, 0) if row is selected else (255, 200, 0),
                           width=3)
        annotated.save(annotated_path)
        if selected_index is not None and rows[selected_index]["owner_similarity"] >= SIMILARITY_MIN:
            feature = np.asarray(features[selected_index], dtype=float)
            self.recent.append(feature)
            self.recent = self.recent[-2:]
            self.events.append({"call_index": self.calls,
                                "source": "original_owner_accepted_crop",
                                "crop_box": selected["box"],
                                "owner_similarity": selected["owner_similarity"],
                                "feature_sha256": hashlib.sha256(
                                    feature.tobytes()).hexdigest()})
        self.events.append({"call_index": self.calls,
                            "source": "matched_absent" if "matched-absent" in annotated_path.name else "live_frame",
                            "accepted": selected is not None,
                            "owner_similarity": None if selected is None else selected["owner_similarity"],
                            "bank_similarity": None if selected is None else selected["similarity"],
                            "winning_prototype_index": None if selected is None else selected["winning_prototype_index"],
                            "bank_recent_count": len(self.recent)})
        rows.sort(key=lambda row: row["similarity"], reverse=True)
        return {"proposals": rows,
                "accepted_box": None if selected is None else selected["box"],
                "accepted_similarity": None if selected is None else selected["similarity"],
                "accepted_world_xyz_m": None if selected is None else selected["depth_world_xyz_m"]}


def main() -> None:
    assert json.loads(V197.read_text())["screen_pass"]
    scene = next(row for row in json.loads(V192.read_text())["rows"]
                 if row["seed"] == 103)
    assert scene["label"] == "generous_kinematic"
    selected = next(row for row in json.loads(V194.read_text())["rows"]
                    if row["seed"] == 103)
    entry = next(row for row in json.loads(V193.read_text())["rows"]
                 if row["seed"] == 103 and not row["absent"] and
                 row["light_multiplier"] == 1.)
    assert selected["status"] == "goal_selected" and entry["correct_acceptances"] >= 2
    positions = scene["audits"]["0.065"]["positions_xy_m"]
    variant = {"variant": "seed103-generous-online-view-bank",
               "cream_cheese_xy_m": positions["CreamCheese"],
               "parmesan_xy_m": positions["Parmesan"]}
    vision = Vision("generic")
    exemplar = np.load(EXEMPLAR)
    bank = OwnerAnchoredViewBank(exemplar)
    old_out = baseline.OUT
    old_infer = baseline.controller.infer
    old_absent_infer = baseline.memory_infer
    OUT.mkdir(parents=True, exist_ok=True)
    baseline.OUT = OUT
    baseline.controller.infer = bank.infer
    baseline.memory_infer = bank.infer
    try:
        row = baseline.score_case(103, variant, selected, entry, vision, exemplar)
    finally:
        baseline.OUT = old_out
        baseline.controller.infer = old_infer
        baseline.memory_infer = old_absent_infer
    bank_winner_accepts = sum(event["source"] == "live_frame" and
                              event["accepted"] and
                              event["winning_prototype_index"] not in (None, 0)
                              for event in bank.events)
    bank_only_accepts = sum(event["source"] == "live_frame" and
                            event["accepted"] and
                            event["owner_similarity"] < SIMILARITY_MIN
                            for event in bank.events)
    result = {"scope": "V198 revealed seed103 development replay of frozen V195 controller/scene/entry/route, only image identity replaced by owner exemplar plus up to2 online original-owner>=0.35 accepted crops. Max similarity>=0.35 and same12cm depth-memory gate. Bank-only accepts never become prototypes. Matched absent uses final online bank. Kinematic/ideal range/instant cancel unchanged.",
              "source_sha256": sha(SOURCE),
              "v195_source_sha256": sha(baseline.SOURCE),
              "v192_sha256": sha(V192), "v193_sha256": sha(V193),
              "v194_sha256": sha(V194), "v197_sha256": sha(V197),
              "bank_events": bank.events,
              "bank_winner_live_acceptances": bank_winner_accepts,
              "bank_only_live_acceptances": bank_only_accepts,
              "development_gate_pass": row["origin_to_target_gate_pass"] and bank_only_accepts >= 1,
              "row": row}
    (OUT / "online-view-bank-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps({"status": row["status"],
                      "moves": row["moves"],
                      "actual_target_gap_m": row["evaluator_actual_target_gap_m"],
                      "min_clearance_m": row["evaluator_min_model_clearance_m"],
                      "matched_absent_accepted": row["matched_absent_accepted"],
                      "bank_only_accepts": bank_only_accepts,
                      "development_gate_pass": result["development_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
