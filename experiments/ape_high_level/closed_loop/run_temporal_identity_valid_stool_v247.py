"""V247: bounded visual-track continuation in the route-valid moving room.

The owner-color rule remains the only way to confirm a new track or add a
prototype. A recently confirmed track can survive a small color shift for
at most two inference calls if appearance and position agree strongly.
This is a revealed development probe, not a promoted identity rule.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

import run_active_view_valid_stool_v246 as valid
import run_rgb_floor_target_v213 as floor_target
from audit_entry_color_v150 import descriptor
from audit_side_look_target_v81 import SIMILARITY_MIN, TARGET_DISTANCE_MAX_M
from run import ROOT, Vision, locate_world, sha
from run_color_bank_live_v201 import HUE_MIN


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v247"
STRONG_SIMILARITY_MIN = .90
CONTINUITY_HUE_MIN = .67
CONTINUITY_LOCATION_MAX_M = .03
MAX_CALLS_SINCE_CONFIRMED = 2


class BoundedTrackBank(floor_target.RgbFloorTargetBank):
    def __init__(self, owner: np.ndarray, owner_hist: np.ndarray):
        super().__init__(owner, owner_hist)
        self.last_owner_confirmed_call = -1000
        self.continuity_acceptances = 0

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
        candidates = []
        for index, (proposal, feature, row_scores) in enumerate(
                zip(proposals, features, scores)):
            # This branch intentionally preserves V213's RGB floor ray for
            # target location. Depth is reserved for obstacle mapping.
            world = locate_world(proposal["box"], camera_pose, None)
            distance = (None if world is None else float(np.linalg.norm(
                np.asarray(world[:2]) - expected_xy)))
            winner = int(np.argmax(row_scores))
            score = float(row_scores[winner])
            color = descriptor(rgb, proposal["box"])
            hue = float(np.minimum(self.owner_hist,
                                   np.asarray(color["hue_histogram"])).sum())
            standard = (score >= SIMILARITY_MIN and distance is not None and
                        distance <= TARGET_DISTANCE_MAX_M and hue >= HUE_MIN)
            continuity = (not standard and
                          self.calls - self.last_owner_confirmed_call <=
                          MAX_CALLS_SINCE_CONFIRMED and
                          score >= STRONG_SIMILARITY_MIN and
                          distance is not None and
                          distance <= CONTINUITY_LOCATION_MAX_M and
                          hue >= CONTINUITY_HUE_MIN)
            rows.append({**proposal,
                         "similarity": round(score, 4),
                         "owner_similarity": round(float(row_scores[0]), 4),
                         "owner_hue_intersection": round(hue, 4),
                         "winning_prototype_index": winner,
                         "depth_world_xyz_m": world,
                         "distance_to_memory_m": (None if distance is None else
                                                  round(distance, 4)),
                         "actor_gate_accept": standard or continuity,
                         "identity_basis": ("owner_confirmed" if standard else
                                            "bounded_continuity" if continuity else
                                            "rejected"),
                         "evaluator_target_iou": 0.})
            if standard or continuity:
                candidates.append(index)
        selected_index = (max(candidates,
                              key=lambda index: rows[index]["similarity"])
                          if candidates else None)
        selected = rows[selected_index] if selected_index is not None else None
        annotated = rgb.copy()
        draw = ImageDraw.Draw(annotated)
        for row in rows:
            draw.rectangle(row["box"],
                           outline=(0, 255, 0) if row is selected else (255, 200, 0),
                           width=3)
        annotated.save(annotated_path)
        if selected is not None and selected["identity_basis"] == "owner_confirmed":
            self.last_owner_confirmed_call = self.calls
            if selected["owner_similarity"] >= SIMILARITY_MIN:
                self.recent.append(np.asarray(features[selected_index], dtype=float))
                self.recent = self.recent[-2:]
        if selected is not None and selected["identity_basis"] == "bounded_continuity":
            self.continuity_acceptances += 1
        self.events.append({"call_index": self.calls,
                            "source": ("matched_absent" if
                                       "matched-absent" in annotated_path.name else
                                       "live_frame"),
                            "accepted": selected is not None,
                            "owner_similarity": None if selected is None else
                                selected["owner_similarity"],
                            "bank_similarity": None if selected is None else
                                selected["similarity"],
                            "owner_hue_intersection": None if selected is None else
                                selected["owner_hue_intersection"],
                            "winning_prototype_index": None if selected is None else
                                selected["winning_prototype_index"],
                            "identity_basis": None if selected is None else
                                selected["identity_basis"],
                            "last_owner_confirmed_call":
                                self.last_owner_confirmed_call,
                            "bank_recent_count": len(self.recent)})
        rows.sort(key=lambda row: row["similarity"], reverse=True)
        return {"proposals": rows,
                "accepted_box": None if selected is None else selected["box"],
                "accepted_similarity": None if selected is None else
                    selected["similarity"],
                "accepted_world_xyz_m": None if selected is None else
                    selected["depth_world_xyz_m"]}


def main() -> None:
    prior = (valid.OUT, valid.SOURCE, floor_target.RgbFloorTargetBank,
             sys.argv[:])
    valid.OUT = OUT
    valid.SOURCE = SOURCE
    floor_target.RgbFloorTargetBank = BoundedTrackBank
    sys.argv = [str(SOURCE), "--seeds", "111", "113"]
    try:
        valid.main()
    finally:
        (valid.OUT, valid.SOURCE, floor_target.RgbFloorTargetBank,
         sys.argv) = prior
    raw = OUT / "active-view-valid-stool-live-audit.json"
    report = json.loads(raw.read_text())
    report["scope"] = (
        "V247 revealed V246 route-valid moving-stool seeds111/113, only identity "
        "rule changed. Owner hue>=.70 remains required to start/reconfirm a "
        "track and update prototypes. For at most two inference calls after "
        "last owner confirmation, a proposed existing track may continue "
        "with bank similarity>=.90, RGB floor-ray memory error<=3cm and "
        "owner hue>=.67. No evaluator truth enters actor decisions. This is "
        "an exploratory development rule pending same-color target-absent "
        "lookalike/lighting tests; kinematic body, ideal depth/virtual range "
        "and instant stop remain.")
    report["source_sha256"] = sha(SOURCE)
    report["v246_sha256"] = sha(ROOT /
        "artifacts/ape-closed-loop-cove50-v246/active-view-valid-stool-live-audit.json")
    (OUT / "temporal-identity-valid-stool-live-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    raw.unlink()
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
