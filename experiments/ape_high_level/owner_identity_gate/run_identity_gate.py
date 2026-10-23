"""Insert an actor-side identity gate before V597's generic-person promotion.

The derivative source is saved with the run. The insertion point is asserted
against the exact imported V597 function so a source change fails closed.
It runs the complete furnished route and turn, not a trace-only replay.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/owner_bystander_substitution"))
import run_substitution as substitution

OUT = ROOT / "artifacts/ape-owner-identity-gate/v1"
TURN = substitution.base.turn
MARKER = '        rows.append({"stage": "owner_person_rgbd_association",'
GATE = '''        rows.append({"stage": "anonymous_person_rgbd_consistent",
                     "time_s": round(time_s, 4),
                     "person_surface_xy_est_m": candidate.tolist(),
                     "distance_from_prior_track_m": displacement,
                     "two_view_point_distance_m": float(np.linalg.norm(points[0] - points[1])),
                     "identity_state": "old_owner_unconfirmed",
                     "actor_action": "request_explicit_reintroduction_before_show"})
        frames.append(owner.owner_frame(
            rgb, "I lost track of who is who. Please show me again.",
            boxes[-1] if boxes else None, None))
        return ("anonymous_person_waiting_for_reintroduction",
                rows, frames, None, None, time_s)
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retag_report(path, gate_source):
    report = json.loads(path.read_text())
    report["task"] = "T-jgx6e7"
    report["source_hashes"][str(Path(__file__).relative_to(ROOT))] = digest(__file__)
    report["derived_actor_source_path"] = str(gate_source)
    report["derived_actor_source_sha256"] = digest(gate_source)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_turn = TURN.owner_turn
    original_source = inspect.getsource(original_turn)
    if original_source.count(MARKER) != 1:
        raise RuntimeError("V597 owner-association insertion point changed")
    derived_source = original_source.replace(MARKER, GATE + MARKER)
    saved = OUT / "guarded_owner_turn_v597.py"
    saved.write_text(derived_source)
    try:
        exec(compile(derived_source, str(saved), "exec"), TURN.__dict__)
        substitution.OUT = OUT / "owner_vs_bystander"
        substitution.main()
        pair_path = substitution.OUT / "report.json"
        retag_report(pair_path, saved)

        base = substitution.base
        base.OUT = OUT / "moving_guard"
        base.CASES = (("moving_guard", "moving", .55, 1.6),)
        base.main()
        moving_path = base.OUT / "report.json"
        retag_report(moving_path, saved)
    finally:
        TURN.owner_turn = original_turn
    combined = {
        "task": "T-jgx6e7",
        "source_sha256": digest(__file__),
        "original_turn_source_sha256": digest(TURN.__file__),
        "guarded_turn_source_sha256": digest(saved),
        "owner_bystander_report": {"path": str(pair_path),
                                    "sha256": digest(pair_path)},
        "moving_guard_report": {"path": str(moving_path),
                                "sha256": digest(moving_path)},
        "rule": "After generic person RGB-D views agree, return an anonymous person observation and request reintroduction before restoring old-owner identity, restaging or asking for a show. An earlier owner-motion alarm still cancels the stale image.",
    }
    (OUT / "report.json").write_text(json.dumps(combined, indent=2) + "\n")


if __name__ == "__main__":
    main()
