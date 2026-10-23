"""Apply one unchanged owner-help transition to alternate-pair identity misses.

The source task first runs from a fresh owner start. Only its saved unknown
present-item episodes branch here; this is a state-continuous policy probe,
not an uninterrupted new owner episode or a real human response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "alt_pair_compound_20_report.json"
SOURCE_PRECOMMIT = ART / "alt_pair_compound_20_precommit.json"
PRECOMMIT = ART / "alt_pair_owner_help_precommit.json"
REPORT = ART / "alt_pair_owner_help_report.json"
OUT = ROOT / "assets/real_indoor_scenes/alt_pair_owner_help_20261015"
ARMS = (("accurate_present", "find_present", "accurate"),
        ("wrong_present", "find_present", "wrong"),
        ("silent_present", "find_present", "silent"),
        ("wrong_absent", "find_absent", "wrong"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def frozen_definition() -> dict:
    rows = json.loads(SOURCE.read_text())["rows"]
    original = {c["scene_key"]: c for c in json.loads(
        SOURCE_PRECOMMIT.read_text())["cases"]}
    if len(rows) != len(original) or len(original) != 20:
        raise ValueError("Complete 20-room source required before owner help")
    cases = []
    for row in rows:
        if row["status"] != "teach_find_completed_far_viewpoint" or \
                row["present"]["action"] != "unidentified_ask_new_view_or_help":
            continue
        key = row["scene_key"]
        case = original[key]
        cases.append({"scene_key": key, "capture_group": case["capture_group"],
            "source_folder": relative(ROOT /
                "assets/real_indoor_scenes/alt_pair_compound_20_20261015" / key),
            "screen": case["screen"], "target": case["target"],
            "distractor": case["distractor"]})
    return {"schema": "ape.alt-pair-owner-help-precommit.v1",
        "source": relative(SOURCE), "source_sha256": sha(SOURCE),
        "source_precommit": relative(SOURCE_PRECOMMIT),
        "source_precommit_sha256": sha(SOURCE_PRECOMMIT),
        "policy_sources": [{"path": relative(p), "sha256": sha(p)} for p in
            (Path(__file__), HERE / "run_compound_owner_point_recovery.py")],
        "cases": cases, "arms": [a[0] for a in ARMS],
        "policy": "After an unknown identity, ask for one owner point in a saved RGB-D view; select the smallest existing metric proposal containing that pixel, refresh the exemplar and search again under the same cool/dim condition. The owner may point accurately, point to a distractor, or stay silent. Keep identity and motion target unknown when evidence remains insufficient; ask the scripted owner to review any newly offered candidate.",
        "score": ["fresh correct present confirmation", "wrong candidate offers",
            "wrong final confirmations", "silent abstentions", "saved RGB-D/video"],
        "limits": "Continuation selected from development-exposed source misses; scripted owner click and yes/no, same static room, ideal geometry and unselected body. Do not count as a fresh-room generalization rate or a physical human interaction."}


def persist(rows: list[dict], frozen: dict) -> None:
    REPORT.write_text(json.dumps({"schema": "ape.alt-pair-owner-help-report.v1",
        "precommit": relative(PRECOMMIT), "processed": len(rows),
        "expected": len(frozen["cases"]) * len(ARMS),
        "rows": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    frozen = frozen_definition()
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Owner-help source or policy changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.freeze_only:
        print(relative(PRECOMMIT), len(frozen["cases"]))
        return
    from run_compound_owner_point_recovery import run_one
    rows = []
    launched = 0
    for case in frozen["cases"]:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm[0] / "trace.json"
            if path.is_file():
                trace = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"], "arm": arm[0],
                       "action": trace["action"], "trace": relative(path)}
            else:
                if args.max_new and launched >= args.max_new:
                    persist(rows, frozen)
                    return
                result = run_one(case, arm, output_root=OUT,
                                 vision_device="cuda")
                row = {**result, "capture_group": case["capture_group"]}
                launched += 1
            rows.append(row)
            persist(rows, frozen)
            print(row["scene_key"], row["arm"], row["action"], flush=True)


if __name__ == "__main__":
    main()
