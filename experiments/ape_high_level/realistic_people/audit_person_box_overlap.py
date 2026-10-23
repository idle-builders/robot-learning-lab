"""Screen saved owner choices for mixed person proposals before flow seeding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def overlap_fraction(box: list[float], other: list[float]) -> float:
    area = lambda r: max(0., r[2] - r[0]) * max(0., r[3] - r[1])
    width = max(0., min(box[2], other[2]) - max(box[0], other[0]))
    height = max(0., min(box[3], other[3]) - max(box[1], other[1]))
    return width * height / max(1., min(area(box), area(other)))


def replay(run: Path) -> dict:
    path = run / "evidence/online-action-trace.json"
    actions = json.loads(path.read_text())["actions"]
    rows = []
    for action in actions:
        decision = action["actor"]
        selected = decision.get("chosen_proposal")
        proposals = decision.get("proposals", [])
        if selected is None or not 0 <= selected < len(proposals):
            continue
        box = proposals[selected]["box_xyxy"]
        fraction = max((overlap_fraction(box, row["box_xyxy"])
                        for index, row in enumerate(proposals) if index != selected),
                       default=0.)
        rows.append({"frame": action["frame"],
                     "selected_box_xyxy": box,
                     "other_person_proposals": len(proposals) - 1,
                     "maximum_smaller_box_overlap_fraction": fraction,
                     "would_unbind_after_initial_owner_click":
                         action["frame"] > 0 and fraction > .25,
                     "issued_command": action["command_for_next_frame"],
                     "intent": decision["intent"]})
    return {"run": str(run), "action_trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "chosen_frames": len(rows),
            "overlap_flags": [row["frame"] for row in rows
                              if row["would_unbind_after_initial_owner_click"]],
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"scope": "Saved-proposal developmental screen only; no live re-identification proof",
              "criterion": "chosen person box overlaps another proposal by >25% of smaller box after first click",
              "results": [replay(run) for run in args.runs]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["results"]:
        print(Path(row["run"]).name, row["overlap_flags"])


if __name__ == "__main__":
    main()
