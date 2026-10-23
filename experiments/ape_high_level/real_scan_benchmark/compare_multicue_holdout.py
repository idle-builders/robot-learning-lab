"""Summarize paired owner-stop item policies without counting scenes as new owners."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
STEM = "common_policy_multicue_holdout_"


def report(name: str) -> dict:
    return json.loads((ARTIFACTS / f"{STEM}{name}_report.json").read_text())


def completed(source: dict) -> dict[tuple[str, str], dict]:
    return {(row["capture_group"], row["target_name"]): row
            for row in source["rows"] if row.get("find")}


def main() -> None:
    baseline = report("dino")
    nominal = report("multi_cue")
    shifted = report("multi_cue_cool_dim")
    paired = completed(baseline)
    changed = completed(nominal)
    stressed = completed(shifted)
    if paired.keys() != changed.keys():
        raise ValueError("Nominal policy comparison is not scene/object paired")
    if stressed.keys() != {key for key in changed if key[1] == "BBQSauce"}:
        raise ValueError("Lighting stress is not paired on every bottle room")
    if not all(all(condition["media_aligned"] for condition in row["find"].values())
               for source in (paired, changed, stressed) for row in source.values()):
        raise ValueError("A saved video/RGB/depth/trace set is not aligned")
    gained = [key for key in paired if not paired[key]["present_correct"] and
              changed[key]["present_correct"]]
    lost = [key for key in paired if paired[key]["present_correct"] and
            not changed[key]["present_correct"]]
    shared_wrong = [key for key in paired if paired[key]["wrong_confident"] and
                    changed[key]["wrong_confident"]]
    stress_lost = [key for key in stressed if changed[key]["present_correct"] and
                   not stressed[key]["present_correct"]]
    stress_gained = [key for key in stressed if not changed[key]["present_correct"] and
                     stressed[key]["present_correct"]]
    result = {
        "schema": "ape.common-policy-multicue-paired-comparison.v1",
        "precommit": nominal["precommit"],
        "policy_reports": {name: f"artifacts/ape-real-room-next-captures-20261014/{STEM}{name}_report.json"
                           for name in ("dino", "multi_cue", "multi_cue_cool_dim")},
        "physical_capture_groups": len({key[0] for key in paired}),
        "paired_completed_object_room_count": len(paired),
        "all_completed_media_aligned": True,
        "baseline": {"present_correct": baseline["present_correct_count"],
                     "absent_safe": baseline["absent_safe_count"],
                     "wrong_confident": baseline["wrong_confident_count"]},
        "multi_cue": {"present_correct": nominal["present_correct_count"],
                      "absent_safe": nominal["absent_safe_count"],
                      "wrong_confident": nominal["wrong_confident_count"]},
        "paired_present_gains": [{"capture_group": group, "target": target,
            "room": changed[(group, target)]["name"]} for group, target in gained],
        "paired_present_losses": [{"capture_group": group, "target": target,
            "room": changed[(group, target)]["name"]} for group, target in lost],
        "shared_wrong_confident": [{"capture_group": group, "target": target,
            "room": changed[(group, target)]["name"]} for group, target in shared_wrong],
        "per_object": {target: {
            "supported_complete": sum(key[1] == target for key in paired),
            "baseline_present_correct": sum(key[1] == target and
                row["present_correct"] for key, row in paired.items()),
            "multi_cue_present_correct": sum(key[1] == target and
                row["present_correct"] for key, row in changed.items()),
            "baseline_absent_safe": sum(key[1] == target and
                row["absent_safe"] for key, row in paired.items()),
            "multi_cue_absent_safe": sum(key[1] == target and
                row["absent_safe"] for key, row in changed.items()),
        } for target in ("BBQSauce", "Peaches")},
        "bottle_search_photometry_stress": {
            "completed": len(stressed),
            "present_correct": shifted["present_correct_count"],
            "absent_safe": shifted["absent_safe_count"],
            "wrong_confident": shifted["wrong_confident_count"],
            "present_losses_vs_nominal": [changed[key]["name"] for key in stress_lost],
            "present_gains_vs_nominal": [changed[key]["name"] for key in stress_gained],
        },
        "limits": "New object pairs, but the same 14 development-exposed physical room groups and retained owner stops. RGB/depth is ideal rendered geometry; lighting shift is synthetic. This is not a 20-group full owner-loop or physical sensor result.",
    }
    output = ARTIFACTS / "common_policy_multicue_holdout_comparison.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print("paired", len(paired), "present", baseline["present_correct_count"],
          "->", nominal["present_correct_count"], "gains", len(gained),
          "losses", len(lost), "absent", baseline["absent_safe_count"],
          "->", nominal["absent_safe_count"], "stress",
          shifted["present_correct_count"], "/", len(stressed))


if __name__ == "__main__":
    main()
