"""Reserve varied RGB entries before fitting the next reach controller."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_MATRIX = BASE / "v105/native-varied-entry-exposure/matrix.json"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
OUT = BASE / "v116"
PLAN = OUT / "reserved-contact-validation-plan.json"
COUNT = 24
PRIOR_FULL_CONTACT_SEEDS = {19011, 19020}
ROUTES = ("center", "pitch-low", "pitch-high", "roll-low", "roll-high")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera_region(row: dict) -> str:
    x, y = row["camera_xy_mm"]
    return ("+" if x >= 0 else "-") + ("+" if y >= 0 else "-")


def pixel_region(row: dict) -> str:
    return "24-63" if row["orange_pixels"] < 64 else "64+"


def coverage(rows: list[dict]) -> dict:
    keys = {
        "route": (lambda row: row["route_variant"], ROUTES),
        "motor_v": (lambda row: str(row["rail_v"]), ("10", "11", "12")),
        "delay_ms": (lambda row: str(row["delay_ms"]), ("0", "5", "10")),
        "gaze_rad": (lambda row: str(row["gaze_head_yaw_rad"]), ("-1.3", "-1.5")),
        "toy_x_m": (lambda row: str(row["toy_offset_x_m"]),
                    ("0.0058", "0.00605", "0.0063", "0.00655")),
        "toy_y_m": (lambda row: str(row["toy_offset_y_m"]),
                    ("0.0057", "0.0059", "0.0061", "0.0063")),
        "camera_region": (camera_region, ("++", "+-", "-+", "--")),
        "pixel_region": (pixel_region, ("24-63", "64+")),
    }
    return {name: {key: Counter(fn(row) for row in rows)[key]
                   for key in levels}
            for name, (fn, levels) in keys.items()}


def choose(rows: list[dict]) -> tuple[list[dict], dict, int]:
    targets = {
        "route": {"center": 5, "pitch-low": 5, "pitch-high": 5,
                  "roll-low": 5, "roll-high": 4},
        "motor_v": {str(v): 8 for v in (10, 11, 12)},
        "delay_ms": {str(v): 8 for v in (0, 5, 10)},
        "gaze_rad": {"-1.3": 12, "-1.5": 12},
        "toy_x_m": {str(v): 6 for v in (.0058, .00605, .0063, .00655)},
        "toy_y_m": {str(v): 6 for v in (.0057, .0059, .0061, .0063)},
        "camera_region": {key: 6 for key in ("++", "+-", "-+", "--")},
        "pixel_region": {"24-63": 12, "64+": 12},
    }
    weights = {"route": 4, "motor_v": 3, "delay_ms": 3,
               "gaze_rad": 2, "toy_x_m": 2, "toy_y_m": 2,
               "camera_region": 2, "pixel_region": 2}
    rng = random.Random(116)
    best = None
    best_key = None
    for _ in range(40000):
        sample = rng.sample(rows, COUNT)
        counts = coverage(sample)
        penalty = sum(weights[name] * sum(
            abs(counts[name][key] - target)
            for key, target in levels.items())
            for name, levels in targets.items())
        key = (penalty, tuple(sorted(row["seed"] for row in sample)))
        if best_key is None or key < best_key:
            best, best_key = sample, key
            if penalty == 0:
                break
    assert best is not None and best_key is not None
    selected = sorted(best, key=lambda row: row["seed"])
    counts = coverage(selected)
    if (min(counts["route"].values()) < 4 or
            min(counts["motor_v"].values()) < 6 or
            min(counts["delay_ms"].values()) < 6 or
            min(counts["gaze_rad"].values()) < 8 or
            min(counts["camera_region"].values()) < 4 or
            min(counts["toy_x_m"].values()) < 4 or
            min(counts["toy_y_m"].values()) < 4 or
            min(counts["pixel_region"].values()) < 8):
        raise ValueError("Could not reserve adequately varied contact holdout")
    return selected, counts, best_key[0]


def main() -> None:
    source_plan = json.loads(V105_PLAN.read_text())
    source_matrix = json.loads(V105_MATRIX.read_text())
    if (not source_matrix["complete"] or source_matrix["case_count"] != 64 or
            source_matrix["entry_available_count"] != 64 or
            source_matrix["plan_sha256"] != sha(V105_PLAN)):
        raise ValueError("V105 source entry matrix must be complete and matched")
    used = {case["seed"] for case in json.loads(V113_PLAN.read_text())["cases"]}
    used |= PRIOR_FULL_CONTACT_SEEDS
    original = {case["id"]: case for case in source_plan["cases"]}
    eligible = [row for row in source_matrix["cases"] if row["seed"] not in used]
    if len(eligible) != 38:
        raise ValueError("Expected 38 still-unvisited full-native V105 starts")
    selected, counts, penalty = choose(eligible)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Holdout applies only to rejected historical sandbox")
    cases, refs = [], []
    for row in selected:
        report = ROOT / row["result"]
        entry = json.loads(report.read_text())
        if (entry["body_sha256"] != source_plan["source_sha256"]["body"] or
                entry["selected_pointer_sha256"] != source_plan["source_sha256"]["pointer"] or
                entry["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"V105 entry identity mismatch: {row['id']}")
        refs.append({"path": row["result"], "sha256": sha(report)})
        cases.append({**original[row["id"]],
                      "id": f"reserved-{row['seed']}",
                      "source_entry_id": row["id"],
                      "expected_selection_mode": row["selection_mode"],
                      "source_entry_camera_xy_mm": row["camera_xy_mm"],
                      "source_entry_orange_pixels": row["orange_pixels"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v116-reserved-contact-validation.v1",
        "role": "reserved_full_native_contact_validation_unopened",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source_plan["source_sha256"],
        "v105_plan_sha256": sha(V105_PLAN),
        "v105_entry_matrix_sha256": sha(V105_MATRIX),
        "v113_training_plan_sha256": sha(V113_PLAN),
        "excluded_full_native_seeds": sorted(used),
        "source_entry_reports": refs,
        "selection": ("Reserve 24 of the 38 V105 entries with no prior full-native "
                      "contact visit. Exclude every V113 training start and "
                      "the two V109 starts. A fixed-seed 40,000-sample search "
                      "balances route, motor, delay, gaze, toy X/Y, camera "
                      "quadrant and visible-pixel bins using RGB entry data "
                      "only. No contact result or scored future outcome enters "
                      "this selection."),
        "selection_balance_penalty": penalty,
        "coverage": counts,
        "case_count": COUNT,
        "cases": cases,
        "evaluation_gate": ("Keep these cases sealed until a candidate "
                            "observable reach controller is fitted from opened "
                            "training data. Replay the entire native sequence "
                            "with full RGB, contact >=1.2 s, support, rated "
                            "motor, action and collision gates; report all "
                            "24 outcomes and actual branch exposure. A passing "
                            "holdout is development evidence, not a new-body "
                            "qualification or substitute for a wholly new "
                            "disjoint score."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V116 holdout changed")
    PLAN.write_bytes(encoded)
    print(f"frozen {PLAN} sha256={sha(PLAN)} coverage={counts}")


if __name__ == "__main__":
    main()
