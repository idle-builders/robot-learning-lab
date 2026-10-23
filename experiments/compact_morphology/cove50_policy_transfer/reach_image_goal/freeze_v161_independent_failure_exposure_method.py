"""Freeze a development-only reach exposure method without running simulation.

V161 is a possible next experiment on the rejected historical body. Its camera
screen, contact pairs and selector audit are separate later actions. This file
pins the complete family and grid before any V161 result can be collected.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v161/independent-failure-exposure-method-freeze.json"
POINTER = ROOT / "robot/ape-current-design.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
V149_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
V149_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
V152_MATRIX = BASE / "v152/native-paired-routes/matrix.json"
V152_SUMMARY = BASE / "v152/partial-source-checked-paired-route-summary.json"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V156_SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
V157_SUMMARY = BASE / "v157/source-checked-rgb-repeatability-summary.json"
V158_PLAN = BASE / "v158/new-seed-contact-plan.json"
V158_MATRIX = BASE / "v158/native-development-contact/matrix.json"
V158_SUMMARY = BASE / "v158/source-checked-development-contact-summary.json"
V160_METHOD = BASE / "v160/new-seed-selector-audit-method-freeze.json"
V160_AUDIT = BASE / "v160/new-seed-selector-grouped-audit.json"
V153_FITTER = Path(__file__).with_name("audit_v153_route_selector.py")
V158_RUNNER = Path(__file__).with_name("run_v158_new_seed_development_contact.py")
V156_RUNNER = Path(__file__).with_name("run_v156_independent_seed_entry_families.py")
HELD = {24606, 24612, 24615, 24618, 24621, 24624}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    sources = {
        "method_source": Path(__file__), "mechanical_pointer": POINTER,
        "untouched_score_reserve": RESERVE,
        "v149_plan": V149_PLAN, "v149_matrix": V149_MATRIX,
        "v152_plan": V152_PLAN, "v152_matrix": V152_MATRIX,
        "v152_summary": V152_SUMMARY,
        "v156_plan": V156_PLAN, "v156_matrix": V156_MATRIX,
        "v156_summary": V156_SUMMARY, "v157_summary": V157_SUMMARY,
        "v158_plan": V158_PLAN, "v158_matrix": V158_MATRIX,
        "v158_summary": V158_SUMMARY,
        "v160_method": V160_METHOD, "v160_audit": V160_AUDIT,
        "v153_fitter": V153_FITTER, "v156_runner": V156_RUNNER,
        "v158_runner": V158_RUNNER,
    }
    pointer, reserve, old_entry, old_plan = (
        read(POINTER), read(RESERVE), read(V149_MATRIX), read(V149_PLAN))
    old_contact = read(V152_MATRIX)
    screen, repeat, paired, audit = (
        read(V156_SUMMARY), read(V157_SUMMARY), read(V158_MATRIX),
        read(V160_AUDIT))
    if (pointer["status"] != "rejected_no_selected_mechanical_design" or
            pointer["selected_design_id"] is not None or
            reserve["status"] != "sealed_no_contact_visits" or
            screen["source_checked_entry_count"] != 168 or
            not screen["complete"] or
            screen["source_sha256"]["matrix"] != sha(V156_MATRIX) or
            repeat["source_checked_repeat_count"] != 24 or
            not repeat["complete"] or
            paired["case_count"] != 18 or
            paired["valid_matched_pair_count"] != 18 or
            paired["held_contact_visit_count"] != 0 or
            read(V158_SUMMARY)["valid_matched_pair_count"] != 18 or
            audit["development_gate_met"] or
            audit["held_out_selected_count"] != 0 or
            audit["held_contact_visit_count"] != 0 or
            audit["independent_score_visit_count"] != 0):
        raise ValueError("V161 requires complete failed development audit and sealed holds")
    entries = {row["seed"]: row for row in old_entry["cases"]}
    cases = {case["seed"]: case for case in old_plan["cases"]}
    contacts = {row["seed"]: row for row in old_contact["cases"]}
    anchors = (("roll_live", 24591, 6, "v16_live"),
               ("roll_locked", 24583, 6, "v16_locked"),
               ("pitch_locked", 24536, 12, "v16_locked"))
    for stratum, seed, _, route in anchors:
        row = entries[seed]
        contact = contacts[seed]
        if (row["selection_mode"] != route or
                row["observed_features"] is None or
                contact["baseline_pass"] or
                not contact["prescribed_pass"] or
                not contact["prescribed_margin_1p2_s_met"] or
                cases[seed]["cohort"] != row["cohort"]):
            raise ValueError(f"Opened anchor is not a measured paired repair: {seed}")
    pitch = [row for row in read(V156_MATRIX)["cases"]
             if row["stratum"] == "pitch_locked"]
    if (len(pitch) != 30 or
            any(row["selection_mode"] != "v21_scoped_y_locked" for row in pitch) or
            min(row["observed_features"]["entry_error_x_mm"] for row in pitch) <= 1.5):
        raise ValueError("V156 pitch route-exposure diagnosis changed")
    known = {case["seed"] for path in (RESERVE, V149_PLAN, V152_PLAN,
                                        V156_PLAN, V158_PLAN)
             for case in read(path)["cases"]}
    if known & set(range(24701, 24725)) or HELD & set(range(24701, 24725)):
        raise ValueError("V161 development seeds overlap a prior source or held set")
    grids = {
        "roll_live": {"toy_delta_x_m": [-.002, 0, .002],
                      "toy_delta_y_m": [-.005, -.0025, 0]},
        "roll_locked": {"toy_delta_x_m": [-.003, -.001, 0],
                        "toy_delta_y_m": [-.001, 0, .001]},
        "pitch_locked": {"toy_delta_x_m": [-.006, -.003, 0],
                         "toy_delta_y_m": [-.001, 0, .001]},
    }
    rng = random.Random(161)
    families = []
    next_seed = 24701
    for stratum, anchor_seed, count, target in anchors:
        anchor = cases[anchor_seed]
        observed = entries[anchor_seed]["observed_features"]
        for index in range(count):
            center_x = round(anchor["toy_offset_x_m"] +
                             rng.uniform(-.0002, .0002), 6)
            center_y = round(anchor["toy_offset_y_m"] +
                             rng.uniform(-.0002, .0002), 6)
            family = {
                "seed": next_seed, "stratum": stratum,
                "anchor_seed_audit_only": anchor_seed,
                "target_observed_route": target,
                "toy_center_x_m": center_x, "toy_center_y_m": center_y,
                "toy_delta_grid_m": grids[stratum],
                "rail_v": (10, 11, 12)[index % 3],
                "delay_ms": (0, 5, 10)[(index // 2) % 3],
                "gaze_head_yaw_rad": (-1.3, -1.5)[index % 2],
                "pitch_offset_rad": (.095, .1)[index % 2]
                if stratum == "pitch_locked" else .105,
                "roll_clearance_rad": -.25 if stratum == "pitch_locked"
                else (-.24, -.245)[index % 2],
                "anchor_observed_error_xy_mm": [
                    observed["entry_error_x_mm"],
                    observed["entry_error_y_mm"]],
            }
            if any(center_x + dx <= 0 or center_y + dy <= 0
                   for dx in grids[stratum]["toy_delta_x_m"]
                   for dy in grids[stratum]["toy_delta_y_m"]):
                raise ValueError(f"V161 toy grid leaves positive offset: {next_seed}")
            families.append(family)
            next_seed += 1
    value = {
        "schema": "robot-learning-lab.cove50-v161-independent-failure-exposure-method.v1",
        "role": "frozen_development_only_method_not_camera_contact_or_selector_result",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {name: sha(path) for name, path in sources.items()},
        "prior_visibility": "Frozen after full V156/V157/V158 and failed V160; their outcomes informed this new development method. All V158 held contacts and the V147 independent score remain unopened.",
        "diagnosis": {
            "v156_pitch_entries": 30,
            "pitch_target_route_exposures": 0,
            "pitch_observed_error_x_min_mm": min(
                row["observed_features"]["entry_error_x_mm"] for row in pitch),
            "scoped_x_boundary_mm": 1.5,
            "v158_roll_prescribed_rescues": 2,
            "v158_pitch_automatic_failures": 0,
            "v160_selected_new_seed_alternatives": 0,
        },
        "seed_family_count": len(families),
        "camera_entry_count_if_fully_run": 9 * len(families),
        "families": families,
        "camera_method": (
            "Run only first used-for-control CUDA RGB entry with the exact V156 "
            "body, evaluator and policy sources, on the nine predeclared toy "
            "positions of each independent seed family. Source-check each "
            "report, trace and RGB frame. Do not use simulator truth as a "
            "runtime feature. An absent target route is a measured exposure "
            "failure, never a reason to force a route or substitute a seed."),
        "one_contact_start_per_seed_family": True,
        "contact_selection": (
            "After all 216 camera entries are source-checked, select at most "
            "one controlled-RGB entry per seed with the predeclared target "
            "automatic route. Choose minimum squared distance of observed "
            "entry error X/Y in millimetres to that stratum's opened anchor "
            "features, breaking ties by entry ID. Families without a target "
            "route receive no contact. Freeze the resulting exact case list "
            "before any V161 contact; screen selection uses no V161 contact."),
        "paired_native_rule": (
            "On each selected development start compare automatic full-native "
            "contact against one prescribed route: specialist when automatic "
            "is scoped, scoped otherwise. Keep seed, physical start and RGB "
            "goal source identical. Require source-matched reports/traces, "
            "screen-to-automatic and automatic-to-prescribed numeric RGB "
            "goal parity within 0.1 mm, and pre-reach qpos max absolute gap "
            "at most 1e-4 in the mixed saved m/rad vector. Any RGB or physical "
            "start mismatch invalidates the original pair; a posthoc repeat "
            "cannot replace it. Apply the full saved contact, support, effort, "
            "collision, action and 1.2 s margin gates."),
        "failure_exposure_gate": (
            "Before any selector audit, require at least three independent "
            "source-valid automatic full-gate failures with passing, >=1.2 s "
            "prescribed contact in roll strata combined and three in pitch. "
            "Report target-route coverage and all automatic failures even if "
            "the gate fails. Do not count nine camera positions within one "
            "seed as independent or turn prescribed rescues into selections."),
        "selector_method": (
            "If and only if the failure-exposure gate passes, refit the "
            "source-pinned V153 one-axis observable route selector with "
            "source-valid historical V152 and V158 development pairs plus "
            "other V161 families; leave each V161 seed out whole. At least "
            "two independent held-out selected full-gate and >=1.2 s margin "
            "rescues spanning roll and pitch, zero selected full-gate "
            "regressions, margin losses and unresolved alternatives are "
            "required to consider a separate future holdout. If the fitter "
            "abstains or the exposure gate fails, stop without opening held "
            "V158 contacts or the V147 score."),
        "collection_stop": (
            "Bound this method to 24 new seeds, 216 camera entries and at "
            "most 24 matched full-native contact pairs. Stop early for source "
            "or parity faults that cannot be repaired without changing a "
            "frozen original. This method authorizes no paid launch by itself; "
            "the current rejected body and selected-body availability must "
            "be reassessed before a compute decision."),
        "held_contact_seed_ids": sorted(HELD),
        "held_contact_visits": 0,
        "independent_score_visits": 0,
        "contains_v161_camera_results": False,
        "contains_v161_contact_results": False,
        "contains_selector_result": False,
    }
    encoded = json.dumps(value, indent=2) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Frozen V161 development method changed")
    OUT.write_text(encoded)
    print(f"V161 method frozen: {len(families)} families, "
          f"{9 * len(families)} future camera entries; sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
