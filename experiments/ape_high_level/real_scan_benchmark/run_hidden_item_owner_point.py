"""Owner-point rescue for the frozen cool/dim hidden-change cohort.

The simulator-scripted owner supplies one pixel click on the first generic
view containing the moved item. The actor gets only that click, RGB-D and its
existing visual proposals; it localizes the clicked proposal and refreshes its
appearance exemplar before resuming bounded motion.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_real_capture_teach_find import SCENES
import run_hidden_item_rearrangement as trial


SOURCE_REPORT = ART / "hidden_item_cool_dim_report.json"
SOURCE_AUDIT = ART / "hidden_item_cool_dim_audit.json"
SOURCE_ASSETS = ROOT / "assets/real_indoor_scenes/hidden_item_cool_dim_20261015"
OUT = ROOT / "assets/real_indoor_scenes/hidden_item_owner_point_20261015"
PRECOMMIT = ART / "hidden_item_owner_point_precommit.json"
REPORT = ART / "hidden_item_owner_point_report.json"


def scripted_owner_click(first_visible: dict) -> tuple[float, float]:
    # Evaluator truth is used only to simulate what a person points at.
    box = first_visible["evaluator_target_box_xyxy"]
    return (float((box[0] + box[2]) / 2),
            float((box[1] + box[3]) / 2))


def actor_select_from_click(click: tuple[float, float], proposals: list[dict]
                            ) -> dict | None:
    x, y = click
    candidates = [p for p in proposals if p.get("actor_map_xy_m") is not None
        and p["box"][0] <= x <= p["box"][2]
        and p["box"][1] <= y <= p["box"][3]
        and not p.get("border_crop")]
    return min(candidates, key=lambda p: (p["box"][2] - p["box"][0]) *
        (p["box"][3] - p["box"][1]), default=None)


def final_search(scene, case, state: dict, start: np.ndarray,
                 target: tuple, distractor: tuple, taught_image: np.ndarray,
                 taught_box: list[float], source_folder: Path, out: Path) -> dict:
    baseline = trial.base.colour_bank.conditional.baseline
    old_teach = json.loads((source_folder / "teach/trace.json").read_text())
    vision = baseline.Vision("generic-proposal")
    vision._teaching_frame_seen = True
    exemplar = vision.feature([Image.fromarray(taught_image).crop(
        tuple(round(v) for v in taught_box))])[0]
    colour = baseline.colour_histogram(taught_image, taught_box)
    return baseline.run_search(scene, vision, state["actual"],
        state["actual_yaw"], state["actor"] - start, state["actor_yaw"],
        exemplar, float(old_teach["acceptance_similarity"]), True, out,
        start_time_s=300., search_xz=target, search_distractor_xz=distractor,
        owner_start_xz=tuple(start), target_name=case["target"],
        distractor_name=case["distractor"], identity_mode="multi_cue",
        colour_exemplar=colour,
        colour_acceptance=float(old_teach["colour_acceptance_similarity"]),
        search_photometry="cool_dim")


def state_after_initial(trace: dict) -> dict:
    walk = trace["initial_walk"]
    last = walk["actions"][-1]
    yaw_delta = last["amount"] if last["decision"] == "turn" else 0.
    return {"actual": np.asarray(walk["actual_final_xz_m"], dtype=float),
        "actor": np.asarray(walk["actor_final_xz_m"], dtype=float),
        "actual_yaw": float(last["evaluator_actual_yaw_deg"] + yaw_delta),
        "actor_yaw": float(last["actor_yaw_deg"] + yaw_delta)}


def compose(old_trace: dict, old_episode: dict, out: Path,
            owner_rgb: Path, click: tuple, walk: dict | None,
            final: dict | None) -> str:
    path = out / "owner_point_full_loop.mp4"
    clips = [("Initial guarded approach", old_trace["initial_walk"]["video"]),
             ("Cool/dim recheck: item unidentified", old_episode["video"])]
    if walk:
        clips.append(("Owner-pointed item approach", walk["video"]))
    if final:
        clips.append(("Fresh owner confirmation", final["video"]))
    with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for title, ref in clips:
            for rgb in imageio.get_reader(str(ROOT / ref)):
                cv2.putText(rgb, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 2)
                writer.append_data(rgb)
            if title.startswith("Cool/dim"):
                image = np.asarray(Image.open(owner_rgb).convert("RGB")).copy()
                cv2.circle(image, (round(click[0]), round(click[1])), 12,
                           (255, 230, 0), 3)
                cv2.putText(image, "Owner points: this one", (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
                writer.append_data(image)
                writer.append_data(image)
    return relative(path)


def run_case(case: dict) -> dict:
    key = case["scene_key"]
    source = ROOT / "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015" / key
    old_folder = SOURCE_ASSETS / key
    old_trace = json.loads((old_folder / "trace.json").read_text())
    old_episode = json.loads((old_folder / "moved/recheck/episode_trace.json").read_text())
    if old_episode["action"] == "owner_confirmed_visual_identity":
        return {"scene_key": key, "mode": "autonomous",
            "complete": old_trace["branches"][0]["final_action"] ==
                "owner_confirmed_visual_identity",
            "source_trace": relative(old_folder / "trace.json")}
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    generic = json.loads((old_folder / "moved/recheck/generic/trace.json").read_text())
    visible = next((o for o in generic["observations"] if
                    o["evaluator_target_box_xyxy"]), None)
    if visible is None:
        return {"scene_key": key, "mode": "owner_requested_new_view",
                "complete": False}
    click = scripted_owner_click(visible)
    selected = actor_select_from_click(click, visible["proposals"])
    if selected is None:
        return {"scene_key": key, "mode": "owner_click_unlocalized",
                "complete": False}
    start = np.asarray(SCENES[key]["start_xz"], dtype=float)
    observed_target = start + np.asarray(selected["actor_map_xy_m"], dtype=float)
    state = state_after_initial(old_trace)
    screen_data = json.loads((ROOT / case["screen"]).read_text())
    screen = next(x for x in screen_data["scenes"] if x.get("scene_key") == key)
    target = tuple(screen["find_distractor_xz"])
    distractor = tuple(screen["find_target_xz"])
    scene = trial.base.make_scene(case, {"find_target_xz": target,
        "find_distractor_xz": distractor})
    walk = trial.walk(scene, state, observed_target, out / "updated_walk",
                      np.asarray(target), photometry="cool_dim")
    frame = visible["frame"]
    owner_rgb = old_folder / "moved/recheck/generic" / f"rgb_{frame:03d}.png"
    taught_image = np.asarray(Image.open(owner_rgb).convert("RGB"))
    final = None
    if walk["result"] == "observed_target_standoff":
        final = final_search(scene, case, state, start, target, distractor,
            taught_image, selected["box"], source, out / "final_check")
    video = compose(old_trace, old_episode, out, owner_rgb, click, walk, final)
    trace = {"schema": "ape.hidden-item-owner-point.v1", "scene_key": key,
        "source_recheck": relative(old_folder / "moved/recheck/episode_trace.json"),
        "owner_input": {"kind": "point_in_existing_rgb_frame",
            "frame": frame, "image": relative(owner_rgb), "click_xy": click},
        "actor_selected_proposal_box": selected["box"],
        "actor_selected_local_depth": selected["local_depth"],
        "actor_observed_target_xz_m": observed_target.tolist(),
        "evaluator_selected_target_iou": selected["evaluator_target_iou"],
        "updated_walk": walk, "final_action": final["action"] if final else None,
        "evaluator_final_target_gap_m": float(np.linalg.norm(state["actual"] -
            np.asarray(target))), "video": video,
        "limits": "Scripted perfect owner click from evaluator identity; owner cooperates after a failed autonomous search. Selected proposal, depth and refreshed appearance are actor-observed. Ideal range/yaw and kinematic proxy persist."}
    (out / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    return {"scene_key": key, "mode": "owner_point_rescue",
        "complete": trace["final_action"] == "owner_confirmed_visual_identity",
        "walk_result": walk["result"], "final_action": trace["final_action"],
        "trace": relative(out / "trace.json"), "video": video}


def main() -> None:
    cases = json.loads(trial.CASES.read_text())["cases"]
    frozen = {"schema": "ape.hidden-item-owner-point-precommit.v1",
        "source_report": relative(SOURCE_REPORT),
        "source_report_sha256": sha(SOURCE_REPORT),
        "source_audit": relative(SOURCE_AUDIT),
        "source_audit_sha256": sha(SOURCE_AUDIT),
        "cases": [c["scene_key"] for c in cases],
        "fallback": "Only after autonomous cool/dim search returns unknown: scripted owner points to the taught item in the first generic view where visible. Actor selects the smallest existing non-border, depth-localized proposal containing the click; refreshes appearance from that proposal crop; approaches its observed RGB-D location and asks for fresh confirmation. No new view or proposal => ask for help with no motion. Autonomous successes stay unchanged.",
        "score": ["total_complete", "rescue_complete", "false_target",
                  "new_motion_safety", "owner_interventions"],
        "limits": "Owner click is perfect scripted human input from evaluator object identity; not autonomous perception or a physical demonstration."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Frozen owner-point rescue changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        row = run_case(case)
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.hidden-item-owner-point-report.v1",
            "precommit": relative(PRECOMMIT), "rows": rows}, indent=2) + "\n")
        print(row, flush=True)


if __name__ == "__main__":
    main()
