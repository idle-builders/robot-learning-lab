"""Execute generic-first teach/find, with an expanded search only when needed.

The owner response is procedural: evaluator truth supplies only the yes/no
answer, never a candidate, crop, score, search direction, or motion target.
Each search stage is a fresh simulator render with its own aligned RGB, depth,
video and trace. A composite video records the exact question/answer order.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import run_real_capture_hybrid_teaching as hybrid
import active_scan
from candidate_review import CandidateReview


baseline = hybrid.union.baseline
ROOT = baseline.ROOT
original_run_search = baseline.run_search


class ConditionalVision(hybrid.HybridTeachingVision):
    def __init__(self, selector: str, device: str | None = None):
        super().__init__(selector, device=device)
        self.search_mode = "generic"

    def detect(self, image):
        if not self._teaching_frame_seen:
            self._teaching_frame_seen = True
            return active_scan.Vision.detect(self, image)
        if self.search_mode == "generic":
            return active_scan.Vision.detect(self, image)
        if self.search_mode == "expanded":
            return hybrid.union.SearchUnionVision.detect(self, image)
        raise ValueError(f"Unknown search mode {self.search_mode}")


def review_phase(folder: Path, result: dict, memory: CandidateReview,
                 answer_at_s: float) -> tuple[dict | None, dict | None]:
    if result["action"] != "candidate_to_owner_for_confirmation":
        return None, None
    verify = result["verification"]
    selected = verify["selected"]
    if selected is None:
        raise ValueError("Offer lacks a verified selected proposal")
    image = folder / verify["rgb_file"]
    captured = float(verify["capture_time_s"])
    offered = memory.offer(image_bytes=image.read_bytes(),
        image_ref=str(image.relative_to(ROOT)), box_xyxy=selected["box"],
        captured_at_s=captured, offered_at_s=captured + .05,
        image_healthy=True)
    if offered["action"] in ("skip_rejected_candidate", "skip_expired_candidate"):
        return offered, None
    if offered["action"] != "ask_owner_to_confirm_displayed_candidate":
        raise ValueError("Unexpected candidate review action")
    # The scripted person sees the highlighted image. Truth determines only
    # their reply; it never enters CandidateReview or the search policy.
    answer_yes = bool(result["target_present"] and
                      verify["evaluator_selected_target_iou"] >= .25)
    answer = memory.answer(candidate_id=offered["candidate_id"],
        image_sha256=offered["image_sha256"], confirmed=answer_yes,
        answered_at_s=max(captured + .05, answer_at_s))
    return offered, answer


def compose_video(out: Path, stages: list[dict]) -> tuple[str, list[dict]]:
    manifest = []
    path = out / "episode.mp4"
    with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        for stage in stages:
            folder = out / stage["name"]
            capture = cv2.VideoCapture(str(folder / "first_person.mp4"))
            index = 0
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                manifest.append({"kind": "sensor_frame", "stage": stage["name"],
                                 "stage_frame": index,
                                 "rgb": str((folder / f"rgb_{index:03d}.png").relative_to(ROOT)),
                                 "depth": str((folder / f"depth_{index:03d}.png").relative_to(ROOT))})
                index += 1
            capture.release()
            if index != stage["result"]["frames"]:
                raise ValueError(f"Stage video and trace disagree: {folder}")
            answer = stage["answer"]
            offer = stage["offer"]
            if offer is None or answer is None:
                continue
            rgb = cv2.imread(str(ROOT / offer["image_ref"]))
            if rgb is None:
                raise ValueError("Cannot show reviewed image")
            box = np.rint(offer["box_xyxy"]).astype(int)
            cv2.rectangle(rgb, tuple(box[:2]), tuple(box[2:]), (0, 235, 255), 2)
            caption = ("Owner: yes, that's it" if answer["action"].startswith(
                "owner_confirmed") else "Owner: no, look again")
            cv2.putText(rgb, caption, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            manifest.append({"kind": "owner_answer_overlay", "stage": stage["name"],
                             "source_rgb": offer["image_ref"],
                             "answer": answer["action"]})
    return str(path.relative_to(ROOT)), manifest


def conditional_run_search(*args, **kwargs) -> dict:
    if len(args) < 10:
        raise ValueError("Expected baseline search arguments")
    vision = args[1]
    out = Path(args[9]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start_time_s = float(kwargs["start_time_s"])
    memory = CandidateReview()
    stages = []

    def run_stage(name: str, mode: str, capture_start_s: float):
        vision.search_mode = mode
        values = list(args)
        values[9] = out / name
        options = {**kwargs, "start_time_s": capture_start_s}
        wall_start = time.perf_counter()
        result = original_run_search(*values, **options)
        elapsed = time.perf_counter() - wall_start
        return result, elapsed

    first, first_elapsed = run_stage("generic", "generic", start_time_s)
    first_offer, first_answer = review_phase(out / "generic", first, memory,
        start_time_s + first_elapsed + .25)
    stages.append({"name": "generic", "mode": "generic", "result": first,
                   "worker_wall_s": first_elapsed,
                   "offer": first_offer, "answer": first_answer})
    confirmed = bool(first_answer and first_answer["action"].startswith(
        "owner_confirmed"))
    fallback_reason = (None if confirmed else
        ("owner_rejected" if first_answer is not None else
         "no_generic_candidate"))
    final_phase = first
    if not confirmed:
        second_start_s = start_time_s + first_elapsed + .3
        second, second_elapsed = run_stage("expanded", "expanded",
                                           second_start_s)
        second_offer, second_answer = review_phase(out / "expanded", second,
            memory, second_start_s + second_elapsed + .25)
        stages.append({"name": "expanded", "mode": "expanded",
                       "result": second, "worker_wall_s": second_elapsed,
                       "offer": second_offer, "answer": second_answer})
        final_phase = second
        confirmed = bool(second_answer and second_answer["action"].startswith(
            "owner_confirmed"))
    video, manifest = compose_video(out, stages)
    status = ("owner_confirmed_visual_identity" if confirmed else
              "unidentified_ask_new_view_or_help")
    full = {"schema": "ape.real-capture-conditional-search.v1",
            "target_name": first["target_name"],
            "distractor_name": first["distractor_name"],
            "target_present": first["target_present"],
            "action": status,
            "candidate_action": final_phase["action"],
            "wrong_confident_candidate": any(
                x["result"]["wrong_confident_candidate"] for x in stages),
            "best_similarity": final_phase["best_similarity"],
            "best_proposal_evaluator_target_iou": final_phase[
                "best_proposal_evaluator_target_iou"],
            "target_visible_in_any_view": any(
                x["result"]["target_visible_in_any_view"] for x in stages),
            "owner_questions": sum(bool(x["answer"]) for x in stages),
            "fallback_reason": fallback_reason,
            "fallback_used": len(stages) == 2,
            "motion_target": None,
            "stage_reports": [{"name": x["name"], "mode": x["mode"],
                "trace": str((out / x["name"] / "trace.json").relative_to(ROOT)),
                "video": x["result"]["video"],
                "worker_wall_s": x["worker_wall_s"],
                "candidate_action": x["result"]["action"],
                "offer": x["offer"], "owner_answer": x["answer"]}
                for x in stages],
            "review_events": memory.events,
            "frames": len(manifest), "video_frames": manifest,
            "video": video,
            "limits": "Scripted truthful owner, stationary search pose and static room during inference; stage worker wall time is measured but moving-sensor service is evaluated separately."}
    (out / "episode_trace.json").write_text(json.dumps(full, indent=2) + "\n")
    return full


baseline.Vision = ConditionalVision
baseline.run_search = conditional_run_search


if __name__ == "__main__":
    baseline.main()
