"""Evaluator-only identity and media audit of V393's owner retry."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from articulated_person_scene import ITEM_SIDE_OFFSET_M
from challenge_raised_replica_show_v381 import REPLICA_XYZ_M
from run import ROOT, sha
from run_live_cued_retry_show_v393 import (OUT as INPUT_ROOT, RETRY_INDEX,
                                           RETRY_MOVE_S,
                                           RETRY_RESPONSE_DELAY_S, VARIANT,
                                           natural_hand)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v395"
REPLICA = np.asarray(REPLICA_XYZ_M, dtype=float)
PERSON_OFFSET = np.asarray(VARIANT["offset_xyz_m"], dtype=float)


def expected_held(mode: str, elapsed_s: float) -> np.ndarray:
    hand = natural_hand(elapsed_s)
    if mode == "compliant" and elapsed_s > .8:
        hand = natural_hand(.8)
    elif mode == "delayed":
        if elapsed_s > 1.1:
            hand = natural_hand(1.1)
        retry_start = RETRY_INDEX * .1 + RETRY_RESPONSE_DELAY_S
        if elapsed_s >= retry_start - 1e-9:
            fraction = min(1., max(0., (elapsed_s - retry_start) /
                                   RETRY_MOVE_S))
            hand = ((1. - fraction) * natural_hand(1.1) +
                    fraction * natural_hand(.8))
    return hand + ITEM_SIDE_OFFSET_M + PERSON_OFFSET


def decoded_frames(path: Path) -> int:
    video = cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = video.read()
            if not ok:
                break
            count += 1
    finally:
        video.release()
    return count


def main() -> None:
    aggregate_path = INPUT_ROOT / "cued-retry-show-audit.json"
    aggregate = json.loads(aggregate_path.read_text())
    rows = []
    for entry in aggregate["reports"]:
        mode = entry["mode"]
        report_path = ROOT / entry["report_path"]
        assert sha(report_path) == entry["report_sha256"]
        report = json.loads(report_path.read_text())
        by_seed = {row["seed"]: row for row in report["owner_show_rows"]}
        for episode in report["rows"]:
            seed = int(episode["case"][0])
            response = by_seed[seed]["response"]
            samples = response["actor_samples"]
            start_s = float(samples[0]["capture_time_s"])
            selected = []
            for sample in samples:
                packet_path = ROOT / sample["actor_packet_path"]
                assert sha(packet_path) == sample["actor_packet_sha256"]
                point = sample["actor_choice"]["accepted_world_xyz_m"]
                if point is None:
                    continue
                elapsed_s = round(float(sample["capture_time_s"]) -
                                  start_s, 6)
                position = np.asarray(point, dtype=float)
                held_error = float(np.linalg.norm(
                    position - expected_held(mode, elapsed_s)))
                replica_error = float(np.linalg.norm(position - REPLICA))
                selected.append({
                    "elapsed_s": elapsed_s,
                    "held_error_m": round(held_error, 5),
                    "replica_error_m": round(replica_error, 5),
                    "closer_to": ("held_item" if held_error < replica_error
                                  else "raised_replica"),
                })
            trace_path = ROOT / episode["trace_path"]
            video_path = ROOT / episode["video_path"]
            assert sha(trace_path) == episode["trace_sha256"]
            assert sha(video_path) == episode["video_sha256"]
            frames = decoded_frames(video_path)
            assert frames == 61, (video_path, frames)
            rows.append({
                "response_mode": mode, "seed": seed,
                "status": response["status"],
                "accepted_held_frames": sum(
                    x["closer_to"] == "held_item" for x in selected),
                "accepted_replica_frames": sum(
                    x["closer_to"] == "raised_replica" for x in selected),
                "final_accepted_class": (selected[-1]["closer_to"]
                                         if selected else None),
                "video_decoded_frames": frames,
                "selected": selected,
            })
            print(mode, seed, response["status"],
                  "held", rows[-1]["accepted_held_frames"],
                  "replica", rows[-1]["accepted_replica_frames"],
                  "video", frames, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "scope": "Independent evaluator-only association check of V393 accepted points against scripted held-item and raised-replica centers. Response labels and truth coordinates are unavailable to the actor. All recorded packet, trace, video and per-mode report hashes are checked; every episode video is decoded. This audit cannot validate physical sensor calibration or real owner response.",
        "source_sha256": sha(SOURCE),
        "input_aggregate_sha256": sha(aggregate_path),
        "rows": rows,
    }
    (OUT / "cued-retry-identity-media-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
