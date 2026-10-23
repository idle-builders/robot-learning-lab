"""Review the existing source roster outside the 20-room item-policy cohort.

This is a source inventory, not a policy holdout score or photoreal certificate.
It preserves every distinct capture in the denominator and shows route start,
middle and end frames before choosing any new policy episode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "assets/real_indoor_scenes/real_capture_40_current_candidate_set.json"
USED = ROOT / "artifacts/ape-real-room-next-captures-20261014/alt_pair_compound_20_precommit.json"
OUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/disjoint_21_source_review"


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def route_frames(scene: dict) -> list[Path]:
    path = ROOT / scene["moving_video"]
    family = scene["family"]
    folder = path.parent
    if family == "replica":
        frames = sorted(folder.glob("frame_*.jpg"))
    elif family == "scansceneverse":
        frames = sorted(folder.glob("frame_*.png"))
        if not frames:
            frames = sorted(folder.glob("rgb_*.png"))
    elif family == "hm3d_example":
        frames = [folder / name for name in ("first.jpg", "middle.jpg", "last.jpg")]
    elif family == "habitat_gs":
        frames = sorted((folder / "sensor_replay").glob("rgb_*.jpg"))
        if not frames:
            frames = sorted((folder / "sensor_replay").glob("rgb_*.png"))
    else:
        raise ValueError(f"Unhandled source family: {family}")
    return [frame for frame in frames if frame.is_file()]


def main() -> None:
    roster = json.loads(ROSTER.read_text())
    used = {case["capture_group"] for case in json.loads(USED.read_text())["cases"]}
    remaining = [scene for scene in roster["scenes"]
                 if scene["capture_group"] not in used]
    if len(used) != 20 or len(remaining) != 21 or len({
            scene["capture_group"] for scene in remaining}) != 21:
        raise ValueError("The 20/21 distinct-capture boundary changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, scene in enumerate(remaining):
        frames = route_frames(scene)
        selected = ([frames[0], frames[len(frames) // 2], frames[-1]]
                    if frames else [])
        rows.append({
            "index": index + 1,
            "scene_id": scene["scene_id"],
            "capture_group": scene["capture_group"],
            "family": scene["family"],
            "category": scene["category"],
            "source_url": scene["source_url"],
            "moving_video": scene["moving_video"],
            "route_frame_count": len(frames),
            "sample_frames": [{"path": relative(p), "sha256": sha(p)} for p in selected],
            "prior_other_task_summary": scene.get("full_task_summary"),
            "status": ("source_frames_available_visual_review_pending"
                       if len(selected) == 3 else "missing_route_frames"),
        })
        if scene["family"] == "habitat_gs":
            screen = ROOT / "assets/real_indoor_scenes/habitat_gs/owner_placement_screen/v3" \
                / scene["scene_id"].split("/")[-1] / "screen.json"
            if screen.is_file():
                rows[-1]["existing_owner_screen"] = relative(screen)
                rows[-1]["existing_owner_screen_sha256"] = sha(screen)
                rows[-1]["existing_owner_status"] = json.loads(screen.read_text())["status"]
    for page in range(3):
        sheet = Image.new("RGB", (3 * 320, 7 * 202), "#202020")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(rows[page * 7:(page + 1) * 7]):
            y = slot * 202
            label = f"{row['index']:02d} {row['scene_id']} ({row['family']})"
            draw.text((5, y + 2), label, fill="white")
            for col, item in enumerate(row["sample_frames"]):
                picture = Image.open(ROOT / item["path"]).convert("RGB")
                picture = ImageOps.contain(picture, (318, 178))
                x = col * 320 + (320 - picture.width) // 2
                sheet.paste(picture, (x, y + 23 + (178 - picture.height) // 2))
        sheet.save(OUT / f"route_contact_{page + 1}.jpg", quality=88)
    report = {
        "schema": "ape.disjoint-21-source-review.v1",
        "roster": relative(ROSTER), "roster_sha256": sha(ROSTER),
        "policy_cohort": relative(USED), "policy_cohort_sha256": sha(USED),
        "existing_policy_capture_groups": len(used),
        "other_distinct_capture_groups": len(remaining),
        "existing_gs_owner_supported": sum(r.get("existing_owner_status") ==
            "provisional_supported_visible" for r in rows),
        "existing_gs_owner_rejected": sum(r.get("existing_owner_status") ==
            "no_supported_visible_owner_placement" for r in rows),
        "maximum_possible_from_this_roster_before_other_source_screens": sum(
            r["family"] != "habitat_gs" or r.get("existing_owner_status") ==
            "provisional_supported_visible" for r in rows),
        "meaning": "Source-only review before new policy outcomes. Several scenes were previously exposed in other experiments; capture-disjoint here refers only to the named 20-room alternate-pair policy cohort.",
        "rows": rows,
        "contact_sheets": [relative(OUT / f"route_contact_{i}.jpg") for i in (1, 2, 3)],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_groups": len(rows),
                      "with_three_route_frames": sum(len(r["sample_frames"]) == 3 for r in rows),
                      "prior_gs_owner_supported": report["existing_gs_owner_supported"],
                      "prior_gs_owner_rejected": report["existing_gs_owner_rejected"],
                      "maximum_before_other_screens": report[
                          "maximum_possible_from_this_roster_before_other_source_screens"],
                      "families": sorted({r["family"] for r in rows})}))


if __name__ == "__main__":
    main()
