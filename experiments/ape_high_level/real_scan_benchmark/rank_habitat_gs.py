"""Rank public real GS scenes for *visual* indoor screening using annotations.

ObjectNav categories only prioritize downloads. They are not proof of an
indoor room, reconstruction quality, or the 50 cm ape's traversability.
"""

import collections
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes/habitat_gs"
REPO = "RukawaY/gs_scenes"
HERE = Path(__file__).resolve().parent
INDOOR = {"bed", "toilet", "sofa", "refrigerator", "sink", "tv",
          "cabinet", "bookshelf", "lamp"}
OUTDOOR = {"tree", "car", "street lamp", "traffic sign", "fire hydrant",
           "bicycle", "barrier"}


def one(index: int) -> dict:
    split = "train" if index <= 55 else "val"
    scene_id = f"scene{index:02d}"
    path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"episodes/objectnav/{split}/content/{scene_id}.json.gz",
        local_dir=ASSETS))
    episodes = json.load(gzip.open(path, "rt"))["episodes"]
    counts = collections.Counter(row.get("object_category") for row in episodes)
    indoor_count = sum(counts[label] for label in INDOOR)
    outdoor_count = sum(counts[label] for label in OUTDOOR)
    return {"scene_id": scene_id, "source_split": split,
            "episode_count": len(episodes), "categories": dict(counts),
            "indoor_hint_count": indoor_count,
            "outdoor_hint_count": outdoor_count,
            "indoor_hint_fraction": round(indoor_count / max(1, len(episodes)), 3),
            "not_a_scene_label": True}


def main() -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(one, range(1, 66)))
    rows.sort(key=lambda row: (-row["indoor_hint_fraction"],
                               row["outdoor_hint_count"], row["scene_id"]))
    result = {"schema": "ape.habitat-gs-indoor-screen-priority.v1",
              "source": f"https://huggingface.co/datasets/{REPO}",
              "source_commit": HfApi().repo_info(REPO, repo_type="dataset").sha,
              "warning": "Generated ObjectNav category counts prioritize visual review only."
                         " They are not scene labels and can be wrong.",
              "scenes": rows}
    output = HERE / "habitat_gs_indoor_priority.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["scene_id"], row["indoor_hint_fraction"],
              row["episode_count"], row["outdoor_hint_count"], flush=True)
    print(output)


if __name__ == "__main__":
    main()
