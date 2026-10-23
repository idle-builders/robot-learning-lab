"""Fetch the exact image-model revisions sealed by the Scan30 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.precommit.read_text())
    models = frozen["models"]
    names = {"detector": "grounding-dino-tiny",
             "retriever": "dinov2-small",
             "click_segmenter": "sam2-hiera-tiny"}
    api = HfApi()
    result = {}
    for key, folder in names.items():
        source = models[key]
        revision = source["revision"]
        info = api.model_info(source["repo"], revision=revision)
        if info.sha != revision:
            raise ValueError(f"Resolved a different {key} revision: {info.sha}")
        target = args.destination / folder
        snapshot_download(repo_id=source["repo"], revision=revision,
                          local_dir=target)
        result[key] = {"repo": source["repo"], "revision": revision,
                       "local_dir": str(target)}
        print(key, revision, flush=True)
    (args.destination / "frozen_revisions.json").write_text(
        json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
