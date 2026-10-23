"""Fetch the two frozen image-model revisions used by Replica policy v4."""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEST = Path(__file__).resolve().parents[1] / ".cache/ape-high-level/models"
SOURCES = {
    "grounding-dino-tiny": ("IDEA-Research/grounding-dino-tiny",
                            "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "dinov2-small": ("facebook/dinov2-small",
                      "ed25f3a31f01632728cabb09d1542f84ab7b0056"),
}


def main() -> None:
    api = HfApi()
    for folder, (repo, revision) in SOURCES.items():
        resolved = api.model_info(repo, revision=revision)
        if resolved.sha != revision:
            raise ValueError(f"Changed model revision for {repo}: {resolved.sha}")
        snapshot_download(repo_id=repo, revision=revision,
                          local_dir=DEST / folder)
        print(folder, revision, flush=True)


if __name__ == "__main__":
    main()
