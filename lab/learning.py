"""Local episode provenance and language feedback records; no text execution."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .config import Strict
from .tasks import PickPlaceTask


class TaskRequest(Strict):
    schema_version: Literal[1] = 1
    instruction: str = Field(min_length=1, max_length=2000)
    task: PickPlaceTask
    interpretation: Literal["explicit_structured_task"] = "explicit_structured_task"


class EpisodeFeedback(Strict):
    schema_version: Literal[1] = 1
    episode_id: str = Field(pattern=r"^episode-[0-9a-f]{24}$")
    text: str = Field(min_length=1, max_length=4000)
    preference: Literal["positive", "negative", "neutral"]
    source: Literal["human", "synthetic_test"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_episode(output, request, seed, controller, initial_observation):
    """Content identity includes initial state, all retained source hashes and outcome."""
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    record = {
        "schema_version": 1,
        "request": request.model_dump(),
        "robot_config_sha256": digest(output / "resolved-config.json"),
        "scene_sha256": digest(output / "scene.xml"),
        "trajectory_sha256": digest(output / "trajectory.jsonl"),
        "report_sha256": digest(output / "report.json"),
        "seed": seed,
        "controller": controller,
        "initial_observation": initial_observation,
        "result": report["status"],
        "feedback_role": "separate preference record; does not overwrite task success or reward",
    }
    identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    record["episode_id"] = "episode-" + identity[:24]
    with (output / "episode.json").open("x") as stream:
        json.dump(record, stream, indent=2)
    return record


def write_feedback(episode_dir, feedback: EpisodeFeedback, destination):
    """Write one immutable event, bound to an unchanged retained episode."""
    episode_dir = Path(episode_dir)
    record = json.loads((episode_dir / "episode.json").read_text())
    unsigned = {k: v for k, v in record.items() if k != "episode_id"}
    expected_id = (
        "episode-"
        + hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()[:24]
    )
    if record["episode_id"] != expected_id:
        raise ValueError("episode record identity changed")
    if feedback.episode_id != record["episode_id"]:
        raise ValueError("feedback episode ID mismatch")
    for filename, key in [
        ("trajectory.jsonl", "trajectory_sha256"),
        ("report.json", "report_sha256"),
        ("resolved-config.json", "robot_config_sha256"),
        ("scene.xml", "scene_sha256"),
    ]:
        if digest(episode_dir / filename) != record[key]:
            raise ValueError(f"episode artifact changed: {filename}")
    event = {
        **feedback.model_dump(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "episode_record_sha256": digest(episode_dir / "episode.json"),
    }
    with Path(destination).open("x") as stream:
        json.dump(event, stream, indent=2)
    return event
