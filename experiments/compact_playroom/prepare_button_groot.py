"""Add the GR00T-specific language and modality metadata to LeRobot v2 button data."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--state-has-goal", action="store_true")
    args = parser.parse_args()
    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, found {info['codebase_version']}")
    for path in sorted((args.dataset / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        task_indices = table.column("task_index").cast(pa.int64())
        if "annotation.human.task_description" in table.column_names:
            if not table.column("annotation.human.task_description").equals(task_indices):
                raise ValueError(f"Language annotation conflicts with task_index in {path}")
        else:
            table = table.append_column("annotation.human.task_description", task_indices)
            pq.write_table(table, path)
    state_modality = (
        {
            "button_relative": {"start": 0, "end": 3, "original_key": "observation.state"},
            "hand_to_button": {"start": 3, "end": 6, "original_key": "observation.state"},
            "body_motion": {"start": 6, "end": 12, "original_key": "observation.state"},
            "left_arm": {"start": 12, "end": 17, "original_key": "observation.state"},
            "previous_action": {"start": 17, "end": 25, "original_key": "observation.state"},
        }
        if args.state_has_goal else
        {
            "body_motion": {"start": 0, "end": 6, "original_key": "observation.state"},
            "left_arm": {"start": 6, "end": 11, "original_key": "observation.state"},
            "previous_action": {"start": 11, "end": 19, "original_key": "observation.state"},
        }
    )
    modality = {
        "state": state_modality,
        "action": {
            "walk": {"start": 0, "end": 3, "original_key": "action"},
            "left_arm": {"start": 3, "end": 8, "original_key": "action"},
        },
        "video": {"external_view": {"original_key": "observation.images.external_view"}},
        "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}},
    }
    (args.dataset / "meta/modality.json").write_text(json.dumps(modality, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.dataset), "episodes": info["total_episodes"], "frames": info["total_frames"]}))


if __name__ == "__main__":
    main()
