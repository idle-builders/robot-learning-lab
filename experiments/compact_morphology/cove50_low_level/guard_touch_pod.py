"""Detached host hard stop for the one-pad A40 pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261015/object-touch-ppo"


def event(**data) -> None:
    with (OUT / "pod-guard-events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **data}) + "\n")
        stream.flush()


def pod_status(pod_id: str) -> dict:
    result = subprocess.run(("runpodctl", "pod", "get", pod_id),
                            text=True, capture_output=True, timeout=60, check=True)
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        parser.error("deadline must include timezone")
    (OUT / "pod-guard-armed.json").write_text(json.dumps({"pod_id": args.pod_id,
        "armed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": deadline.isoformat()}, indent=2) + "\n")
    event(action="armed", pod_id=args.pod_id, deadline_utc=deadline.isoformat())
    while datetime.now(timezone.utc) < deadline:
        time.sleep(min(15, max(1, (deadline - datetime.now(timezone.utc)).total_seconds())))
    for attempt in range(20):
        try:
            status = pod_status(args.pod_id)
            if status.get("runtimeStatus") == "stopped":
                event(action="provider_stopped_confirmed", attempt=attempt + 1)
                return
            event(action="stop_requested", attempt=attempt + 1)
            subprocess.run(("runpodctl", "pod", "stop", args.pod_id),
                           text=True, capture_output=True, timeout=60, check=True)
        except Exception as exc:
            event(action="stop_or_query_failed", attempt=attempt + 1, reason=str(exc))
        time.sleep(10)
    event(action="guard_exhausted_requires_manual_stop")


if __name__ == "__main__":
    main()
