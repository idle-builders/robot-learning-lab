"""Independent provider stop for the bounded assisted-release diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261015/support-assist-feasibility"


def record(event: dict) -> None:
    with (OUT / "watchdog-events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(), **event}) + "\n")
        stream.flush()


def cli(*args: str) -> object:
    result = subprocess.run(["runpodctl", *args], capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[:500])
    return json.loads(result.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        raise ValueError("Deadline needs a timezone")
    (OUT / "watchdog-armed.json").write_text(json.dumps({
        "pod_id": args.pod, "deadline_utc": deadline.isoformat(),
        "armed_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n")
    record({"event": "armed", "pod_id": args.pod, "deadline_utc": deadline.isoformat()})
    while datetime.now(timezone.utc) < deadline:
        time.sleep(min(15, max(1, (deadline - datetime.now(timezone.utc)).total_seconds())))
    for attempt in range(12):
        try:
            pod = cli("pod", "get", args.pod)
            if pod.get("runtimeStatus") == "stopped":
                record({"event": "already_stopped", "attempt": attempt + 1})
                return
            cli("pod", "stop", args.pod)
            record({"event": "stop_requested", "attempt": attempt + 1})
        except Exception as exc:
            record({"event": "stop_failed", "attempt": attempt + 1, "error": str(exc)})
        time.sleep(10)
    record({"event": "manual_stop_required"})


if __name__ == "__main__":
    main()
