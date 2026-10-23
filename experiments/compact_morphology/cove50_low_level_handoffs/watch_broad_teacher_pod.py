"""Local provider-side one-hour stop guard for the bounded V4 teacher screen."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time


STAMP = re.compile(r"(?:Rented by User|Started by user): (\w+ \w+ +\d+ \d+ \d+:\d+:\d+ GMT[+-]\d{4})")


def get_pod(pod_id: str) -> dict:
    return json.loads(subprocess.check_output(["runpodctl", "pod", "get", pod_id], text=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    pod = get_pod(args.pod_id)
    if pod["id"] != args.pod_id or pod["name"] != args.expected_name or pod["desiredStatus"] != "RUNNING":
        raise RuntimeError("Provider pod identity/status differs; refusing to guard another job")
    match = STAMP.search(pod["lastStatusChange"])
    if not match:
        raise RuntimeError("Provider start time unavailable")
    started = datetime.strptime(match.group(1), "%a %b %d %Y %H:%M:%S GMT%z").timestamp()
    deadline = started + 3600
    if started > time.time() or deadline <= time.time():
        raise RuntimeError("Provider start time is outside the one-hour pilot window")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"pod_id": args.pod_id, "name": args.expected_name,
                                       "pod_start_unix": started, "deadline_unix": deadline,
                                       "status": "watching"}, indent=2) + "\n")
    while time.time() < deadline:
        time.sleep(min(30, max(1, deadline - time.time())))
        pod = get_pod(args.pod_id)
        if pod["id"] != args.pod_id or pod["name"] != args.expected_name:
            raise RuntimeError("Provider pod identity changed")
        if pod["desiredStatus"] != "RUNNING":
            args.report.write_text(json.dumps({"pod_id": args.pod_id,
                                               "pod_start_unix": started,
                                               "deadline_unix": deadline,
                                               "status": "already_stopped"}, indent=2) + "\n")
            return
    subprocess.run(["runpodctl", "pod", "stop", args.pod_id], check=True)
    pod = get_pod(args.pod_id)
    args.report.write_text(json.dumps({"pod_id": args.pod_id,
                                       "pod_start_unix": started, "deadline_unix": deadline,
                                       "status": "stopped_at_deadline",
                                       "provider_status": pod["desiredStatus"]}, indent=2) + "\n")
    if pod["desiredStatus"] == "RUNNING":
        raise RuntimeError("Provider still reports paid pod running after stop")


if __name__ == "__main__":
    main()
