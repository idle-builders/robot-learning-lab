"""Last-resort provider stop for the one paid multi-command CUDA pilot."""

import argparse
from datetime import datetime, timezone
import json
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pod", required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    while datetime.now(timezone.utc) < deadline:
        time.sleep(10)
    for attempt in range(12):
        state = subprocess.run(["runpodctl", "pod", "get", args.pod],
                               text=True, capture_output=True, check=False)
        if state.returncode == 0:
            pod = json.loads(state.stdout)
            if pod.get("runtimeStatus") == "stopped":
                print("already stopped", flush=True)
                return
        result = subprocess.run(["runpodctl", "pod", "stop", args.pod],
                                text=True, capture_output=True, check=False)
        print(f"stop attempt {attempt + 1}: {result.returncode}", flush=True)
        time.sleep(10)
    raise RuntimeError("provider stop not confirmed")


if __name__ == "__main__":
    main()
