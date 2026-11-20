#!/usr/bin/env python3
"""Foreground Terraform entry point; credentials stay in the child environment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
INFRA = ROOT / "infra/compute/runpod"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "fmt", "validate", "plan", "apply", "output"))
    args = parser.parse_args()
    commands = {
        "init": ["init", "-input=false"],
        "fmt": ["fmt", "-check"],
        "validate": ["validate"],
        "plan": ["plan", "-input=false", "-parallelism=1", "-out=bootstrap.tfplan"],
        "apply": ["apply", "-input=false", "-parallelism=1", "bootstrap.tfplan"],
        "output": ["output", "-json"],
    }
    # A foreground lock prevents overlapping local plans/applies. No daemon or retry.
    with (INFRA / ".operation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another Runpod infrastructure operation is active.")
        env = dict(os.environ)
        if args.action in ("plan", "apply"):
            from runpod_flash.core.credentials import get_api_key
            key = get_api_key()
            if not key:
                raise SystemExit("Authenticate with .venv-runpod/bin/flash login first.")
            env["RUNPOD_API_KEY"] = key
            account = INFRA / "account.json"  # Git-ignored; copy account.example.json and fill in
            if not account.exists():
                raise SystemExit("Create infra/compute/runpod/account.json from account.example.json first.")
            expected = json.loads(account.read_text())["account_id"]
            result = subprocess.run(["runpodctl", "user"], env=env, capture_output=True,
                                    text=True, timeout=40)
            if result.returncode:
                raise SystemExit("Account lookup failed; stop here. No automatic retries. "
                                 "Inspect rate limits before another operation.")
            if json.loads(result.stdout).get("id") != expected:
                raise SystemExit("Authenticated account differs from infra/compute/runpod/account.json.")
        # A single bounded operation; API errors are not retried by this launcher.
        result = subprocess.run(["terraform", *commands[args.action]], cwd=INFRA,
                                env=env, timeout=300)
        if result.returncode == 0 and args.action == "apply":
            (INFRA / "bootstrap.tfplan").unlink(missing_ok=True)
        if result.returncode == 0 and args.action in ("apply", "output"):
            outputs = subprocess.run(["terraform", "output", "-json"], cwd=INFRA,
                                     capture_output=True, text=True, check=True, timeout=30)
            transport = json.loads(outputs.stdout)["transport"]["value"]
            config = {"bucket": transport["id"], "region": transport["data_center"],
                      "endpoint": transport["s3_endpoint"], "prefix": "robot-learning-lab/v1"}
            (ROOT / "infra/compute/serverless/storage_config.py").write_text(
                '# Generated from Terraform outputs; contains no credentials.\nCONFIG = '
                + repr(config) + '\n')
            for state in INFRA.glob("*.tfstate*"):
                state.chmod(0o600)
        return result.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.TimeoutExpired:
        raise SystemExit("Foreground operation timed out. Inspect state before retrying; "
                         "a provider mutation may have completed.")
