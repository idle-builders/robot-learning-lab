#!/usr/bin/env python3
"""Launch the official Flash CLI using the existing runpodctl credential.

runpodctl 2.14 writes `apikey`; Flash 1.19 reads `api_key`. Bridge in the
child environment only; do not rewrite either tool's credentials file.
"""
import os
from pathlib import Path
import sys
import tomllib

root = Path(__file__).resolve().parents[1]
if not os.environ.get("RUNPOD_API_KEY"):
    config = tomllib.loads((Path.home() / ".runpod/config.toml").read_text())
    key = config.get("apikey") or config.get("api_key")
    if not isinstance(key, str) or not key.strip():
        raise SystemExit("No saved Runpod credential found; authenticate runpodctl first.")
    os.environ["RUNPOD_API_KEY"] = key
os.chdir(root / "workers/serverless")
binary = str(root / ".venv-runpod/bin/flash")
os.execv(binary, [binary, *sys.argv[1:]])
