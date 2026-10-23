"""Private archive settings. Public clones have no configured archive by default."""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "data/private-archive.json"


def settings():
    local = json.loads(LOCAL_CONFIG.read_text()) if LOCAL_CONFIG.exists() else {}
    return {
        "bucket": os.environ.get("ROBOTLAB_ARCHIVE_BUCKET", local.get("bucket")),
        "endpoint": os.environ.get("ROBOTLAB_ARCHIVE_ENDPOINT", local.get(
            "endpoint", "https://s3.pub2.infomaniak.cloud")),
        "region": os.environ.get("ROBOTLAB_ARCHIVE_REGION", local.get("region", "us-east-1")),
        "keychain_service_prefix": local.get("keychain_service_prefix"),
    }


def client(*, endpoint=None, region=None):
    config = settings()
    if not config["bucket"]:
        raise RuntimeError("Configure ROBOTLAB_ARCHIVE_BUCKET or data/private-archive.json to use a private archive")
    endpoint = endpoint or config["endpoint"]
    if not endpoint.startswith("https://"):
        raise ValueError("Archive endpoint must use HTTPS")
    import boto3
    from botocore.config import Config

    auth = {}
    access, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if bool(access) != bool(secret):
        raise RuntimeError("Set both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    prefix = config["keychain_service_prefix"]
    if not access and prefix and sys.platform == "darwin":
        def key(name):
            return subprocess.run(
                ["security", "find-generic-password", "-s", f"{prefix}/{name}",
                 "-a", getpass.getuser(), "-w"], check=True, capture_output=True,
                text=True).stdout.strip()
        access, secret = key("AWS_ACCESS_KEY_ID"), key("AWS_SECRET_ACCESS_KEY")
    if access and secret:
        auth = {"aws_access_key_id": access, "aws_secret_access_key": secret}
        if os.environ.get("AWS_SESSION_TOKEN"):
            auth["aws_session_token"] = os.environ["AWS_SESSION_TOKEN"]
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name=region or config["region"], **auth,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"},
                      max_pool_connections=16, connect_timeout=15, read_timeout=120,
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"))
