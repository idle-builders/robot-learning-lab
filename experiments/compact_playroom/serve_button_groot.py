"""Expose NVIDIA's GR00T PolicyServer with its supported API-token check."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    token = os.environ.get("GROOT_API_TOKEN") or Path("/workspace/groot-api-token").read_text().strip()
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        device="cuda:0",
        strict=True,
    )
    PolicyServer.start_server(policy=policy, host=args.host, port=args.port, api_token=token)


if __name__ == "__main__":
    main()
