"""Initialize native PPO's actor to a physics-held one-foot motor target.

This is an initial controller, not a learned hold or a motion demonstration.
Pollen/MJLab PPO must subsequently improve it and pass fresh replay.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("constant_target_report", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    args = parser.parse_args()

    report = json.loads(args.constant_target_report.read_text())
    assert report["longest_single_support_s"] >= 5.5
    assert not report["fell_or_terminated"]
    action = torch.as_tensor(report["hold_action"], dtype=torch.float32)
    checkpoint = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
    actor = checkpoint["actor_state_dict"]
    final_weight, final_bias = actor["mlp.6.weight"], actor["mlp.6.bias"]
    assert action.shape == final_bias.shape == torch.Size([22])
    final_weight.zero_()
    final_bias.copy_(action)
    checkpoint["iter"] = 0
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    manifest = {
        "status": "initialized_controller_not_learned",
        "source_checkpoint": str(args.initial_checkpoint),
        "source_sha256": sha256(args.initial_checkpoint),
        "constant_target_report": str(args.constant_target_report),
        "report_sha256": sha256(args.constant_target_report),
        "output_checkpoint": str(args.output_checkpoint),
        "output_sha256": sha256(args.output_checkpoint),
        "action": action.tolist(),
        "changed_parameters": ["actor.mlp.6.weight", "actor.mlp.6.bias"],
    }
    args.output_checkpoint.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "action"}, indent=2))


if __name__ == "__main__":
    main()
