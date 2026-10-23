"""Save the compiled native actor/action contract for the touch teacher set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv

from free_object_touch_env import ROOT, touch_env_cfg


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    cfg = touch_env_cfg(body, entry_mode="stand-tabletop")
    cfg.scene.num_envs = 1
    cfg.seed = 30005
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset(seed=30005)
        term = env.action_manager.get_term("joint_pos")
        report = {
            "schema": "robot-learning-lab.cove50-free-object-touch-interface.v1",
            "body_sha256": digest(body),
            "environment_source_sha256": digest(ROOT / "experiments/compact_morphology/cove50_low_level/free_object_touch_env.py"),
            "observation_source_sha256": digest(ROOT / "experiments/compact_morphology/cove50_policy_transfer/reach_observation.py"),
            "actor_terms": env.observation_manager.active_terms["actor"],
            "actor_shape": list(obs["actor"].shape),
            "actor_last_six": ["rgb_centroid_u", "rgb_centroid_v", "rgb_red_area", "rgb_fresh", "rgb_age_fraction", "rgb_missing"],
            "critic_terms": env.observation_manager.active_terms["critic"],
            "critic_shape": list(obs["critic"].shape),
            "action_names_in_order": list(term._target_names),
            "action_shape": list(env.action_space.shape),
            "action_offsets_rad": term._offset[0].detach().cpu().tolist(),
            "action_scales_rad": term._scale[0].detach().cpu().tolist(),
            "first_rgb_target": obs["actor"][0, -6:].detach().cpu().tolist(),
            "step_dt_s": env.step_dt,
            "claim_limit": "Native source/interface contract only. Actor RGB is reduced to six values; no object truth/contact is exposed to actor."
        }
        assert report["actor_terms"] == ["joint_pos", "joint_vel", "actions", "rgb_target"]
        assert report["actor_shape"] == [1, 69] and report["action_shape"] == [1, 21]
        assert report["first_rgb_target"][3] == 1 and report["first_rgb_target"][5] == 0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"out": str(args.out), "actor_shape": report["actor_shape"],
                          "action_shape": report["action_shape"]}))
    finally:
        env.close()


if __name__ == "__main__":
    main()
