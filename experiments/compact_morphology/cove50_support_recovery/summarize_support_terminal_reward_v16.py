"""Summarize frozen V16 terminal-reward checks on opened and fresh pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import mujoco

from summarize_support_reward_prefit_v15 import score_case, sha


def mapped_case(root: Path, model: mujoco.MjModel, side: str,
                label: str, source_stem: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="support-v16-score-") as tmp:
        stage = Path(tmp)
        for extension in ("json", "npz"):
            (stage / f"{side}-{label}.{extension}").symlink_to(
                (root / f"{source_stem}.{extension}").resolve()
            )
        row = score_case(stage, model, side, label)
        row["source_stem"] = source_stem
        return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    cases = (
        ("opened_right", "right", "reference", "right-reference"),
        ("opened_right", "right", "ppo005", "right-ppo005"),
        ("fresh_left", "left", "reference", "left-reference"),
        ("fresh_left", "left", "ppo005", "left-ppo005"),
        ("fresh_right", "right", "reference", "fresh-right-reference"),
        ("fresh_right", "right", "ppo005", "fresh-right-ppo005"),
    )
    rows = []
    for family, side, label, stem in cases:
        row = mapped_case(args.root, model, side, label, stem)
        row["family"] = family
        source = json.loads((args.root / f"{stem}.json").read_text())["reward_episode_preflight"]
        row["seed"] = source["seed"]
        row["terminal_proxy_bonus"] = source["reward_terms_sum"]["stage_a_terminal_proxy"]
        rows.append(row)
    comparisons = []
    for family in ("opened_right", "fresh_left", "fresh_right"):
        pair = {row["controller"]: row for row in rows if row["family"] == family}
        assert len(pair) == 2
        assert pair["reference"]["initial_state_sha256"] == pair["ppo005"]["initial_state_sha256"]
        comparisons.append({
            "family": family,
            "same_initial_state": True,
            "reference_full_gate": pair["reference"]["full_gate_pass"],
            "sampled_full_gate": pair["ppo005"]["full_gate_pass"],
            "reference_reward": pair["reference"]["reward_sum"],
            "sampled_reward": pair["ppo005"]["reward_sum"],
            "passing_reward_above_failing": (
                pair["reference"]["reward_sum"] > pair["ppo005"]["reward_sum"]
                if pair["reference"]["full_gate_pass"] and not pair["ppo005"]["full_gate_pass"]
                else None
            ),
        })
    result = {
        "schema": "robot-learning-lab.support-terminal-reward-v16.development-score.v1",
        "method_sha256": sha(args.root / "method.json"),
        "body_sha256": sha(args.body_xml),
        "scorer_sha256": sha(Path(__file__)),
        "opened_pairs": 1,
        "fresh_development_pairs": 2,
        "rows": rows,
        "comparisons": comparisons,
        "claim_limit": "One previously opened reward shortcut and two fresh local CPU starts; no optimizer updates, CUDA score or learned support policy. Offline raw-XML sole patch still needs assembled-scene parity.",
    }
    (args.root / "development-score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": [{k: row[k] for k in ("family", "controller", "seed", "full_gate_pass",
                                                   "failure_reasons", "reward_sum", "terminal_proxy_bonus")}
                               for row in rows], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
