"""Compare first-step and early-contact evidence in opened V26 CUDA pairs.

This is an offline diagnostic. It neither advances physics nor changes a score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
OUTPUT = ROOT / "artifacts/ape-cove50-stand-recovery/seated-first-divergence-v27"
SEEDS = tuple(range(2231, 2237))
DT = 0.02


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_time(condition: np.ndarray) -> float | None:
    hits = np.flatnonzero(condition)
    return round((int(hits[0]) + 1) * DT, 2) if len(hits) else None


def mean_terms(rows: list[dict], key: str, term_names: list[str]) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float).mean(axis=0)
    return {name: float(value) for name, value in zip(term_names, values)}


def main() -> None:
    source_report_path = SOURCE / "report.json"
    source_report = json.loads(source_report_path.read_text())
    assert source_report["status"] == "complete_source_checked_cuda_opened_development_diagnostic"
    assert source_report["summary"]["opened_seed_families"] == len(SEEDS)
    source_rows = {(row["seed"], row["route"]): row for row in source_report["rows"]}
    assert set(source_rows) == {(seed, route) for seed in SEEDS for route in ("model25", "scripted")}

    pairs = []
    term_names = None
    for seed in SEEDS:
        routes = {}
        for route in ("model25", "scripted"):
            row = source_rows[(seed, route)]
            report_path = ROOT / row["source_report"]
            arrays_path = ROOT / row["native_trajectory"]
            assert sha(report_path) == row["source_report_sha256"]
            assert sha(arrays_path) == row["native_trajectory_sha256"]
            report = json.loads(report_path.read_text())
            assert report["seed"] == seed and report["device"] == "cuda:0"
            assert report["body_sha256"] == source_report["body_sha256"]
            assert report["initial_state_sha256"] == row["initial_state_sha256"]
            assert len(report["timeseries"]) == 500
            with np.load(arrays_path, allow_pickle=False) as native:
                arrays = {name: native[name].copy() for name in (
                    "actor_observation", "raw_action", "processed_target",
                    "qpos", "reward_rate"
                )}
            assert arrays["actor_observation"].shape == (500, 73)
            assert arrays["raw_action"].shape == (500, 21)
            assert arrays["processed_target"].shape == (500, 21)
            assert arrays["reward_rate"].shape == (500, 15)
            assert all(np.isfinite(value).all() for value in arrays.values())
            routes[route] = (row, report, arrays)

        learned_row, learned, learned_arrays = routes["model25"]
        script_row, script, script_arrays = routes["scripted"]
        assert learned_row["initial_state_sha256"] == script_row["initial_state_sha256"]
        assert learned["reward_term_names"] == script["reward_term_names"]
        names = script["reward_term_names"]
        if term_names is None:
            term_names = names
        assert names == term_names
        joint_names = script["initial_motors"]["target_names"]
        assert len(joint_names) == 21

        observation_gap = float(np.max(np.abs(
            learned_arrays["actor_observation"][0] -
            script_arrays["actor_observation"][0]
        )))
        assert observation_gap == 0.0
        action_gap = np.abs(learned_arrays["raw_action"][0] - script_arrays["raw_action"][0])
        target_gap = np.abs(learned_arrays["processed_target"][0] - script_arrays["processed_target"][0])
        top_actions = [
            {"joint": joint_names[int(index)], "abs_raw_action_gap": float(action_gap[index])}
            for index in np.argsort(-action_gap)[:5]
        ]
        learned_both = np.asarray([all(step["feet_grounded"]) for step in learned["timeseries"]])
        anchor = next((i for i in range(496) if learned_both[i:i + 5].all()), None)
        assert anchor is not None
        lost = np.flatnonzero(~learned_both[anchor:]) + anchor
        first_loss = int(lost[0]) if len(lost) else None
        first_loss_time = round((first_loss + 1) * DT, 2) if first_loss is not None else None
        first_loss_side = (
            "left" if not learned["timeseries"][first_loss]["feet_grounded"][0]
            else "right" if first_loss is not None else None
        ) if first_loss is not None else None
        height_threshold = 0.22
        learned_height = first_time(np.asarray([
            step["pelvis_z_m"] >= height_threshold for step in learned["timeseries"]
        ]))
        script_height = first_time(np.asarray([
            step["pelvis_z_m"] >= height_threshold for step in script["timeseries"]
        ]))
        reward_difference = (script_arrays["reward_rate"] - learned_arrays["reward_rate"]) * DT
        reward_windows = {
            "first_0p2_s": reward_difference[:10].sum(axis=0).tolist(),
            "first_1p0_s": reward_difference[:50].sum(axis=0).tolist(),
            "remainder_1_to_10_s": reward_difference[50:].sum(axis=0).tolist(),
        }
        pairs.append({
            "seed": seed,
            "initial_state_sha256": learned_row["initial_state_sha256"],
            "first_actor_observation_max_abs_gap": observation_gap,
            "first_raw_action_max_abs_gap": float(action_gap.max()),
            "first_processed_target_max_abs_gap_rad": float(target_gap.max()),
            "largest_first_raw_action_differences": top_actions,
            "learned_first_five_frame_double_support_anchor_s": round((anchor + 1) * DT, 2),
            "learned_first_post_anchor_support_loss_s": first_loss_time,
            "learned_first_lift_side": first_loss_side,
            "learned_height_0p22m_s": learned_height,
            "scripted_height_0p22m_s": script_height,
            "reward_script_minus_learned_by_term": reward_windows,
            "scripted_full_gate": bool(script_row["full_gate"]),
            "learned_full_gate": bool(learned_row["full_gate"]),
        })

    assert term_names is not None
    windows = {
        name: mean_terms([
            {name: pair["reward_script_minus_learned_by_term"][name]} for pair in pairs
        ], name, term_names)
        for name in ("first_0p2_s", "first_1p0_s", "remainder_1_to_10_s")
    }
    assert all(pair["scripted_full_gate"] and not pair["learned_full_gate"] for pair in pairs)
    result = {
        "schema": "robot-learning-lab.cove50-seated-first-divergence.v27",
        "status": "source_checked_offline_opened_development_diagnostic",
        "source_v26_report": str(source_report_path.relative_to(ROOT)),
        "source_v26_report_sha256": sha(source_report_path),
        "analysis_source": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_source_sha256": sha(Path(__file__)),
        "body_sha256": source_report["body_sha256"],
        "seed_count": len(pairs),
        "new_cuda_visits": 0,
        "optimizer_updates": 0,
        "reward_term_names": term_names,
        "mean_reward_script_minus_learned_by_term": windows,
        "pairs": pairs,
        "interpretation_limit": "Paired action/target/height/contact timing and reward contributions identify the first observed divergence, not why PPO selected it. No per-state critic predictions or corrective-teacher branch are in V26 arrays.",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "report": str((OUTPUT / "report.json").relative_to(ROOT)),
        "seeds": len(pairs),
        "first_support_loss_s": [p["learned_first_post_anchor_support_loss_s"] for p in pairs],
        "height_0p22_learned_s": [p["learned_height_0p22m_s"] for p in pairs],
        "height_0p22_script_s": [p["scripted_height_0p22m_s"] for p in pairs],
    }, indent=2))


if __name__ == "__main__":
    main()
