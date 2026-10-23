"""Check retrieved original V4 paired visits and publish a flat score matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015"
SCORE = BASE / "broad-handoff-v4-l4-score"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method_path = BASE / "broad-handoff-v4-l4-score-method.json"
    method = json.loads(method_path.read_text())
    scored = json.loads((SCORE / "report.json").read_text())
    assert scored["method_sha256"] == digest(method_path)
    assert scored["planned"] == scored["completed"] == len(method["cases"]) == 4
    assert [row["case"] for row in scored["rows"]] == method["cases"]
    assert not (SCORE / "incomplete.json").exists()
    bank = json.loads((BASE / "broad-handoff-v4-l4-bank/report.json").read_text())
    assert not {row["case"]["seed"] for row in scored["rows"]} & {
        entry["seed"] for entry in bank["entries"]
    }
    rows = []
    for pair in scored["rows"]:
        case = pair["case"]
        assert pair["first_qpos_gap"] <= method["matched_entry_tolerance"]
        assert pair["first_obs_gap"] <= method["matched_entry_tolerance"]
        original = {}
        for role in ("teacher", "student"):
            item = pair[role]
            files = {key: SCORE / Path(item[key]).name for key in ("report", "trace", "log")}
            for key, path in files.items():
                assert path.is_file() and digest(path) == item[f"{key}_sha256"], path
            report = json.loads(files["report"].read_text())
            assert report["seed"] == case["seed"]
            assert report["device"] == "cuda:0" and report["body_sha256"] == method["body"]["sha256"]
            assert report["rail_v"] == case["rail_v"] and report["delay_ms"] == case["delay_ms"]
            assert report["walk_velocity_m_s"] == case["walk_velocity_m_s"]
            assert report["steps"] == item["steps"] == 1200
            assert report["full_continuity_gate_pass"] == item["pass"]
            assert report["switch_qpos_gap"] == 0.0
            expected = (method["actors"] if role == "teacher" else method["student_checkpoint"])
            if role == "teacher":
                assert all(report["checkpoint_sha256"][name] == spec["sha256"]
                           for name, spec in expected.items())
            else:
                assert report["student_checkpoint_sha256"] == expected["sha256"]
                assert report["max_target_mapping_gap"] <= 1e-5
            with np.load(files["trace"], allow_pickle=False) as trace:
                pre, post, obs = (trace[name] for name in ("pre_qpos", "post_qpos", "actor_obs"))
                assert len(pre) == len(post) == len(obs) == 1200
                assert obs.shape == (1200, 72)
                assert np.isfinite(pre).all() and np.isfinite(post).all() and np.isfinite(obs).all()
                assert np.max(np.abs(pre[0] - report["initial_qpos"])) < 1e-8
                assert np.max(np.abs(post[:-1] - pre[1:])) < 1e-8
                original[role] = (pre[0].copy(), obs[0].copy())
        assert np.max(np.abs(original["teacher"][0] - original["student"][0])) == pair["first_qpos_gap"]
        assert np.max(np.abs(original["teacher"][1] - original["student"][1])) == pair["first_obs_gap"]
        assert pair["rescue"] == (not pair["teacher"]["pass"] and pair["student"]["pass"])
        assert pair["regression"] == (pair["teacher"]["pass"] and not pair["student"]["pass"])
        rows.append({"seed": case["seed"], "walk_velocity_m_s": case["walk_velocity_m_s"],
                     "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
                     "teacher_pass": pair["teacher"]["pass"], "student_pass": pair["student"]["pass"],
                     "rescue": pair["rescue"], "regression": pair["regression"],
                     "student_failures": pair["student"]["failure_reasons"],
                     "teacher_failures": pair["teacher"]["failure_reasons"],
                     "first_qpos_gap": pair["first_qpos_gap"], "first_obs_gap": pair["first_obs_gap"]})
    assert sum(row["teacher_pass"] for row in rows) == scored["teacher_passes"]
    assert sum(row["student_pass"] for row in rows) == scored["student_passes"]
    assert sum(row["rescue"] for row in rows) == scored["rescues"]
    assert sum(row["regression"] for row in rows) == scored["regressions"]
    result = {"schema": "robot-learning-lab.broad-handoff-v4-l4-local-source-check.v1",
              "classification": "retrieved original matched CUDA full episodes; no mechanical transfer",
              "method_sha256": digest(method_path), "score_report_sha256": digest(SCORE / "report.json"),
              "planned": scored["planned"], "completed": scored["completed"],
              "teacher_passes": scored["teacher_passes"], "student_passes": scored["student_passes"],
              "rescues": scored["rescues"], "regressions": scored["regressions"], "rows": rows}
    (SCORE / "source-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
