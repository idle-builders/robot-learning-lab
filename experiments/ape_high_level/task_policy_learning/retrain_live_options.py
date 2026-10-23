"""Retrain the event policy with teacher corrections at learner-visited states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

import train_event_policy as base

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def teacher_for(phase: str, features: dict) -> str:
    """Annotate a saved learner-visited state using the frozen safe teacher."""
    if phase == "hazard_interruption":
        return "HOLD"
    if phase == "source_restored":
        return "REOBSERVE" if features["restored_source_healthy"] else "HOLD"
    if phase == "owner_result":
        return "ASK_RESHOW" if features["owner_safe_to_ask"] else "HOLD"
    if phase == "search_result":
        return "FRESH_RELOOK" if features["present_confirmed"] else "WIDE_LOOK"
    if phase == "wide_result":
        return "HOLD" if features.get("final_wide_source_healthy") is False else (
            "FRESH_RELOOK" if features["resolved_confirmed"] else "ASK_OWNER")
    if phase == "pre_motion":
        return base.safe_gate("APPROACH", features)
    if phase == "skill_result":
        return "WAIT_SOURCE" if features["source_gap"] else (
            "FINAL_REVIEW" if features["at_standoff"] else "HOLD")
    if phase == "route_blocked":
        return "ASK_OWNER" if features["source_healthy"] and \
            features["blocked_route_observed"] else "HOLD"
    if phase == "final_review":
        if not features["final_source_healthy"]:
            return "HOLD"
        if features["final_confirmed"]:
            return "COMPLETE"
        if features.get("final_retry_exhausted"):
            return "ASK_OWNER"
        if features.get("final_no_candidate"):
            return "WIDE_LOOK"
        return "HOLD"
    raise ValueError(f"No teacher for {phase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", action="append", type=Path, required=True)
    parser.add_argument("--output-policy", type=Path,
                        default=HERE / "event_policy_live_v2.joblib")
    parser.add_argument("--output-report", type=Path,
                        default=HERE / "live_retrain_report.json")
    args = parser.parse_args()
    records, sources = base.extract()
    rows = base.read(base.REPORTS.relative_to(base.ROOT) / base.SOURCES[0])["rows"]
    split = {r["capture_group"]: r["cohort"] for r in rows}
    old_train = [r for r in records if split[r["capture_group"]] == "development"]
    holdout = [r for r in records if split[r["capture_group"]] == "prior_transfer"]
    live: list[dict] = []
    episode_sources = []
    for path in args.episode:
        data = json.loads(path.read_text())
        if data.get("schema") != "ape.learner-option-episode.v1":
            raise ValueError(f"Not a live option episode: {path}")
        if data["cohort"] != "development":
            raise ValueError("Only development rooms may train the policy")
        for event in data["events"]:
            if event["phase"] == "owner_point_result":
                continue  # This bounded transition was not learned in v1.
            # Early pilot logs carried a counterfactual absent-control bit.
            # It was not observed in the positive episode and cannot train.
            features = {k: v for k, v in event["observation"].items()
                        if k != "absent_unresolved"}
            teacher = teacher_for(event["phase"], features)
            if event.get("teacher", teacher) != teacher:
                raise ValueError(f"Saved teacher disagrees with annotation: {path}")
            live.append({"capture_group": data["capture_group"],
                "source": str(path), "phase": event["phase"],
                "features": features, "teacher": teacher,
                "requested": event["requested"], "executed": event["executed"]})
        episode_sources.append({"path": str(path), "sha256": digest(path),
            "result": data["result"], "fault": data.get("fault", "none")})
    vectorizer = DictVectorizer(sparse=False)
    examples = old_train + live
    x = vectorizer.fit_transform([r["features"] for r in examples])
    model = RandomForestClassifier(n_estimators=160, max_depth=7,
        min_samples_leaf=1, random_state=17, class_weight="balanced_subsample")
    model.fit(x, [r["teacher"] for r in examples],
              sample_weight=[1.] * len(old_train) + [4.] * len(live))
    old = joblib.load(HERE / "event_policy.joblib")
    x_hold = vectorizer.transform([r["features"] for r in holdout])
    old_hold = [base.safe_gate(str(a), r["features"]) for a, r in zip(
        old["model"].predict(old["vectorizer"].transform(
            [r["features"] for r in holdout])), holdout)]
    new_hold = [base.safe_gate(str(a), r["features"]) for a, r in zip(
        model.predict(x_hold), holdout)]
    x_live_old = old["vectorizer"].transform([r["features"] for r in live])
    x_live_new = vectorizer.transform([r["features"] for r in live])
    old_raw = [str(a) for a in old["model"].predict(x_live_old)]
    new_raw = [str(a) for a in model.predict(x_live_new)]
    adapted_phases = sorted({r["phase"] for r in live
                             if r["requested"] != r["teacher"]})
    composed_hold = [new if r["phase"] in adapted_phases else previous
                     for new, previous, r in zip(new_hold, old_hold, holdout)]
    report = {"schema": "ape.high-level-live-dagger-pilot.v1",
        "limits": "Small DAgger-style correction pilot on previously exposed development rooms. Offline authored events still dominate training; holdout capture groups have been used elsewhere in the project, so they are not blind. Only options after initial owner teaching are learner-controlled. No policy robustness or body/sensor qualification is claimed.",
        "base_source_sha256": sources,
        "episode_sources": episode_sources,
        "offline_train_events": len(old_train), "live_train_events": len(live),
        "learner_visited_corrections": sum(r["requested"] != r["teacher"]
                                          for r in live),
        "live_teacher_counts": dict(Counter(r["teacher"] for r in live)),
        "adapted_phases": adapted_phases,
        "holdout_events": len(holdout),
        "old_gated_holdout_correct": sum(a == r["teacher"] for a, r in
                                          zip(old_hold, holdout)),
        "new_gated_holdout_correct": sum(a == r["teacher"] for a, r in
                                          zip(new_hold, holdout)),
        "composed_gated_holdout_correct": sum(a == r["teacher"] for a, r in
                                               zip(composed_hold, holdout)),
        "live_decisions": [{"scene": r["capture_group"], "phase": r["phase"],
            "teacher": r["teacher"], "old_raw": a, "new_raw": b,
            "new_authorized": base.safe_gate(b, r["features"])}
            for r, a, b in zip(live, old_raw, new_raw)]}
    joblib.dump({"schema": report["schema"], "vectorizer": vectorizer,
                 "model": model, "options": base.OPTIONS,
                 "adapted_phases": adapted_phases,
                 "fallback_model": old["model"],
                 "fallback_vectorizer": old["vectorizer"]},
                args.output_policy)
    args.output_report.write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "offline_train_events", "live_train_events", "learner_visited_corrections",
        "old_gated_holdout_correct", "new_gated_holdout_correct",
        "composed_gated_holdout_correct", "adapted_phases",
        "live_decisions")}, indent=2))


if __name__ == "__main__":
    main()
