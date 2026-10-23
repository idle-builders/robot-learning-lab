"""Fit an owner-stage option head from captured, actor-visible route states."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

import owner_route_policy as owner
import train_event_policy as events

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE_POLICY = HERE / "event_policy_live_v4_blocked.joblib"
OUTPUT_POLICY = HERE / "event_policy_live_v6_owner_route.joblib"
OUTPUT_REPORT = HERE / "owner_route_v6_report.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    provenance = {}
    for name in events.SOURCES:
        report = events.REPORTS / name
        provenance[name] = digest(report)
        for case in json.loads(report.read_text())["rows"]:
            summary = ROOT / case["base_summary"]
            trace = summary.parent / "initial/owner_approach/trace.json"
            if not trace.is_file():
                raise FileNotFoundError(trace)
            frames = json.loads(trace.read_text())["trace"]
            for frame in frames:
                intent = frame["planner_decision"]["intent"]
                if intent == "hold_request_help":
                    continue  # The route skill's supported failure remains a stop.
                if intent not in ("step_to_waypoint", "turn_to_waypoint",
                                  "turn_to_owner", "hold_viewpoint",
                                  "hold_viewpoint_far"):
                    raise ValueError(f"Unexpected owner-route decision: {intent}")
                signal = owner.features(frame["decision_owner_actor"],
                    pose_healthy=frame["pose_healthy"],
                    known_low_beams=frame["known_low_beam_count"],
                    known_mid_beams=frame["known_mid_beam_count"],
                    source_healthy=True)
                label = (owner.TEACH if intent in ("hold_viewpoint",
                         "hold_viewpoint_far") else owner.CONTINUE)
                rows.append({"capture_group": case["capture_group"],
                    "cohort": case["cohort"], "source": name,
                    "frame": frame["frame"], "features": signal,
                    "teacher": label, "planner_intent": intent})
    return rows, provenance


def main() -> None:
    rows, provenance = records()
    train = [row for row in rows if row["cohort"] == "development"]
    hold = [row for row in rows if row["cohort"] == "prior_transfer"]
    assert len({r["capture_group"] for r in train}) == 20
    assert len({r["capture_group"] for r in hold}) == 10
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform([r["features"] for r in train])
    x_hold = vectorizer.transform([r["features"] for r in hold])
    model = RandomForestClassifier(n_estimators=160, max_depth=7,
        min_samples_leaf=3, class_weight="balanced_subsample", random_state=29)
    model.fit(x_train, [r["teacher"] for r in train])
    raw = [str(x) for x in model.predict(x_hold)]
    authorized = [owner.safe_option(x, row["features"])
                  for x, row in zip(raw, hold)]
    truth = [r["teacher"] for r in hold]
    teach_indices = [i for i, label in enumerate(truth) if label == owner.TEACH]
    false_teach = [i for i, (pred, label) in enumerate(zip(authorized, truth))
                   if pred == owner.TEACH and label != owner.TEACH]
    missed_teach = [i for i in teach_indices if authorized[i] != owner.TEACH]
    base = joblib.load(BASE_POLICY)
    saved = {**base, "owner_route_model": model,
        "owner_route_vectorizer": vectorizer,
        "owner_route_schema": "ape.owner-route-option-head.v1"}
    joblib.dump(saved, OUTPUT_POLICY)
    report = {"schema": "ape.owner-route-option-imitation.v1",
        "base_policy": str(BASE_POLICY.relative_to(ROOT)),
        "base_policy_sha256": digest(BASE_POLICY),
        "source_reports_sha256": provenance,
        "train_capture_groups": sorted({r["capture_group"] for r in train}),
        "holdout_capture_groups": sorted({r["capture_group"] for r in hold}),
        "train_frames": len(train), "holdout_frames": len(hold),
        "train_labels": dict(Counter(r["teacher"] for r in train)),
        "holdout_labels": dict(Counter(truth)),
        "holdout_raw_correct": sum(a == b for a, b in zip(raw, truth)),
        "holdout_authorized_correct": sum(a == b for a, b in
                                          zip(authorized, truth)),
        "holdout_teach_recall": (len(teach_indices) - len(missed_teach)) /
            len(teach_indices),
        "holdout_false_teach": len(false_teach),
        "holdout_missed_teach": len(missed_teach),
        "errors": [{"capture_group": hold[i]["capture_group"],
            "source": hold[i]["source"], "frame": hold[i]["frame"],
            "features": hold[i]["features"], "teacher": hold[i]["teacher"],
            "raw": raw[i], "authorized": authorized[i]}
            for i in sorted(set(false_teach + missed_teach))],
        "limits": "Offline imitation of an authored owner route, not a live learned rollout. The 20/10 split is by capture group but all rooms were exposed earlier in this project. Historical traces have no separate owner-stage monitor, so source_healthy is a training assumption; live use must supply a real packet gate. Only continue-versus-teach is learned. Bounded turns/steps, safe route failure and teaching itself remain skills. A classifier score does not establish task robustness, sensor choice or body qualification."}
    OUTPUT_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in (
        "train_frames", "holdout_frames", "train_labels", "holdout_labels",
        "holdout_raw_correct", "holdout_authorized_correct",
        "holdout_teach_recall", "holdout_false_teach",
        "holdout_missed_teach")}, indent=2))


if __name__ == "__main__":
    main()
