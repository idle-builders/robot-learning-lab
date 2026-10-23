"""Small, body-independent decision-policy pilot from recorded whole tasks.

Only previously observed actor/skill results enter a decision. This is an
offline imitation screen, not a replacement for the live whole-task loop.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import confusion_matrix

import owner_route_policy as owner_route


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCES = (
    "scan30_unified_policy_report.json",
    "scan30_lighting_cross_report.json",
    "scan30_walk_range_gap_report.json",
    "scan30_same_runtime_healthy_report.json",
    "scan30_dual_rgbd_route_report.json",
    "scan30_dual_worker_healthy_report.json",
    "scan30_compound_pose_light_range_report.json",
)
OPTIONS = ("ASK_RESHOW", "WIDE_LOOK", "ASK_OWNER", "FRESH_RELOOK",
           "APPROACH", "WAIT_SOURCE", "REOBSERVE", "RESUME",
           "FINAL_REVIEW", "COMPLETE", "HOLD")


def read(path: str | Path) -> dict:
    return json.loads((ROOT / path).read_text())


def event(group: str, source: str, phase: str, features: dict,
          teacher: str) -> dict:
    assert teacher in OPTIONS
    return {"capture_group": group, "source": source, "phase": phase,
            "features": {"phase": phase, **features}, "teacher": teacher}


def confirmed(action: str | None) -> bool:
    return action == "owner_confirmed_visual_identity"


def safe_gate(option: str, features: dict) -> str:
    """A learned request cannot bypass independent, time-local evidence."""
    if features.get("phase") == "owner_route":
        return owner_route.safe_option(option, features)
    if (features.get("phase") == "wide_result" and
            "final_wide_source_healthy" in features and
            not features["final_wide_source_healthy"]):
        return "HOLD"
    if features.get("phase") == "final_review":
        if not features.get("final_source_healthy"):
            return "HOLD"
        if option == "WIDE_LOOK":
            if features.get("final_retry_exhausted"):
                return "ASK_OWNER"
            return ("WIDE_LOOK" if features.get("final_no_candidate") else
                    "HOLD")
        if option == "ASK_OWNER":
            return "ASK_OWNER"
        if option not in ("COMPLETE", "HOLD"):
            return "HOLD"
    if features.get("phase") == "route_blocked":
        return ("ASK_OWNER" if option == "ASK_OWNER" and
                features.get("source_healthy") and
                features.get("blocked_route_observed") else "HOLD")
    if features.get("moving_obstacle"):
        return "HOLD"
    if features.get("phase") == "hazard_interruption" and not \
            features.get("source_healthy"):
        return "HOLD"
    if option == "ASK_RESHOW" and "owner_safe_to_ask" in features and not \
            features["owner_safe_to_ask"]:
        return "HOLD"
    if option == "APPROACH" and not all((
        features.get("present_confirmed"),
        features.get("fresh_confirmed"),
        features.get("source_healthy"),
        features.get("pose_healthy"),
        features.get("association_healthy"),
    )):
        return "HOLD"
    if option == "RESUME" and not all((
        features.get("restored_source_healthy"),
        features.get("new_relook_confirmed"),
    )):
        return "HOLD"
    if option == "REOBSERVE" and not features.get("restored_source_healthy"):
        return "HOLD"
    if option == "COMPLETE" and not all((
        features.get("final_confirmed"), features.get("final_source_healthy"),
    )):
        return "HOLD"
    return option


def decide(saved_policy: dict, observation: dict) -> dict:
    """Issue one bounded option; the caller owns the skill and its live guard."""
    if observation.get("phase") == "owner_route":
        if "owner_route_model" not in saved_policy:
            raise ValueError("Policy has no owner-route option head")
        raw = str(saved_policy["owner_route_model"].predict(
            saved_policy["owner_route_vectorizer"].transform(
                [observation]))[0])
        return {"requested_option": raw,
                "authorized_option": safe_gate(raw, observation),
                "observation": observation}
    use_fallback = ("adapted_phases" in saved_policy and
                    observation.get("phase") not in saved_policy["adapted_phases"])
    model = saved_policy["fallback_model"] if use_fallback else saved_policy["model"]
    vectorizer = (saved_policy["fallback_vectorizer"] if use_fallback else
                  saved_policy["vectorizer"])
    raw = str(model.predict(vectorizer.transform([observation]))[0])
    return {"requested_option": raw,
            "authorized_option": safe_gate(raw, observation),
            "observation": observation}


def extract() -> tuple[list[dict], dict]:
    records: list[dict] = []
    provenance: dict = {}
    for name in SOURCES:
        path = REPORTS / name
        provenance[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = read(path.relative_to(ROOT))["rows"]
        assert len(rows) == 30, name
        for row in rows:
            group = row["capture_group"]
            source = name.replace("_report.json", "")
            base = read(row["base_summary"])
            ready = base.get("owner_status", "").startswith("teach_find_completed")
            if not ready:
                records.append(event(group, source, "owner_result",
                    {"owner_ready": False, "owner_status": base.get("owner_status", "unknown")},
                    "ASK_RESHOW" if row.get("show_summary") else "HOLD"))
                continue

            present = base.get("present_action")
            absent = base.get("absent_action")
            records.append(event(group, source, "search_result",
                {"present_confirmed": confirmed(present)},
                "FRESH_RELOOK" if confirmed(present) else "WIDE_LOOK"))
            resolved = base.get("resolved_present_action")
            if not confirmed(present):
                records.append(event(group, source, "wide_result",
                    {"wide_confirmed": confirmed(base.get("wide_present_action")),
                     "resolved_confirmed": confirmed(resolved)},
                    "FRESH_RELOOK" if confirmed(resolved) else "ASK_OWNER"))

            health = base.get("fresh_source_health") or {}
            initial_path = ROOT / base["initial_summary"]
            owner_trace = read(initial_path.parent.relative_to(ROOT) /
                               "owner_approach/trace.json")
            pose_healthy = bool(owner_trace["trace"][-1]["pose_healthy"])
            association = base.get("identity_location_association_gap_m")
            features = {"present_confirmed": confirmed(resolved),
                "fresh_confirmed": confirmed(base.get("fresh_action")),
                "source_healthy": bool(health.get("authorized")),
                "pose_healthy": pose_healthy,
                "association_healthy": association is not None and association <= .10}
            if base.get("fresh_action") is not None:
                records.append(event(group, source, "pre_motion", features,
                    "APPROACH" if base.get("motion_authorized") else "HOLD"))

            walk = base.get("walk_result")
            if walk is not None:
                records.append(event(group, source, "skill_result",
                    {"source_gap": walk == "source_gap",
                     "at_standoff": walk == "observed_target_standoff"},
                    "WAIT_SOURCE" if walk == "source_gap" else
                    "FINAL_REVIEW" if walk == "observed_target_standoff" else "HOLD"))
            recovery = row.get("range_recovery") or {}
            if recovery.get("attempted"):
                restored = bool((recovery.get("restored_health") or {}).get("authorized"))
                records.append(event(group, source, "source_restored",
                    {"restored_source_healthy": restored},
                    "REOBSERVE" if restored else "HOLD"))
                relook = recovery.get("relook_action")
                if relook is not None:
                    resume_features = {"restored_source_healthy": restored,
                        "new_relook_confirmed": confirmed(relook)}
                    records.append(event(group, source, "new_relook", resume_features,
                        "RESUME" if recovery.get("resumed") else "HOLD"))
            final = base.get("final_action")
            if final is not None:
                final_features = {"final_confirmed": confirmed(final),
                    "final_source_healthy": bool((base.get("final_source_health") or {}).get("authorized"))}
                records.append(event(group, source, "final_review", final_features,
                    "COMPLETE" if confirmed(final) else "HOLD"))
    return records, provenance


def main() -> None:
    records, provenance = extract()
    groups = sorted({r["capture_group"] for r in records})
    # The twenty original development captures teach; all ten transfer
    # captures are held out in every fault condition, including alternate runs.
    cohort = read(REPORTS.relative_to(ROOT) / SOURCES[0])["rows"]
    split = {r["capture_group"]: r["cohort"] for r in cohort}
    assert len(groups) == 30 and Counter(split.values()) == {
        "development": 20, "prior_transfer": 10}
    train = [r for r in records if split[r["capture_group"]] == "development"]
    hold = [r for r in records if split[r["capture_group"]] == "prior_transfer"]
    vectorizer = DictVectorizer(sparse=False)
    x_train = vectorizer.fit_transform([r["features"] for r in train])
    x_hold = vectorizer.transform([r["features"] for r in hold])
    model = RandomForestClassifier(n_estimators=160, max_depth=7,
                                   min_samples_leaf=2, random_state=17,
                                   class_weight="balanced_subsample")
    model.fit(x_train, [r["teacher"] for r in train])
    raw = list(model.predict(x_hold))
    gated = [safe_gate(action, record["features"])
             for action, record in zip(raw, hold)]
    truth = [r["teacher"] for r in hold]
    by_phase: dict[str, dict] = {}
    for phase in sorted({r["phase"] for r in hold}):
        indices = [i for i, r in enumerate(hold) if r["phase"] == phase]
        by_phase[phase] = {"count": len(indices),
            "raw_correct": sum(raw[i] == truth[i] for i in indices),
            "gated_correct": sum(gated[i] == truth[i] for i in indices),
            "teacher": dict(Counter(truth[i] for i in indices)),
            "predicted": dict(Counter(gated[i] for i in indices))}
    labels = sorted(set(truth) | set(gated))
    unsafe = sum(action in ("APPROACH", "RESUME", "COMPLETE") and
                 safe_gate(action, record["features"]) == "HOLD"
                 for action, record in zip(raw, hold))
    phase_majority = {phase: Counter(r["teacher"] for r in train
                      if r["phase"] == phase).most_common(1)[0][0]
                      for phase in {r["phase"] for r in train}}
    stress = defaultdict(lambda: {"cases": 0, "raw_motion_or_claim": 0,
                                  "gated_motion_or_claim": 0})
    for record in hold:
        if record["phase"] not in ("pre_motion", "new_relook", "final_review"):
            continue
        feature = record["features"]
        signals = {
            "pre_motion": ("source_healthy", "fresh_confirmed", "pose_healthy",
                           "association_healthy"),
            "new_relook": ("restored_source_healthy", "new_relook_confirmed"),
            "final_review": ("final_source_healthy", "final_confirmed"),
        }[record["phase"]]
        for signal in signals:
            if not feature.get(signal):
                continue
            perturbed = {**feature, signal: False}
            predicted = str(model.predict(vectorizer.transform([perturbed]))[0])
            checked = safe_gate(predicted, perturbed)
            item = stress[f"{record['phase']}:{signal}"]
            item["cases"] += 1
            item["raw_motion_or_claim"] += int(predicted in
                ("APPROACH", "RESUME", "COMPLETE"))
            item["gated_motion_or_claim"] += int(checked in
                ("APPROACH", "RESUME", "COMPLETE"))
    report = {"schema": "ape.high-level-event-policy-pilot.v1",
        "limits": "Offline one-step imitation only. All 30 rooms were previously exposed to the authored teacher; held-out capture groups were unseen by this model but not blind task scenarios. Every held-out pre-motion decision is a positive approach example. In missing-signal probes the raw model still asks to move or claim, so the independent gate is necessary. No learner-driven closed-loop rollout, new capture, moving person, physical sensor or body qualification.",
        "source_sha256": provenance,
        "train_capture_groups": sorted(g for g in groups if split[g] == "development"),
        "holdout_capture_groups": sorted(g for g in groups if split[g] == "prior_transfer"),
        "train_events": len(train), "holdout_events": len(hold),
        "train_options": dict(Counter(r["teacher"] for r in train)),
        "holdout_options": dict(Counter(truth)),
        "raw_correct": sum(a == t for a, t in zip(raw, truth)),
        "gated_correct": sum(a == t for a, t in zip(gated, truth)),
        "phase_majority_correct": sum(phase_majority[r["phase"]] == t
                                      for r, t in zip(hold, truth)),
        "teacher_help_requests": sum(t in ("ASK_RESHOW", "ASK_OWNER")
                                     for t in truth),
        "gated_help_requests": sum(a in ("ASK_RESHOW", "ASK_OWNER")
                                   for a in gated),
        "gated_motion_requests": sum(a in ("APPROACH", "RESUME") for a in gated),
        "unsafe_raw_requests_blocked": unsafe,
        "missing_signal_stress": dict(stress),
        "by_phase": by_phase,
        "confusion_labels": labels,
        "confusion": confusion_matrix(truth, gated, labels=labels).tolist(),
        "errors": [{"capture_group": r["capture_group"], "source": r["source"],
                    "phase": r["phase"], "teacher": t, "raw": a, "gated": g,
                    "features": r["features"]}
                   for r, t, a, g in zip(hold, truth, raw, gated) if g != t]}
    HERE.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema": report["schema"], "vectorizer": vectorizer,
                 "model": model, "options": OPTIONS}, HERE / "event_policy.joblib")
    (HERE / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("train_events", "holdout_events",
        "raw_correct", "gated_correct", "phase_majority_correct",
        "teacher_help_requests", "gated_help_requests", "gated_motion_requests",
        "unsafe_raw_requests_blocked", "missing_signal_stress", "by_phase")},
        indent=2))


if __name__ == "__main__":
    main()
