"""Replay post-interruption clearance, live source health and fresh owner claims."""

from __future__ import annotations

import json
import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_joined_backup_recovery_16 as trial
import run_joined_backup_range_16 as backup
import run_joined_live_hazard_16 as original
import audit_shared_world_return as video_audit


OUT = ART / "joined_backup_recovery_16_audit.json"


def source_packets(packets: list[dict], baseline: list) -> float:
    if len(packets) < 2:
        raise ValueError("Missing live recovery range packets")
    base = np.asarray([np.inf if x is None else x for x in baseline], float)
    for index, packet in enumerate(packets):
        if packet["tick"] != index or packet["source_health"] != "healthy" or (
                packet["delivery_wall_s"] < packet["capture_wall_s"]):
            raise ValueError("Recovery packet timing/health changed")
        if index and packets[index - 1]["capture_wall_s"] >= packet[
                "capture_wall_s"]:
            raise ValueError("Recovery packet order changed")
        rays = np.asarray([np.inf if x is None else x for x in
                           packet["low_range_m"]], float)
        if len(rays) != len(base):
            raise ValueError("Recovery source resolution changed")
        with np.errstate(invalid="ignore"):
            changed = bool(np.any((base - rays > .08) & (rays < .9)))
        if changed != packet["changed_return_alarm"]:
            raise ValueError("Recovery alarm cannot be replayed")
    return max(b["capture_wall_s"] - a["capture_wall_s"]
               for a, b in zip(packets, packets[1:]))


def audit_one(row: dict, case: dict) -> dict:
    x = json.loads((ROOT / row["trace"]).read_text())
    source_path = backup.OUT / row["scene_key"] / "backup_10deg/trace.json"
    source = json.loads(source_path.read_text())
    if x["source_interruption_trace"] != relative(source_path) or not np.allclose(
            x["response_endpoint_xz_m"], source["response_endpoint_xz_m"]):
        raise ValueError("Recovery endpoint/source changed")
    scene, _, _, _ = original.scene_and_props(case)
    endpoint = np.asarray(x["response_endpoint_xz_m"], float)
    floor = scene.floor_supported(*endpoint)
    clearance = x["clearance_packets"]
    if len(clearance) != 3 or any(p["source_health"] != "healthy" or
            p["nearest_range_m"] is None or p["clear"] != (
            p["nearest_range_m"] > .22) for p in clearance):
        raise ValueError("Fresh clearance gate differs")
    if any(b["capture_monotonic_s"] - a["capture_monotonic_s"] < .09
           for a, b in zip(clearance, clearance[1:])):
        raise ValueError("Clearance packets not independently spaced")
    if x["outcome"] in ("modeled_hazard_breach_hold_unknown",
                         "clearance_unavailable_hold_unknown"):
        if x["recovery_worker_trace"] is not None:
            raise ValueError("Worker ran after unsupported clearance")
        return {"scene_key": row["scene_key"], "branch": row["branch"],
            "outcome": x["outcome"], "floor_supported": floor,
            "max_packet_interval_s": None}
    cadence = max(source_packets(x["monitor_primary_packets"],
        x["monitor_primary_baseline_m"]), source_packets(x[
        "monitor_backup_packets"], x["monitor_backup_baseline_m"]))
    if not x["worker_world_captures"] or not any(c["channel"] == "rgbd"
            for c in x["worker_world_captures"]):
        raise ValueError("Recovery image worker did not capture RGB-D")
    episode = json.loads((ROOT / x["recovery_worker_trace"]).read_text())
    if episode["action"] != x["worker_action"] or (
            video_audit.decoded_frames(ROOT / x["recovery_worker_video"]) !=
            len(episode["video_frames"])):
        raise ValueError("Recovery worker decision/video differs")
    alarmed = any(p["changed_return_alarm"] for packets in (
        x["monitor_primary_packets"], x["monitor_backup_packets"])
        for p in packets)
    expected = ("changed_scene_hold_unknown" if alarmed else
        "fresh_owner_confirmed" if x["worker_action"] ==
        "owner_confirmed_visual_identity" else "request_better_presentation")
    if x["outcome"] != expected or not floor:
        raise ValueError("Recovery task outcome or floor differs")
    return {"scene_key": row["scene_key"], "branch": row["branch"],
        "outcome": x["outcome"], "floor_supported": floor,
        "worker_action": x["worker_action"],
        "worker_first_capture_s": next(c["capture_wall_s"] for c in
            x["worker_world_captures"] if c["channel"] == "rgbd"),
        "max_packet_interval_s": cadence,
        "recovery_video": x["recovery_worker_video"]}


def main() -> None:
    report = json.loads(trial.REPORT.read_text())
    frozen = json.loads(trial.PRECOMMIT.read_text())
    if len(report["rows"]) != 16 or frozen[
            "source_backup_audit_sha256"] != sha(ART /
            "joined_backup_range_16_audit.json"):
        raise ValueError("Recovery cohort incomplete/changed")
    cases = {c["scene_key"]: c for c in frozen["cases"]}
    if {r["scene_key"] for r in report["rows"]} != set(cases):
        raise ValueError("Recovery cases missing/repeated")
    rows = [audit_one(r, cases[r["scene_key"]]) for r in report["rows"]]
    result = {"schema": "ape.joined-backup-recovery-16-audit.v1",
        "report": relative(trial.REPORT), "runs": len(rows),
        "fresh_owner_confirmed": sum(r["outcome"] ==
            "fresh_owner_confirmed" for r in rows),
        "request_better_presentation": sum(r["outcome"] ==
            "request_better_presentation" for r in rows),
        "changed_scene_or_clearance_hold": sum(r["outcome"] not in (
            "fresh_owner_confirmed", "request_better_presentation") for r in rows),
        "floor_unsupported": sum(not r["floor_supported"] for r in rows),
        "max_packet_interval_s": max(r["max_packet_interval_s"] for r in rows
            if r["max_packet_interval_s"] is not None),
        "rows": rows, "limits": frozen["limits"]}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
