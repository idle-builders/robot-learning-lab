"""Resume the owner task only after a backed-up hazard response and fresh clearance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
import time

import numpy as np

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES, BODY_RADIUS_M
import run_joined_live_hazard_16 as original
import run_joined_backup_range_16 as backup
import run_monitored_return_stress as earlier


PRECOMMIT = ART / "joined_backup_recovery_16_precommit.json"
REPORT = ART / "joined_backup_recovery_16_report.json"
OUT = ROOT / "assets/real_indoor_scenes/joined_backup_recovery_16_20261015"
PERIOD_S = .1
BACKUP_OFFSET_S = .05


def monitor(conn, case: dict, origin_xy: list[float], yaw: float,
            presented_target_xy: list[float] | None = None) -> None:
    """Watch the held recovery pose from two separately clocked virtual sources."""
    try:
        scene, base, _, _ = original.scene_and_props(case)
        if presented_target_xy is not None:
            if case["branch"] == "direct":
                from run_cooperative_represent_probe import PersonVisibleScene
                from run_real_capture_teach_find import SCENES
                scene = PersonVisibleScene(scene,
                    SCENES[case["scene_key"]]["owner_xz"])
            prop, _, angle = base[case["target"]]
            scene.set_props({**base, case["target"]:
                (prop, tuple(presented_target_xy), angle)})
        origin_xy = np.asarray(origin_xy, float)
        eye = np.array([origin_xy[0], scene.floor_y + .42, origin_xy[1]])
        backup_angles = np.arange(-180., 180., 10., dtype=np.float32)
        baselines = (scene.range_fan(eye, yaw, owner_xz=None,
            angles_deg=ANGLES)[0], scene.range_fan(eye, yaw, owner_xz=None,
            angles_deg=backup_angles, heights_m=(.1,))[0])
        conn.send({"kind": "ready"})
        message = conn.recv()
        if message["kind"] != "start":
            raise ValueError("Recovery monitor missing clock")
        origin = float(message["origin"])
        ticks = [0, 0]
        packets = [[], []]
        while True:
            due = (ticks[0] * PERIOD_S,
                   BACKUP_OFFSET_S + ticks[1] * PERIOD_S)
            source = 0 if due[0] <= due[1] else 1
            delay = origin + due[source] - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            captured = time.perf_counter() - origin
            if captured > 90.:
                raise TimeoutError("Recovery worker exceeded monitor bound")
            if source == 0:
                rays = scene.range_fan(eye, yaw, owner_xz=None,
                    angles_deg=ANGLES)[0]
            else:
                rays = scene.range_fan(eye, yaw, owner_xz=None,
                    angles_deg=backup_angles, heights_m=(.1,))[0]
            delivered = time.perf_counter() - origin
            with np.errstate(invalid="ignore"):
                changed = bool(np.any((baselines[source] - rays > .08) &
                                      (rays < .9)))
            packets[source].append({"tick": ticks[source],
                "capture_wall_s": round(captured, 6),
                "delivery_wall_s": round(delivered, 6),
                "source_health": "healthy",
                "changed_return_alarm": changed,
                "low_range_m": [float(v) if np.isfinite(v) else None
                                for v in rays]})
            ticks[source] += 1
            if conn.poll():
                message = conn.recv()
                if message["kind"] == "stop":
                    break
        conn.send({"kind": "result", "primary_packets": packets[0],
            "backup_packets": packets[1],
            "primary_baseline_low_range_m": [float(v) if np.isfinite(v)
                else None for v in baselines[0]],
            "backup_baseline_low_range_m": [float(v) if np.isfinite(v)
                else None for v in baselines[1]]})
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def clearance(scene, position: np.ndarray, yaw: float) -> list[dict]:
    result = []
    for index in range(3):
        if index:
            time.sleep(PERIOD_S)
        capture = time.perf_counter()
        eye = np.array([position[0], scene.floor_y + .42, position[1]])
        rays = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)
        finite = rays[np.isfinite(rays)]
        nearest = float(finite.min()) if len(finite) else None
        result.append({"capture_monotonic_s": capture,
            "source_health": "healthy", "nearest_range_m": nearest,
            "clear": nearest is not None and nearest > BODY_RADIUS_M + .05})
    return result


def recovered_state(case: dict, source_trace: dict,
                    endpoint: np.ndarray) -> dict:
    state = dict(original.actor_state(case)["worker_state"])
    delta = endpoint - np.asarray(source_trace["start_xz_m"], float)
    state["evaluator_actual_xz_m"] = endpoint.tolist()
    if case["branch"] == "direct":
        actor = np.asarray(state["actor_pose_world_xz_m"], float) + delta
        state["actor_pose_world_xz_m"] = actor.tolist()
    else:
        actor = np.asarray(state["actor_pose_xy_m"], float) + delta
        state["actor_pose_xy_m"] = actor.tolist()
    return state


def run_one(case: dict) -> dict:
    key = case["scene_key"]
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    source_path = backup.OUT / key / "backup_10deg/trace.json"
    source = json.loads(source_path.read_text())
    endpoint = np.asarray(source["response_endpoint_xz_m"], float)
    scene, _, placement, target = original.scene_and_props(case)
    checked = clearance(scene, endpoint, source["heading_deg"])
    result = {"schema": "ape.joined-backup-recovery-16.v1",
        "scene_key": key, "branch": case["branch"],
        "source_interruption_trace": relative(source_path),
        "response_endpoint_xz_m": endpoint.tolist(),
        "clearance_packets": checked,
        "recovery_worker_trace": None, "recovery_worker_video": None,
        "worker_world_captures": None,
        "monitor_primary_packets": None, "monitor_backup_packets": None,
        "monitor_primary_baseline_m": None,
        "monitor_backup_baseline_m": None,
        "worker_action": None, "outcome": None,
        "limits": "Continues only the 10deg-aligned backup arm after a scripted low prop departs. Three fresh healthy stationary clearance packets precede a new owner-reviewed image worker; two virtual range sources remain active during that worker. Person, item and room are static; response endpoint is ideal command-integrated 17cm disk motion. No new moving entrant, mounted-sensor occlusion, articulated stop or real owner."}
    if source["min_evaluator_proxy_gap_m"] < .05:
        result["outcome"] = "modeled_hazard_breach_hold_unknown"
    elif not all(x["clear"] for x in checked):
        result["outcome"] = "clearance_unavailable_hold_unknown"
    else:
        state = recovered_state(case, source, endpoint)
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=monitor,
            args=(child, case, endpoint.tolist(), source["heading_deg"]))
        process.start()
        child.close()
        if not parent.poll(45.):
            process.terminate(); process.join()
            raise TimeoutError("Recovery monitor did not start")
        ready = parent.recv()
        if ready["kind"] != "ready":
            process.join()
            raise RuntimeError(f"Recovery monitor failed: {ready}")
        origin = time.perf_counter() + .2
        parent.send({"kind": "start", "origin": origin})
        wrapped = earlier.TimedStaticScene(scene, origin)
        delay = origin - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(original.run_worker, wrapped, case,
                placement, target, state, folder / "fresh_owner_check")
            worker = future.result(timeout=90.)
        worker_end = time.perf_counter() - origin
        parent.send({"kind": "stop", "worker_end_s": worker_end})
        if not parent.poll(30.):
            process.terminate(); process.join()
            raise TimeoutError("Recovery monitor did not return")
        monitored = parent.recv()
        process.join(timeout=10.)
        if process.is_alive():
            process.terminate(); process.join()
            raise TimeoutError("Recovery monitor did not exit")
        parent.close()
        if monitored["kind"] != "result":
            raise RuntimeError(f"Recovery monitor failed: {monitored}")
        alarmed = any(packet["changed_return_alarm"] for name in
            ("primary_packets", "backup_packets") for packet in monitored[name])
        result.update({"recovery_worker_trace": relative(folder /
            "fresh_owner_check/episode_trace.json"),
            "recovery_worker_video": worker["video"],
            "worker_world_captures": wrapped.captures,
            "monitor_primary_packets": monitored["primary_packets"],
            "monitor_backup_packets": monitored["backup_packets"],
            "monitor_primary_baseline_m": monitored[
                "primary_baseline_low_range_m"],
            "monitor_backup_baseline_m": monitored[
                "backup_baseline_low_range_m"],
            "worker_action": worker["action"],
            "worker_wall_s": worker_end,
            "outcome": ("changed_scene_hold_unknown" if alarmed else
                "fresh_owner_confirmed" if worker["action"] ==
                    "owner_confirmed_visual_identity" else
                "request_better_presentation")})
    path = folder / "trace.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return {"scene_key": key, "branch": case["branch"],
        "trace": relative(path), "outcome": result["outcome"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = original.case_rows()
    frozen = {"schema": "ape.joined-backup-recovery-16-precommit.v1",
        "source_backup_audit": relative(ART /
            "joined_backup_range_16_audit.json"),
        "source_backup_audit_sha256": sha(ART /
            "joined_backup_range_16_audit.json"),
        "cases": cases, "arm": "backup_10deg", "monitor_period_s": PERIOD_S,
        "clearance_packets": 3,
        "policy": "After the backed-up final-worker entrant response, discard the old image claim. At the reached endpoint require three fresh 0.1s-separated healthy near-body packets with >body-radius+5cm nearest return. Keep the original owner item memory and source pose estimate plus commanded displacement. Run a new RGB-D owner check while primary and low backup range are clocked independently at 10Hz. Confirm only on a new owner-reviewed image with no range change; otherwise request a better presentation or hold unknown.",
        "limits": "Static owner/item/room after scripted entrant departure; ideal command-integrated response endpoint and range. Not a physical body or changing-person recovery trial."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Recovery precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        path = OUT / case["scene_key"] / "trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"], "branch": case["branch"],
                "trace": relative(path), "outcome": trace["outcome"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.joined-backup-recovery-16-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["outcome"], flush=True)


if __name__ == "__main__":
    main()
