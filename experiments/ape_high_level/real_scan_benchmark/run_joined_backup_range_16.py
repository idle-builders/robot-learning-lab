"""Compare independently timed sparse/denser low range backup at final owner checks."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time

import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
import run_cross_room_inference_response as motion
import run_joined_live_hazard_16 as original


PRECOMMIT = ART / "joined_backup_range_16_precommit.json"
REPORT = ART / "joined_backup_range_16_report.json"
OUT = ROOT / "assets/real_indoor_scenes/joined_backup_range_16_20261015"
ARMS = ("backup_10deg", "backup_20deg")
PERIOD_S = .1
BACKUP_OFFSET_S = .05


def monitor(conn, case: dict, origin_xy: list[float], yaw: float,
            arm: str) -> None:
    """Two virtual range sources share geometry but have distinct packet clocks."""
    try:
        scene, base, _, _ = original.scene_and_props(case)
        hazard_prop = TexturedFloorObject(original.HAZARD_PROP, scale=.012)
        start = np.asarray(origin_xy, float)
        bearing = math.radians(yaw + case["entry_relative_deg"])
        incoming = np.array([math.sin(bearing), math.cos(bearing)])
        step = 10 if arm == "backup_10deg" else 20
        backup_angles = np.arange(-180., 180., step, dtype=np.float32)
        eye = np.array([start[0], scene.floor_y + .42, start[1]])
        primary_baseline = scene.range_fan(eye, yaw, owner_xz=None,
                                            angles_deg=ANGLES)[0]
        backup_baseline = scene.range_fan(eye, yaw, owner_xz=None,
            angles_deg=backup_angles, heights_m=(.1,))[0]
        conn.send({"kind": "ready"})
        message = conn.recv()
        if message["kind"] != "start":
            raise ValueError("Missing common clock")
        origin = float(message["origin"])
        primary_packets, backup_packets = [], []
        primary_tick = backup_tick = 0
        alarm_capture = alarm_delivery = alarm_angle = health_lost = None
        alarm_source = None
        escape = None
        while True:
            primary_due = primary_tick * PERIOD_S
            backup_due = BACKUP_OFFSET_S + backup_tick * PERIOD_S
            source = "primary" if primary_due <= backup_due else "backup"
            due = primary_due if source == "primary" else backup_due
            delay = origin + due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            captured = time.perf_counter() - origin
            if captured > 30.:
                raise TimeoutError("Backup comparison exceeded bounded worker")
            actor = motion.actor_xy(start, escape, alarm_delivery, captured)
            hazard = motion.incoming_xy(start, incoming, captured)
            scene.set_props({**base, original.HAZARD:
                (hazard_prop, tuple(hazard), 0.)})
            eye = np.array([actor[0], scene.floor_y + .42, actor[1]])
            missing = source == "primary" and (original.OUTAGE[0] <=
                captured < original.OUTAGE[1])
            if missing:
                rays = None
            elif source == "primary":
                rays = scene.range_fan(eye, yaw, owner_xz=None,
                    angles_deg=ANGLES)[0]
            else:
                rays = scene.range_fan(eye, yaw, owner_xz=None,
                    angles_deg=backup_angles, heights_m=(.1,))[0]
            delivered = time.perf_counter() - origin
            if missing and health_lost is None:
                health_lost = delivered
            baseline = primary_baseline if source == "primary" else backup_baseline
            changed = np.zeros(len(baseline), bool)
            if rays is not None:
                with np.errstate(invalid="ignore"):
                    changed = (baseline - rays > .08) & (rays < .9)
            if alarm_delivery is None and np.any(changed):
                ids = np.flatnonzero(changed)
                selected = ids[np.argmin(rays[ids])]
                alarm_capture, alarm_delivery = captured, delivered
                alarm_angle = float((ANGLES if source == "primary"
                    else backup_angles)[selected])
                alarm_source = source + "_range"
                opposite = math.radians(yaw + alarm_angle + 180.)
                escape = np.array([math.sin(opposite), math.cos(opposite)])
            packet = {"tick": primary_tick if source == "primary" else backup_tick,
                "capture_wall_s": round(captured, 6),
                "delivery_wall_s": round(delivered, 6),
                "source_health": "missing" if missing else "healthy",
                "actor_xz_m": actor.tolist(), "hazard_xz_m": hazard.tolist(),
                "range_change_alarm": bool(np.any(changed)),
                "low_range_m": ([float(v) if np.isfinite(v) else None
                    for v in rays] if rays is not None else None)}
            if source == "primary":
                primary_packets.append(packet)
                primary_tick += 1
            else:
                backup_packets.append(packet)
                backup_tick += 1
            if conn.poll():
                message = conn.recv()
                if message["kind"] == "stop":
                    stop_s = float(message["worker_end_s"])
                    break
        endpoint = motion.actor_xy(start, escape, alarm_delivery, stop_s)
        conn.send({"kind": "result", "packets": primary_packets,
            "backup_packets": backup_packets,
            "baseline_low_range_m": [float(v) if np.isfinite(v) else None
                for v in primary_baseline],
            "backup_baseline_low_range_m": [float(v) if np.isfinite(v) else None
                for v in backup_baseline],
            "alarm_capture_s": alarm_capture,
            "alarm_delivery_s": alarm_delivery,
            "alarm_relative_angle_deg": alarm_angle,
            "alarm_source": alarm_source,
            "source_health_lost_s": health_lost,
            "response_endpoint_xz_m": endpoint.tolist()})
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = original.case_rows()
    radius = TexturedFloorObject(original.HAZARD_PROP,
                                 scale=.012).footprint_radius_m
    frozen = {"schema": "ape.joined-backup-range-16-precommit.v1",
        "source_audit": relative(ART / "joined_live_hazard_16_audit.json"),
        "source_audit_sha256": sha(ART / "joined_live_hazard_16_audit.json"),
        "cases": cases, "arms": list(ARMS),
        "primary_outage_s": original.OUTAGE,
        "primary_range": "two heights (0.1/0.5m), 360deg at 5deg, 10Hz",
        "backup_range": "one low 0.1m plane, 360deg at 10deg or 20deg, 10Hz offset 0.05s",
        "policy": "Same actor-admitted corridor and timed entrant as the prior 16-case final image-worker trial. The primary source is absent from 0.25-1.25s. An independently timed virtual backup low ring samples throughout; first changed return from either source commands the same delayed opposite-bearing ideal proxy response. Source loss or alarm invalidates in-flight visual identity. Compare 10deg and 20deg angular sampling without changing the owner worker.",
        "screening_disclosure": "A development-room static geometry screen at 5/10/20deg preceded this live run: 10deg saw all 16 entrants by 0.4s; 20deg did not see many side entrants until 1.2-1.3s. These arms are development-exposed, not a blind sensor-selection test.",
        "limits": "Both virtual sources share ideal center geometry and one spawned supervisor process; this tests independent packet clocks and primary-source outage, not physical common-mode failure. Scripted rigid prop, ideal disk response, no articulated stop/owner-task return."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Backup precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"scene_key": case["scene_key"], "branch": case["branch"],
                    "arm": arm, "trace": relative(path),
                    "min_gap_m": trace["min_evaluator_proxy_gap_m"],
                    "alarm_source": trace["alarm_source"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = original.run_one(case, arm, radius, sensor_target=monitor,
                    out_root=OUT, trace_schema="ape.joined-backup-range-16.v1")
                trace = json.loads((ROOT / row["trace"]).read_text())
                row["alarm_source"] = trace["alarm_source"]
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.joined-backup-range-16-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(case["scene_key"], arm, round(row["min_gap_m"], 3),
                  row["alarm_source"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
