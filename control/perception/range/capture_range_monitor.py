"""Independent, timestamped room-range stream for a captured owner task.

The source casts its own two-height fan at 10 Hz while the image worker and
walk loop run. A pose command is acknowledged by sequence number in a later
capture, so an action cannot rely on a packet from its previous pose.
"""

from __future__ import annotations

import multiprocessing as mp
import math
import threading
import time

import numpy as np


ANGLES = np.arange(-180., 180., 5., dtype=np.float32)
PERIOD_S = .1
MAX_AGE_S = .25
MAX_INTERVAL_S = .2
BODY_RADIUS_M = .17
CLEARANCE_MARGIN_M = .05


def coverage_mask(coverage: str) -> np.ndarray:
    if coverage == "all360":
        return np.ones_like(ANGLES, dtype=bool)
    if coverage == "front140":
        return np.abs(ANGLES) <= 70.
    raise ValueError("Unsupported range-monitor coverage")


def moving_person_xz(start: np.ndarray, heading_deg: float,
                     entry_relative_deg: float, elapsed_s: float) -> np.ndarray:
    """One bounded approach and retreat, shared by RGB-D and range workers."""
    angle = math.radians(heading_deg + entry_relative_deg)
    direction = np.array([math.sin(angle), math.cos(angle)])
    if elapsed_s < .2:
        distance = 1.3
    elif elapsed_s < 2.2:
        distance = 1.3 - .65 * (elapsed_s - .2) / 2.
    elif elapsed_s < 3.2:
        distance = .65
    elif elapsed_s < 5.2:
        distance = .65 + .65 * (elapsed_s - 3.2) / 2.
    else:
        distance = 1.3
    return start + distance * direction


def source_process(conn, case: dict, position: list[float],
                   heading_deg: float, coverage: str,
                   props_visible_at_start: bool) -> None:
    try:
        from run_compound_owner_point_recovery import make_scene

        scene = make_scene(case, True, range_only=True)
        saved_props = scene.props.copy()
        props_visible = props_visible_at_start
        if not props_visible:
            scene.props = {}
        visible = coverage_mask(coverage)
        xz = np.asarray(position, float)
        yaw = float(heading_deg)
        pose_seq = 0
        moving = None
        stationary_person = None
        static_low = None

        def cast(owner_xz=None) -> np.ndarray:
            eye = np.array([xz[0], scene.floor_y + .42, xz[1]])
            return scene.range_fan(eye, yaw, owner_xz=owner_xz,
                                   angles_deg=ANGLES)

        cast()  # Warm ray acceleration before the timed source starts.
        conn.send({"kind": "ready"})
        if conn.recv().get("kind") != "start":
            raise ValueError("Range source expected start")
        next_due = time.perf_counter()
        gap_until = 0.
        index = 0
        while True:
            while conn.poll():
                command = conn.recv()
                kind = command.get("kind")
                if kind == "stop":
                    conn.send({"kind": "stopped"})
                    return
                if kind == "pose":
                    xz = np.asarray(command["position"], float)
                    yaw = float(command["heading_deg"])
                    pose_seq = int(command["pose_seq"])
                    # Compare a moving return with the static room at the
                    # *same* pose. The previous background belongs to the
                    # previous pose and otherwise treats walls as entrants.
                    if moving is not None:
                        static_low = cast()[0]
                elif kind == "gap":
                    gap_until = max(gap_until,
                                    time.perf_counter() + float(command["seconds"]))
                elif kind == "moving_person":
                    moving = command
                    static_low = cast()[0]
                elif kind == "stationary_person":
                    stationary_person = tuple(command["position_xz_m"])
                elif kind == "clear_stationary_person":
                    stationary_person = None
                elif kind == "props_visible":
                    props_visible = bool(command["visible"])
                    scene.props = saved_props.copy() if props_visible else {}
                else:
                    raise ValueError(f"Unknown range command: {kind}")
            now = time.perf_counter()
            if now < next_due:
                time.sleep(min(.01, next_due - now))
                continue
            captured = time.perf_counter()
            missing = captured < gap_until
            person = (moving_person_xz(np.asarray(moving["start_xz_m"]),
                float(moving["heading_deg"]),
                float(moving["entry_relative_deg"]),
                captured - float(moving["origin_wall_s"]))
                if moving is not None and captured >= moving["origin_wall_s"]
                else None)
            ranges = None if missing else cast(
                tuple(person) if person is not None else stationary_person)
            sensed = None if ranges is None else ranges.copy()
            if sensed is not None:
                sensed[:, ~visible] = np.nan
            delivered = time.perf_counter()
            finite = sensed[np.isfinite(sensed)] if sensed is not None else []
            alarm = False
            alarm_angles = []
            if person is not None and ranges is not None:
                with np.errstate(invalid="ignore"):
                    changed = ((static_low - ranges[0] > .08) &
                               (ranges[0] < .9))
                    alarm_angles = ANGLES[changed & visible].tolist()
                    alarm = bool(alarm_angles)
            conn.send({"kind": "packet", "index": index,
                "capture_wall_s": captured, "delivery_wall_s": delivered,
                "source_health": "missing" if missing else "healthy",
                "pose_seq": pose_seq, "position_xz_m": xz.tolist(),
                "heading_deg": yaw,
                "nearest_m": float(np.min(finite)) if len(finite) else None,
                "moving_person_xz_m": person.tolist() if person is not None else None,
                "stationary_person_xz_m": (list(stationary_person)
                    if stationary_person is not None else None),
                "props_visible": props_visible,
                "near_body_change_alarm": alarm,
                "monitor_coverage": coverage,
                "alarm_angles_deg": alarm_angles,
                "low_m": ([float(v) if np.isfinite(v) else None
                           for v in sensed[0]] if sensed is not None else None),
                "mid_m": ([float(v) if np.isfinite(v) else None
                           for v in sensed[1]] if sensed is not None else None),
                "evaluator_all_angle_low_m": ([float(v) if np.isfinite(v)
                           else None for v in ranges[0]]
                           if ranges is not None else None)})
            index += 1
            next_due += PERIOD_S
            if next_due < delivered:
                next_due = delivered
    except BaseException as exc:
        try:
            conn.send({"kind": "error", "error": repr(exc)})
        except BaseException:
            pass
    finally:
        conn.close()


class CaptureRangeMonitor:
    def __init__(self, case: dict, position: np.ndarray,
                 heading_deg: float, coverage: str = "all360",
                 props_visible_at_start: bool = True) -> None:
        coverage_mask(coverage)
        context = mp.get_context("spawn")
        self.parent, child = context.Pipe()
        self.process = context.Process(target=source_process,
            args=(child, case, position.tolist(), float(heading_deg), coverage,
                  props_visible_at_start))
        self.packets: list[dict] = []
        self.error: str | None = None
        self.pose_seq = 0
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.process.start()
        child.close()
        self.reader.start()
        if not self.ready.wait(120.):
            self.close()
            raise TimeoutError("Independent room-range source did not become ready")
        if self.error:
            self.close()
            raise RuntimeError(self.error)
        self.parent.send({"kind": "start"})
        deadline = time.perf_counter() + 3.
        while time.perf_counter() < deadline:
            with self.lock:
                if self.packets:
                    break
                error = self.error
            if error:
                self.close()
                raise RuntimeError(error)
            time.sleep(.01)
        else:
            self.close()
            raise TimeoutError("Independent room-range source emitted no packet")

    def _read(self) -> None:
        try:
            while True:
                message = self.parent.recv()
                kind = message["kind"]
                with self.lock:
                    if kind == "packet":
                        self.packets.append(message)
                    elif kind == "error":
                        self.error = message["error"]
                        self.ready.set()
                        return
                    elif kind == "ready":
                        self.ready.set()
                    elif kind == "stopped":
                        self.stopped.set()
                        return
        except EOFError:
            with self.lock:
                if not self.stopped.is_set():
                    self.error = "Range source closed before stop acknowledgement"
                    self.ready.set()

    @staticmethod
    def mark() -> float:
        return time.perf_counter()

    def update_pose(self, position: np.ndarray, heading_deg: float) -> None:
        self.pose_seq += 1
        self.parent.send({"kind": "pose", "position": position.tolist(),
            "heading_deg": float(heading_deg), "pose_seq": self.pose_seq})

    def inject_gap(self, seconds: float = 1.) -> float:
        started = time.perf_counter()
        self.parent.send({"kind": "gap", "seconds": float(seconds)})
        return started

    def start_moving_person(self, position: np.ndarray,
                            heading_deg: float,
                            entry_relative_deg: float) -> float:
        origin = time.perf_counter() + .25
        self.parent.send({"kind": "moving_person",
            "start_xz_m": position.tolist(),
            "heading_deg": float(heading_deg),
            "entry_relative_deg": float(entry_relative_deg),
            "origin_wall_s": origin})
        return origin

    def place_stationary_person(self, position: np.ndarray) -> float:
        started = time.perf_counter()
        self.parent.send({"kind": "stationary_person",
                          "position_xz_m": position.tolist()})
        return started

    def wait_for_stationary_person(self, position: np.ndarray,
                                   since_wall_s: float,
                                   timeout_s: float = 1.) -> bool:
        deadline = time.perf_counter() + timeout_s
        expected = position.tolist()
        while time.perf_counter() < deadline:
            with self.lock:
                observed = any(p["capture_wall_s"] >= since_wall_s and
                    p.get("stationary_person_xz_m") == expected
                    for p in self.packets)
            if observed:
                return True
            time.sleep(.01)
        return False

    def clear_stationary_person(self) -> float:
        started = time.perf_counter()
        self.parent.send({"kind": "clear_stationary_person"})
        return started

    def set_props_visible(self, visible: bool) -> float:
        started = time.perf_counter()
        self.parent.send({"kind": "props_visible", "visible": bool(visible)})
        return started

    def wait_for_scene_state(self, since_wall_s: float, *,
                             props_visible: bool,
                             stationary_person: bool,
                             timeout_s: float = 1.) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self.lock:
                observed = any(p["capture_wall_s"] >= since_wall_s and
                    p["props_visible"] == props_visible and
                    (p["stationary_person_xz_m"] is not None) ==
                    stationary_person for p in self.packets)
            if observed:
                return True
            time.sleep(.01)
        return False

    def wait_for_stationary_person_clear(self, since_wall_s: float,
                                          timeout_s: float = 1.) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self.lock:
                observed = any(p["capture_wall_s"] >= since_wall_s and
                    p.get("stationary_person_xz_m") is None
                    for p in self.packets)
            if observed:
                return True
            time.sleep(.01)
        return False

    def observed_gap(self, since_wall_s: float) -> bool:
        with self.lock:
            return any(p["capture_wall_s"] >= since_wall_s and
                       p["source_health"] == "missing" for p in self.packets)

    def wait_for_gap(self, since_wall_s: float, timeout_s: float = .4) -> bool:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if self.observed_gap(since_wall_s):
                return True
            time.sleep(.01)
        return self.observed_gap(since_wall_s)

    def health(self, since_wall_s: float) -> dict:
        checked = time.perf_counter()
        with self.lock:
            packets = [p for p in self.packets if
                       p["capture_wall_s"] >= since_wall_s and
                       p["delivery_wall_s"] <= checked]
            error = self.error
        intervals = [b["capture_wall_s"] - a["capture_wall_s"]
                     for a, b in zip(packets, packets[1:])]
        last = packets[-1] if packets else None
        age = checked - last["delivery_wall_s"] if last else None
        gap = any(p["source_health"] != "healthy" for p in packets)
        alarm = any(p.get("near_body_change_alarm", False) for p in packets)
        maximum = max(intervals, default=0.)
        current_pose = last is not None and last["pose_seq"] == self.pose_seq
        nearest = last["nearest_m"] if last else None
        authorized = bool(not error and packets and not gap and not alarm and current_pose and
            maximum <= MAX_INTERVAL_S and age is not None and
            0 <= age <= MAX_AGE_S and nearest is not None and
            nearest > BODY_RADIUS_M + CLEARANCE_MARGIN_M)
        return {"authorized": authorized,
            "reason": ("source_error" if error else "no_packet" if not packets
                       else "source_gap" if gap else "moving_obstacle" if alarm
                       else "old_pose_packet"
                       if not current_pose else "slow_cadence"
                       if maximum > MAX_INTERVAL_S else "stale_packet"
                       if age is None or age > MAX_AGE_S else
                       "range_unavailable" if nearest is None else
                       "near_body_blocked" if nearest <= BODY_RADIUS_M +
                       CLEARANCE_MARGIN_M else
                       "healthy"),
            "checked_wall_s": checked, "packet_count": len(packets),
            "max_capture_interval_s": maximum, "last_packet_age_s": age,
            "source_gap_seen": gap, "pose_seq": self.pose_seq,
            "moving_obstacle_seen": alarm,
            "last_packet_pose_seq": last["pose_seq"] if last else None,
            "nearest_m": nearest, "error": error}

    def wait_health(self, since_wall_s: float,
                    timeout_s: float = .35) -> dict:
        deadline = time.perf_counter() + timeout_s
        while True:
            state = self.health(since_wall_s)
            if state["authorized"] or state["reason"] not in (
                    "no_packet", "old_pose_packet") or \
                    time.perf_counter() >= deadline:
                return state
            time.sleep(.01)

    def close(self) -> list[dict]:
        if self.process.is_alive():
            self.parent.send({"kind": "stop"})
            self.stopped.wait(3.)
        self.process.join(timeout=5.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5.)
        with self.lock:
            packets = list(self.packets)
        self.parent.close()
        return packets
