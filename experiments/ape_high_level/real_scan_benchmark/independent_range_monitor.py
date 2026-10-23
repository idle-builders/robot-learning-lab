"""Separately clocked two-height range source for a whole owner task.

The worker owns a ray scene and samples at 10 Hz even while RGB inference is
running. Source gaps are fault injection in this simulator, not detected real
hardware faults. The actor only sees delivered packets and their timestamps.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

ANGLES = np.arange(-90., 91., 5., dtype=np.float32)
PERIOD_S = .1
MAX_AGE_S = .2
MAX_INTERVAL_S = .2
BODY_RADIUS_M = .17


def source_process(conn, converted_mesh: str, position: tuple[float, float, float],
                   heading_deg: float) -> None:
    try:
        mesh = trimesh.load(Path(converted_mesh), process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Range source requires one converted triangle mesh")
        static = o3d.t.geometry.RaycastingScene()
        static.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)),
                             o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
        x, base_y, z = position
        yaw_deg = heading_deg

        def cast() -> np.ndarray:
            rays = []
            for height in (.1, .5):
                for offset in ANGLES:
                    yaw = math.radians(yaw_deg + float(offset))
                    rays.append([x, base_y + height, z,
                                 math.sin(yaw), 0., math.cos(yaw)])
            packet = o3d.core.Tensor(np.asarray(rays, np.float32))
            return static.cast_rays(packet)["t_hit"].numpy().reshape(2, -1)

        cast()  # Initialize the accelerator before releasing the image worker.
        conn.send({"kind": "ready"})
        command = conn.recv()
        if command.get("kind") != "start":
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
                    x, base_y, z = command["position"]
                    yaw_deg = float(command["heading_deg"])
                elif kind == "gap":
                    gap_until = max(gap_until,
                                    time.perf_counter() + float(command["seconds"]))
                else:
                    raise ValueError(f"Unknown range command: {kind}")
            now = time.perf_counter()
            if now < next_due:
                time.sleep(min(.01, next_due - now))
                continue
            captured = time.perf_counter()
            missing = captured < gap_until
            ranges = None if missing else cast()
            delivered = time.perf_counter()
            finite = ranges[np.isfinite(ranges)] if ranges is not None else []
            conn.send({"kind": "packet", "index": index,
                "capture_wall_s": captured, "delivery_wall_s": delivered,
                "source_health": "missing" if missing else "healthy",
                "nearest_m": float(np.min(finite)) if len(finite) else None,
                "low_m": ([float(v) if np.isfinite(v) else None
                           for v in ranges[0]] if ranges is not None else None),
                "mid_m": ([float(v) if np.isfinite(v) else None
                           for v in ranges[1]] if ranges is not None else None)})
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


class IndependentRangeMonitor:
    def __init__(self, converted_mesh: Path, position: np.ndarray,
                 base_y: float, heading_deg: float):
        self.context = mp.get_context("spawn")
        self.parent, child = self.context.Pipe()
        self.process = self.context.Process(target=source_process,
            args=(child, str(converted_mesh),
                  (float(position[0]), float(base_y), float(position[1])),
                  float(heading_deg)))
        self.packets: list[dict] = []
        self._closed = False
        self.error: str | None = None
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.process.start()
        child.close()
        self.reader.start()
        if not self.ready.wait(45.):
            self.close()
            raise TimeoutError("Independent range source did not become ready")
        if self.error:
            self.close()
            raise RuntimeError(self.error)
        self.started_wall_s = time.perf_counter()
        self.parent.send({"kind": "start"})
        deadline = self.started_wall_s + 3.
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
            raise TimeoutError("Independent range source emitted no first packet")

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

    def mark(self) -> float:
        return time.perf_counter()

    def update_pose(self, position: np.ndarray, base_y: float,
                    heading_deg: float) -> None:
        self.parent.send({"kind": "pose", "position": [
            float(position[0]), float(base_y), float(position[1])],
            "heading_deg": float(heading_deg)})

    def inject_gap(self, seconds: float = 1.) -> None:
        self.parent.send({"kind": "gap", "seconds": float(seconds)})

    def health(self, *, since_wall_s: float) -> dict:
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
        source_gap = any(p["source_health"] != "healthy" for p in packets)
        maximum = max(intervals, default=0.)
        nearest = last["nearest_m"] if last else None
        authorized = bool(not error and packets and not source_gap and
            maximum <= MAX_INTERVAL_S and age is not None and
            0 <= age <= MAX_AGE_S and nearest is not None and
            nearest > BODY_RADIUS_M)
        return {"authorized": authorized,
            "reason": ("source_error" if error else "no_packet" if not packets
                       else "source_gap" if source_gap
                       else "slow_cadence" if maximum > MAX_INTERVAL_S
                       else "stale_packet" if age is None or age > MAX_AGE_S
                       else "range_unavailable" if nearest is None
                       else "near_body_blocked" if nearest <= BODY_RADIUS_M
                       else "healthy"),
            "checked_wall_s": checked,
            "packet_count": len(packets),
            "max_capture_interval_s": maximum,
            "last_packet_age_s": age,
            "source_gap_seen": source_gap,
            "nearest_m": nearest,
            "error": error}

    def close(self) -> list[dict]:
        if self._closed:
            with self.lock:
                return list(self.packets)
        self._closed = True
        if self.process.is_alive():
            self.parent.send({"kind": "stop"})
            self.stopped.wait(5.)
            if self.process.is_alive():
                self.process.terminate()
        self.process.join(timeout=5.)
        self.reader.join(timeout=1.)
        self.parent.close()
        with self.lock:
            return list(self.packets)
