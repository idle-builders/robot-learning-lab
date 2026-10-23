"""Use rendered owner-view depth instead of an ideal upper range plane.

The existing RoboCasa owner actor and route are imported unchanged. A separate
RGB-D view is rendered at each guard tick while the mid-height owner sector is
frozen; depth is projected through commanded camera geometry to 72 world
bearings. This tests the sensing role and its wall-time cost, not hardware.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as audit  # noqa: E402
import robocasa_first_return_person_reacquire_turn_v597 as turn  # noqa: E402
import robocasa_person_interaction as interaction  # noqa: E402
import run_robocasa_first_return_person_reacquire_v577 as baseline  # noqa: E402
import run_robocasa_joined_owner_route_v410 as owner  # noqa: E402
import run_robocasa_layout2_route_v407 as route  # noqa: E402
import run_robocasa_person_joined_v497 as joined  # noqa: E402
import run_robocasa_layout2_multitrack_v485 as multi  # noqa: E402
from run_robocasa_dynamic_suite_v439 import motion  # noqa: E402

OUT = ROOT / "artifacts/ape-depth-guard-choice/v1"
RESPONSE = "early_owner_mid_sector_blackout_late_inference"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DepthSectorSensor:
    """An actor camera depth frame projected to a sparse 360-degree range packet.

    The return mask remains unknown outside the camera field. Camera rotation
    comes from the commanded kinematic head; XY is the same simulator-supplied
    virtual sensor origin used by the baseline planar scan. No object label,
    segmentation, contact, or future trajectory enters the packet.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.renderer: mujoco.Renderer | None = None
        self.model_id: int | None = None
        self.logs: list[dict] = []
        self.samples: dict[int, np.ndarray] = {}

    def scan(self, model: mujoco.MjModel, data: mujoco.MjData,
             xy: np.ndarray, max_range_m: float,
             rng: np.random.Generator) -> list[float | None]:
        start = time.perf_counter()
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            self.model_id = id(model)
        elif id(model) != self.model_id:
            raise RuntimeError("Depth renderer bound to another scene model")
        camera = model.camera(interaction.CAMERA).id
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=interaction.CAMERA)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()

        # MuJoCo depth is metric optical-axis z. Match the project's existing
        # depth_point camera convention, then use a sparse 8-pixel sample grid.
        v, u = np.mgrid[4:480:8, 4:640:8]
        z = depth[v, u]
        valid = np.isfinite(z) & (z >= .15) & (z <= max_range_m)
        focal = 480. / (2. * math.tan(math.radians(float(model.cam_fovy[camera])) / 2.))
        points_camera = np.stack(((u[valid] - 320.) * z[valid] / focal,
                                  (240. - v[valid]) * z[valid] / focal,
                                  -z[valid]), axis=1)
        rotation = data.cam_xmat[camera].reshape(3, 3)
        origin = np.array([float(xy[0]), float(xy[1]),
                           float(data.cam_xpos[camera, 2])])
        points = points_camera @ rotation.T + origin
        height = points[:, 2]
        keep = (height >= .35) & (height <= 1.65)
        delta = points[keep, :2] - xy[:2]
        distance = np.linalg.norm(delta, axis=1)
        bearing = np.arctan2(delta[:, 1], delta[:, 0])
        index = np.floor((bearing + math.pi) / (2. * math.pi) * 72. + .5).astype(int) % 72
        nearest = np.full(72, np.inf)
        np.minimum.at(nearest, index, distance)
        packet = [round(float(value), 3) if np.isfinite(value) and .14 < value < max_range_m
                  else None for value in nearest]
        number = len(self.logs)
        if number in (0, 1, 10, 20, 40):
            self.samples[number] = depth
        self.samples[-1] = depth
        self.logs.append({
            "index": number,
            "wall_timestamp_s": start,
            "render_and_project_wall_s": round(time.perf_counter() - start, 5),
            "camera_name": interaction.CAMERA,
            "camera_fovy_degrees": round(float(model.cam_fovy[camera]), 4),
            "camera_xyz_m": data.cam_xpos[camera].round(5).tolist(),
            "camera_rotation_world_from_camera": rotation.round(5).tolist(),
            "valid_depth_sample_count": int(valid.sum()),
            "height_filtered_point_count": int(keep.sum()),
            "packet_supported_beams": int(sum(value is not None for value in packet)),
            "packet_72_world_bearings_m": packet,
            "depth_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        })
        return packet

    def save(self) -> dict:
        self.folder.mkdir(parents=True, exist_ok=True)
        sample_paths = []
        for index, depth in self.samples.items():
            label = "last" if index == -1 else f"{index:03d}"
            path = self.folder / f"depth-guard-{label}.npy"
            np.save(path, depth)
            sample_paths.append({"index": index, "path": str(path.relative_to(ROOT)),
                                 "sha256": sha(path)})
        log_path = self.folder / "depth-guard-packets.json"
        log_path.write_text(json.dumps(self.logs, indent=2) + "\n")
        intervals = np.diff([row["wall_timestamp_s"] for row in self.logs])
        return {"packet_trace": str(log_path.relative_to(ROOT)),
                "packet_trace_sha256": sha(log_path),
                "sample_depth_frames": sample_paths,
                "packet_count": len(self.logs),
                "median_render_and_project_wall_s": round(float(np.median(
                    [row["render_and_project_wall_s"] for row in self.logs])), 5),
                "maximum_render_and_project_wall_s": round(float(max(
                    row["render_and_project_wall_s"] for row in self.logs)), 5),
                "maximum_packet_start_gap_wall_s": round(float(intervals.max()), 5)
                if len(intervals) else None}

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def run_case(condition: str, case: dict, prior: np.ndarray, policy: dict,
             vision: owner.Vision, exemplar: np.ndarray) -> tuple[dict, dict]:
    sensor = DepthSectorSensor(OUT / condition)
    in_worker = threading.Event()
    frozen_values: list[float | None] | None = None
    sector_center: np.ndarray | None = None
    frozen_count = 0
    original_scan = baseline.owner_scan(True)

    class GuardPool(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            in_worker.set()
            return super().submit(*args, **kwargs)

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                in_worker.clear()

    def mid_scan(model, data, xy, max_range_m, rng):
        nonlocal frozen_values, sector_center, frozen_count
        packet = original_scan(model, data, xy, max_range_m, rng)
        if condition != "moving_frozen_mid" or not in_worker.is_set():
            return packet
        if sector_center is None:
            tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
            sector_center = np.asarray(max(tracks, key=lambda row: row["hits"])["center_xy_m"])
        direction = sector_center - np.asarray(xy)
        bearing = math.atan2(direction[1], direction[0])
        angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
        delta = (angles - bearing + math.pi) % (2. * math.pi) - math.pi
        mask = np.abs(delta) <= math.radians(40.)
        if frozen_values is None:
            frozen_values = list(packet)
        frozen_count += 1
        return [frozen_values[i] if mask[i] else value for i, value in enumerate(packet)]

    old_scan, old_secondary = turn.SCAN, turn.SECONDARY_SCAN
    old_pool, old_hand = turn.ThreadPoolExecutor, interaction.SHOW_HAND.copy()
    try:
        turn.SCAN = mid_scan
        turn.SECONDARY_SCAN = sensor.scan
        turn.ThreadPoolExecutor = GuardPool
        interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
        turn.OWNER_APPROACH_M = 0.
        turn.FIRST_RETURN_OWNER_M = .4
        turn.FIRST_RETURN_OWNER_S = 1.
        turn.LATE_OWNER_APPROACH_M = .8 if condition == "moving_frozen_mid" else 0.
        turn.SECOND_RETURN_OWNER_M = 0.
        turn.DEPTH_NOISE_SIGMA_M = 0.
        turn.DEPTH_DROPOUT_FRACTION = 0.
        turn.PROP_SCRIPT = motion(case)
        turn.PACKET_OUT = OUT / condition / "owner-depth-packets.npz"
        response = (f"{RESPONSE}_frozen_sector" if condition == "moving_frozen_mid"
                    else "stationary_owner_mid_sector_blackout_healthy")
        episode = joined.episode(case, response, True, prior, policy,
                                 vision, exemplar)
        episode["depth_guard_sensor"] = sensor.save()
        episode["frozen_mid_guard_packets"] = frozen_count
        geometry = audit.audit_episode(episode)
        geometry.pop("geometry_samples", None)
        return episode, geometry
    finally:
        turn.SCAN = old_scan
        turn.SECONDARY_SCAN = old_secondary
        turn.ThreadPoolExecutor = old_pool
        interaction.SHOW_HAND = old_hand
        sensor.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    joined.OUT = OUT
    joined.owner_turn = turn.owner_turn
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    rows = []
    try:
        for condition in ("moving_frozen_mid", "stationary_healthy"):
            episode, geometry = run_case(condition, case, prior, policy, vision, exemplar)
            rows.append({"condition": condition, "episode": episode,
                         "geometry_audit": geometry})
            print(condition, episode["outcome"],
                  geometry["person_exact_nearest"]["gap_m"],
                  episode["depth_guard_sensor"]["packet_count"], flush=True)
    finally:
        route.MARGIN_M = .10
        route.LIDAR_Z_M = .30
    report = {"schema": "robot-learning-lab.depth-guard-choice.v1",
              "task": "T-hwtsgv",
              "scope": "Same furnished RoboCasa route/owner turn as V596–V598; only the secondary ideal 45 cm range plane is replaced with a rendered 640x480 owner-view depth packet projected at 8px stride into 72 world bearings. Camera/extrinsics and depth are ideal; kinematic ape, scripted moving owner, no physical stop or camera selection.",
              "projection": {"depth_min_m": .15, "depth_max_m": 5.,
                             "world_height_band_m": [.35, 1.65],
                             "pixel_stride": 8, "world_bearing_bins": 72,
                             "unknown_outside_camera_fov": True},
              "source_sha256": {"runner": sha(Path(__file__)),
                                "owner_turn": sha(Path(turn.__file__)),
                                "joined_route": sha(Path(joined.__file__)),
                                "geometry_audit": sha(Path(audit.__file__)),
                                "person_detector_weights": sha(turn.person_reacquire.DETECTOR / "model.safetensors"),
                                "config": sha(joined.CONFIG), "prior_map": sha(joined.PRIOR)},
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
