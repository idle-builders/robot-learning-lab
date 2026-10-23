"""Historical-body ray screen for V17's proposed right low-fan origin."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import dense_low_trial as dense
import flank_sensor_trial as left
import kitchen_native as kitchen
import kitchen_trial as trial
import mount_visibility as mount
import right_entry_trial as right
import run as guard


OUT = right.OUT / "right-mount-visibility.json"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    data = mujoco.MjData(model)
    rows = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for world in range(4):
            for t in mount.TIMES_S:
                pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
                origin = left.world_origin(pose, right.RIGHT_ORIGIN_LOCAL_M)
                yaw = guard.replay.yaw_from_quat(pose[3:7])
                hits = []
                for bearing in dense.YAW_DEG:
                    if bearing >= -20.:
                        continue
                    angle = yaw + math.radians(float(bearing))
                    direction = np.array([math.cos(angle), math.sin(angle), 0.])
                    geomid = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(model, data, origin, direction,
                                             mount.APE_ONLY, 1, -1, geomid)
                    if 0 <= distance <= dense.MAX_M:
                        hits.append({"bearing_deg": float(bearing),
                                     "distance_m": float(distance),
                                     "geom": model.geom(int(geomid[0])).name})
                rows.append({"world": world, "time_s": t,
                             "origin_world_m": origin.tolist(),
                             "rays": int(sum(b < -20. for b in dense.YAW_DEG)),
                             "ape_intercepts_within_range": hits})
    report = {
        "schema": "robot-learning-lab.ape-native-right-low-fan-mount-screen.v1",
        "scope": "Historical rejected Direct Frame group-4 modeled solids only; no final shell or bracket",
        "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "source_sha256": trial.sha(Path(__file__)),
        "right_mount_root_local_m": right.RIGHT_ORIGIN_LOCAL_M.tolist(),
        "max_range_m": dense.MAX_M,
        "screened_rays": sum(row["rays"] for row in rows),
        "ape_intercepts_within_range": sum(len(row["ape_intercepts_within_range"])
                                           for row in rows),
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(OUT, report["screened_rays"], report["ape_intercepts_within_range"])


if __name__ == "__main__":
    main()
