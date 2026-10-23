"""Ray screen of the V15 low-fan mounts against the historical ape solids.

This checks only represented group-4 geometry at saved gait poses. The final
shell, cables, brackets and sensor optical package are absent from this model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import dense_low_trial as dense
import flank_sensor_trial as flank
import kitchen_native as kitchen
import kitchen_trial as trial
import run as guard


OUT = flank.OUT / "mount-visibility.json"
TIMES_S = (.02, .22, .30, 1., 2., 3.)
APE_ONLY = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(kitchen.OUT / "scene.xml"))
    data = mujoco.MjData(model)
    rows = []
    with np.load(kitchen.TRACE, allow_pickle=False) as saved:
        for world in range(4):
            for t in TIMES_S:
                pose = kitchen.put_native_pose(model, data, saved, world, t, 3.)
                yaw = guard.replay.yaw_from_quat(pose[3:7])
                for source, local, use in (
                    ("front_low", flank.FRONT_ORIGIN_LOCAL_M,
                     lambda bearing: bearing <= 20.),
                    ("left_flank_low", flank.FLANK_ORIGIN_LOCAL_M,
                     lambda bearing: bearing > 20.),
                ):
                    origin = flank.world_origin(pose, local)
                    hits = []
                    for bearing in dense.YAW_DEG:
                        if not use(bearing):
                            continue
                        angle = yaw + math.radians(float(bearing))
                        direction = np.array([math.cos(angle), math.sin(angle), 0.])
                        geomid = np.array([-1], dtype=np.int32)
                        distance = mujoco.mj_ray(model, data, origin, direction,
                                                 APE_ONLY, 1, -1, geomid)
                        if 0 <= distance <= dense.MAX_M:
                            hits.append({"bearing_deg": float(bearing),
                                         "distance_m": float(distance),
                                         "geom": model.geom(int(geomid[0])).name})
                    rows.append({"world": world, "time_s": t, "source": source,
                                 "origin_world_m": origin.tolist(),
                                 "rays": int(sum(use(b) for b in dense.YAW_DEG)),
                                 "ape_intercepts_within_range": hits})
    result = {
        "schema": "robot-learning-lab.ape-native-low-fan-mount-screen.v1",
        "scope": "Historical rejected Direct Frame group-4 modeled solids only; no final shell or bracket",
        "scene_sha256": trial.sha(kitchen.OUT / "scene.xml"),
        "native_trace_sha256": trial.sha(kitchen.TRACE),
        "source_sha256": trial.sha(Path(__file__)),
        "max_range_m": dense.MAX_M,
        "mounts_root_local_m": {"front_low": flank.FRONT_ORIGIN_LOCAL_M.tolist(),
                                "left_flank_low": flank.FLANK_ORIGIN_LOCAL_M.tolist()},
        "screened_rays": sum(row["rays"] for row in rows),
        "ape_intercepts_within_range": sum(len(row["ape_intercepts_within_range"])
                                           for row in rows),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(OUT, result["screened_rays"], result["ape_intercepts_within_range"])


if __name__ == "__main__":
    main()
