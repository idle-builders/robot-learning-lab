"""Screen angular-grid phase before interpreting the live backup-range trial."""

from __future__ import annotations

import json
import math

import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ART, ROOT, relative, sha
import run_cross_room_inference_response as motion
import run_joined_live_hazard_16 as trial


OUT = ART / "backup_range_angular_phase_16_screen.json"
STEPS = (5, 10, 20)
PHASE_FRACTIONS = (0., .25, .5, .75)
SAMPLES = tuple(.05 + .1 * i for i in range(13))


def main() -> None:
    cases = trial.case_rows()
    hazard_prop = TexturedFloorObject(trial.HAZARD_PROP, scale=.012)
    rows = []
    for case in cases:
        scene, base, _, _ = trial.scene_and_props(case)
        state = trial.actor_state(case)
        start = state["actual"]
        yaw = state["yaw"]
        heading = math.radians(yaw + case["entry_relative_deg"])
        incoming = np.array([math.sin(heading), math.cos(heading)])
        eye = np.array([start[0], scene.floor_y + .42, start[1]])
        for step in STEPS:
            for phase_fraction in PHASE_FRACTIONS:
                phase = step * phase_fraction
                angles = np.arange(-180. + phase, 180. + phase, step,
                                   dtype=np.float32)
                scene.set_props(base)
                baseline = scene.range_fan(eye, yaw, owner_xz=None,
                    angles_deg=angles, heights_m=(.1,))[0]
                alarm_time = alarm_angle = None
                for t in SAMPLES:
                    hazard = motion.incoming_xy(start, incoming, t)
                    scene.set_props({**base, trial.HAZARD:
                        (hazard_prop, tuple(hazard), 0.)})
                    rays = scene.range_fan(eye, yaw, owner_xz=None,
                        angles_deg=angles, heights_m=(.1,))[0]
                    with np.errstate(invalid="ignore"):
                        changed = (baseline - rays > .08) & (rays < .9)
                    if np.any(changed):
                        ids = np.flatnonzero(changed)
                        chosen = ids[np.argmin(rays[ids])]
                        alarm_time, alarm_angle = t, float(angles[chosen])
                        break
                escape = None
                if alarm_angle is not None:
                    opposite = math.radians(yaw + alarm_angle + 180.)
                    escape = np.array([math.sin(opposite), math.cos(opposite)])
                gaps = [motion.proxy_gap(motion.actor_xy(start, escape,
                    alarm_time, float(t)), motion.incoming_xy(start,
                    incoming, float(t)), hazard_prop.footprint_radius_m)
                    for t in np.arange(0., 2.001, .02)]
                rows.append({"scene_key": case["scene_key"],
                    "branch": case["branch"],
                    "entry_relative_deg": case["entry_relative_deg"],
                    "step_deg": step, "phase_deg": phase,
                    "first_changed_return_s": alarm_time,
                    "selected_return_angle_deg": alarm_angle,
                    "ideal_response_min_gap_m": min(gaps)})
        print(case["scene_key"], flush=True)
    result = {"schema": "ape.backup-range-angular-phase-16-screen.v1",
        "source_precommit": relative(trial.PRECOMMIT),
        "source_precommit_sha256": sha(trial.PRECOMMIT),
        "samples_s": SAMPLES,
        "rows": rows,
        "limits": "Development-exposed static ray screen at each saved wait pose. The actor stays at its start pose while seeking a changed return; ideal response is replayed after detection. No concurrent image worker, sensor delivery latency, body occlusion, range noise or post-response ray updates. This is angular-phase sensitivity, not physical sensor qualification."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for step in STEPS:
        for fraction in PHASE_FRACTIONS:
            subset = [x for x in rows if x["step_deg"] == step and
                      x["phase_deg"] == step * fraction]
            print(step, fraction, "detected", sum(x[
                "first_changed_return_s"] is not None for x in subset),
                "gap_breaches", sum(x["ideal_response_min_gap_m"] < .05
                                    for x in subset))


if __name__ == "__main__":
    main()
