"""Measure effective sole/terrain friction in the assembled MJLab scene.

This compiles the pinned Direct Frame body with the Cove velocity scene and
samples contact endpoints. It is a setup check, not a policy evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from functools import partial
from pathlib import Path

import mujoco
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import (
    DIRECT_FRAME_SHA256, configure_frame50_motors,
    give_soles_contact_priority, require_direct_frame,
)


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def contact_rows(model: mujoco.MjModel, grip: float) -> list[dict]:
    for name in ("robot/left_sole", "robot/right_sole"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom < 0:
            raise ValueError(f"Missing assembled scene geom {name}")
        model.geom_friction[geom, 0] = grip
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    initial_height = float(data.qpos[2])
    for millimeters in range(0, 81):
        data.qpos[2] = initial_height - millimeters / 1000.0
        mujoco.mj_forward(model, data)
        rows = []
        for i in range(data.ncon):
            contact = data.contact[i]
            first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if "terrain" not in (first, second) or not any("sole" in value for value in (first, second)):
                continue
            rows.append({"geoms": [first, second],
                         "root_drop_m": millimeters / 1000.0,
                         "effective_sliding_friction": float(contact.friction[0])})
        if rows:
            return rows
    raise ValueError("No sole/terrain contact within 80 mm of assembled keyframe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, default=BODY)
    parser.add_argument("--scene", choices=("velocity", "support"), default="velocity")
    parser.add_argument("--static-grip", type=float,
                        help="Apply the repaired evaluator's sole-only static grip before compilation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    body = args.body.resolve()
    require_direct_frame(body)
    cfg = (ape.env_cfg(elbowed=True) if args.scene == "velocity" else
           load_env_cfg(ape.ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID, play=True))
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, body)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    configure_frame50_motors(cfg, body)
    give_soles_contact_priority(cfg)
    if args.static_grip is not None:
        cfg.scene.entities["robot"].collisions[0].friction = {
            r"^(left|right)_sole$": (args.static_grip,)
        }
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(mujoco.MjModel.from_xml_path(str(body)).qpos0[2]))
    model = Scene(cfg.scene, device="cpu").compile()
    sole_geoms = {}
    for name in ("robot/left_sole", "robot/right_sole"):
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        sole_geoms[name] = {"priority": int(model.geom_priority[geom]),
                            "compiled_friction": float(model.geom_friction[geom, 0])}
        if args.static_grip is not None and abs(
            sole_geoms[name]["compiled_friction"] - args.static_grip
        ) > 1e-6:
            raise ValueError(f"Static evaluator grip failed to compile for {name}")
    terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    endpoints = {str(grip): contact_rows(model, grip) for grip in (0.65, 1.15)}
    for key, rows in endpoints.items():
        if any(abs(row["effective_sliding_friction"] - float(key)) > 1e-6 for row in rows):
            raise ValueError(f"Friction endpoint {key} was masked")
    report = {
        "schema": "robot-learning-lab.cove50-assembled-contact-probe.v1",
        "body_sha256": DIRECT_FRAME_SHA256,
        "mujoco_version": mujoco.__version__,
        "pollen_mdp_source": str(Path(inspect.getfile(ape.microduck_mdp)).resolve()),
        "pollen_mdp_sha256": hashlib.sha256(
            Path(inspect.getfile(ape.microduck_mdp)).read_bytes()
        ).hexdigest(),
        "scene": f"MJLab {args.scene} flat ape, Cove Direct Frame motor and collision config",
        "static_evaluator_grip": args.static_grip,
        "terrain": {"priority": int(model.geom_priority[terrain]),
                    "compiled_friction": float(model.geom_friction[terrain, 0])},
        "sole_geoms": sole_geoms,
        "sole_terrain_pair_overrides": [
            i for i in range(model.npair)
            if terrain in (model.pair_geom1[i], model.pair_geom2[i])
            and any("sole" in mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                    for geom in (model.pair_geom1[i], model.pair_geom2[i]))
        ],
        "contact_endpoints": endpoints,
        "claim_limit": "Compiled local CPU scene contact endpoint check; not CUDA DR draws or policy performance.",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
