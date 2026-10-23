"""Audit saved ape policy body/action interfaces against an explicit target XML.

This records a candidate channel map. It never applies that map to an actor or
claims behavioral transfer; observation, reset and action history need separate
runtime checks followed by fresh native evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "skills/ape/registry.json"
POINTER = ROOT / "robot/ape-current-design.json"
HISTORICAL_SANDBOX = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
DEFAULT_OUT = ROOT / "artifacts/ape-policy-interface-audit-20261012/report.json"
BC_ACTOR_TERMS = ["joint_pos", "joint_vel", "ee_to_cube",
                  "cube_to_goal", "actions", "elapsed_episode_fraction"]
BC_SOURCE_FILES = (
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/reach.py",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/grasp.py",
    "experiments/compact_morphology/mjlab_ape/src/ape_walk/place_tracking.py",
    "experiments/compact_morphology/mjlab_ape/collect_jaw_demos.py",
    "experiments/compact_morphology/mjlab_ape/collect_place_demos.py",
    "experiments/compact_morphology/mjlab_ape/collect_place_from_acquire.py",
    "experiments/compact_morphology/mjlab_ape/train_jaw_bc.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value: str) -> Path:
    path = ROOT / value
    if not path.is_file():
        raise ValueError(f"Missing source: {path}")
    return path


def exact_source(path: Path, expected_sha: str, label: str) -> None:
    if sha(path) != expected_sha:
        raise ValueError(f"Saved source changed: {label}: {path}")


def joint_info(model: mujoco.MjModel, joint_id: int) -> dict:
    return {
        "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        "type": int(model.jnt_type[joint_id]),
        "qpos_address": int(model.jnt_qposadr[joint_id]),
        "neutral_qpos": round(float(model.qpos0[model.jnt_qposadr[joint_id]]), 9),
        "limited": bool(model.jnt_limited[joint_id]),
        "range": [round(float(x), 9) for x in model.jnt_range[joint_id]],
    }


def body_contract(path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    joints = [joint_info(model, i) for i in range(model.njnt)]
    actuators = []
    for i in range(model.nu):
        transmission = int(model.actuator_trntype[i])
        joint_id = int(model.actuator_trnid[i, 0])
        joint = (joint_info(model, joint_id) if transmission in (
            int(mujoco.mjtTrn.mjTRN_JOINT),
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT)) and joint_id >= 0 else None)
        actuators.append({
            "index": i,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
            "transmission_type": transmission,
            "joint": joint,
            "ctrl_limited": bool(model.actuator_ctrllimited[i]),
            "ctrl_range": [round(float(x), 9) for x in model.actuator_ctrlrange[i]],
            "force_limited": bool(model.actuator_forcelimited[i]),
            "force_range": [round(float(x), 9) for x in model.actuator_forcerange[i]],
            "gear": [round(float(x), 9) for x in model.actuator_gear[i]],
        })
    names = [item["name"] for item in actuators]
    if any(name is None for name in names) or len(set(names)) != len(names):
        raise ValueError(f"Unnamed or duplicate actuators: {path}")
    return {
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "joint_order": [item["name"] for item in joints],
        "actuators": actuators,
    }


def mapping(source: dict, target: dict) -> dict:
    src = {row["name"]: row for row in source["actuators"]}
    dst = {row["name"]: row for row in target["actuators"]}
    common = sorted(src.keys() & dst.keys(), key=lambda name: src[name]["index"])
    rows = []
    for name in common:
        old, new = src[name], dst[name]
        old_joint, new_joint = old["joint"], new["joint"]
        joint_fields_changed = (
            [field for field in ("name", "type", "neutral_qpos", "limited", "range")
             if old_joint[field] != new_joint[field]]
            if old_joint is not None and new_joint is not None else
            (["joint_attachment"] if old_joint != new_joint else []))
        changed = [field for field in (
            "transmission_type", "ctrl_limited", "ctrl_range",
            "force_limited", "force_range", "gear") if old[field] != new[field]]
        rows.append({"name": name, "source_index": old["index"],
                     "target_index": new["index"],
                     "index_moved": old["index"] != new["index"],
                     "source_joint_qpos_address": (old_joint["qpos_address"]
                                                   if old_joint else None),
                     "target_joint_qpos_address": (new_joint["qpos_address"]
                                                   if new_joint else None),
                     "joint_fields_changed": joint_fields_changed,
                     "body_control_fields_changed": changed})
    return {
        "candidate_name_map_only": rows,
        "removed_from_source": sorted(src.keys() - dst.keys()),
        "new_in_target": sorted(dst.keys() - src.keys()),
        "moved_index_count": sum(row["index_moved"] for row in rows),
        "moved_joint_qpos_address_count": sum(
            row["source_joint_qpos_address"] != row["target_joint_qpos_address"]
            for row in rows),
        "changed_control_field_count": sum(bool(row["body_control_fields_changed"])
                                           for row in rows),
        "changed_joint_field_count": sum(bool(row["joint_fields_changed"])
                                         for row in rows),
        "source_and_target_actuator_order_identical":
            [a["name"] for a in source["actuators"]] ==
            [a["name"] for a in target["actuators"]],
        "source_and_target_joint_order_identical":
            source["joint_order"] == target["joint_order"],
    }


def checkpoint_actor_dims(path: Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "actor_state_dict" in saved:
        state = saved["actor_state_dict"]
        first, last = state["mlp.0.weight"], state["mlp.6.weight"]
        return {"format": "rsl_rl", "observation_features": int(first.shape[1]),
                "action_channels": int(last.shape[0])}
    metadata = saved["shape_metadata"]
    if metadata["all_obs_keys"] != ["actor"]:
        raise ValueError(f"Unaccounted checkpoint observation keys: {path}")
    return {"format": "robomimic_bc",
            "observation_features": int(metadata["all_shapes"]["actor"][0]),
            "action_channels": int(metadata["ac_dim"])}


def config_actor_terms(path: Path) -> list[str]:
    """Read saved YAML term order without constructing its Python-tagged objects."""
    document = yaml.compose(path.read_text())
    def child(node: yaml.MappingNode, key: str) -> yaml.Node:
        return next(value for name, value in node.value if name.value == key)
    terms = child(child(child(document, "observations"), "actor"), "terms")
    return [name.value for name, _ in terms.value]


def observation_handoff(terms: list[str] | None) -> str:
    if terms is None:
        return "dataset_actor_vector_feature_semantics_unresolved"
    if "elapsed_episode_fraction" in terms and "ee_to_cube" in terms:
        return "source_derived_object_geometry_and_phase_requires_perception_and_clock"
    if {"ee_to_cube", "cube_to_goal"} & set(terms):
        return "simulator_object_geometry_requires_deployable_perception_replacement"
    if {"motion_anchor_pos_b", "motion_anchor_ori_b"} & set(terms):
        return "reference_motion_and_clock_interface_required"
    if "phase" in terms:
        return "explicit_phase_interface_required"
    return "imu_joint_previous_action_and_command_layout_to_reconcile"


def inferred_actor_layout(terms: list[str], source_dof: int) -> list[dict]:
    """Check ordered term widths, without claiming runtime value or order parity."""
    tracking = "motion_anchor_pos_b" in terms
    fixed = {
        "base_ang_vel": 3, "base_lin_vel": 3, "projected_gravity": 3,
        "ee_to_cube": 3, "cube_to_goal": 3, "right_arm_command": 3,
        "motion_anchor_pos_b": 3, "motion_anchor_ori_b": 6,
        "phase": 1, "elapsed_episode_fraction": 1,
    }
    dynamic = {"joint_pos": source_dof, "joint_vel": source_dof,
               "actions": source_dof,
               "command": 2 * source_dof if tracking else 3}
    result = []
    offset = 0
    for name in terms:
        if name not in fixed and name not in dynamic:
            raise ValueError(f"Unaccounted actor observation term: {name}")
        width = fixed[name] if name in fixed else dynamic[name]
        result.append({"term": name, "width": width,
                       "start": offset, "stop_exclusive": offset + width})
        offset += width
    return result


def check_bc_observation_source() -> None:
    markers = {
        BC_SOURCE_FILES[0]: 'cfg.observations["actor"].terms["ee_to_cube"]',
        BC_SOURCE_FILES[1]: "cfg = reach_env_cfg(play=play)",
        BC_SOURCE_FILES[2]: "cfg = held_env_cfg(play=play, model=COLLISION_MODEL)",
        BC_SOURCE_FILES[3]: 'return obs["actor"]',
        BC_SOURCE_FILES[4]: 'actor.append(obs["actor"]',
        BC_SOURCE_FILES[5]: 'actor_rows.append(obs["actor"]',
        BC_SOURCE_FILES[6]: "np.concatenate((original, phase), axis=1)",
    }
    for path, marker in markers.items():
        if marker not in (ROOT / path).read_text():
            raise ValueError(f"BC observation lineage changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-body", type=Path, default=HISTORICAL_SANDBOX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    target_path = args.target_body.resolve()
    out_path = args.out.resolve()
    if not target_path.is_relative_to(ROOT) or not out_path.is_relative_to(ROOT):
        raise ValueError("Target body and audit output must stay in this repository")
    pointer = json.loads(POINTER.read_text())
    registry = json.loads(REGISTRY.read_text())
    target = body_contract(target_path)
    sources: dict[str, dict] = {}
    rows = []
    for policy in registry["policies"]:
        identifier = policy["id"]
        checkpoint = local_path(policy["local_checkpoint"])
        exact_source(checkpoint, policy["sha256"], identifier + " checkpoint")
        replay_body = policy.get("body", registry["body"])
        training_body = policy.get("training_body", replay_body)
        for label, body in (("training", training_body),
                            ("retained replay", replay_body)):
            source_path = local_path(body["path"])
            exact_source(source_path, body["sha256"], identifier + " " + label + " body")
            if body["sha256"] not in sources:
                sources[body["sha256"]] = body_contract(source_path)
        config = {}
        for label, item in policy.get("training_config", {}).items():
            path = local_path(item["local_path"])
            exact_source(path, item["sha256"], identifier + " " + label)
            config[label] = {"path": item["local_path"], "sha256": item["sha256"]}
        source = sources[training_body["sha256"]]
        replay = sources[replay_body["sha256"]]
        actor_dims = checkpoint_actor_dims(checkpoint)
        if actor_dims["action_channels"] != policy["source_dof"]:
            raise ValueError(f"Checkpoint action width differs from shelf: {identifier}")
        actor_terms = (config_actor_terms(ROOT / config["env.yaml"]["path"])
                       if "env.yaml" in config else None)
        actor_terms_basis = "saved_env_config" if actor_terms else None
        if actor_dims["format"] == "robomimic_bc":
            check_bc_observation_source()
            if (actor_dims["observation_features"] != 73 or
                    "config.json" not in config):
                raise ValueError(f"BC observation source differs: {identifier}")
            actor_terms = BC_ACTOR_TERMS
            actor_terms_basis = "source_derived_collector_and_phase_transform"
        if training_body != replay_body:
            env_path = local_path(policy["training_config"]["env.yaml"]["local_path"])
            if "spec_fn: !!python/name:ape_walk.task.ape_spec ''" not in env_path.read_text():
                raise ValueError(f"Training spec does not identify the 20-axis source: {identifier}")
            if (actor_dims != {"format": "rsl_rl",
                               "observation_features": 69,
                               "action_channels": 20} or source["nu"] != 20 or
                    replay["nu"] != 22):
                raise ValueError(f"Saved 20-to-22 actor interface differs: {identifier}")
        layout = inferred_actor_layout(actor_terms, source["nu"])
        if layout[-1]["stop_exclusive"] != actor_dims["observation_features"]:
            raise ValueError(f"Inferred actor term widths differ from checkpoint: {identifier}")
        rows.append({
            "id": identifier,
            "task": policy["task"],
            "checkpoint": policy["local_checkpoint"],
            "checkpoint_sha256": policy["sha256"],
            "declared_training_source_dof": policy["source_dof"],
            "compiled_source_body_actuator_count": source["nu"],
            "declared_training_dof_matches_compiled_source_body":
                policy["source_dof"] == source["nu"],
            "source_body_sha256": training_body["sha256"],
            "retained_replay_body_sha256": replay_body["sha256"],
            "compiled_retained_replay_body_actuator_count": replay["nu"],
            "checkpoint_actor_shape": actor_dims,
            "config_actor_observation_terms": actor_terms,
            "inferred_actor_observation_layout": layout,
            "inferred_layout_width_matches_checkpoint": True,
            "actor_observation_terms_basis": actor_terms_basis,
            "observation_handoff": observation_handoff(actor_terms),
            "training_config": config,
            "interface": mapping(source, target),
            "training_to_retained_replay_interface": mapping(source, replay),
            "retained_replay_to_target_interface": mapping(replay, target),
            "transfer_status": "unverified_no_selected_body_or_fresh_policy_rollout",
        })
    sandbox = target_path == HISTORICAL_SANDBOX.resolve()
    if sandbox and pointer["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical sandbox label no longer matches current pointer")
    selected = (not sandbox and bool(pointer.get("selected_design_id")) and
                pointer.get("physics", {}).get("source_body_sha256") ==
                target["sha256"])
    report = {
        "schema": "robot-learning-lab.ape-policy-control-interface-audit.v3",
        "role": ("rejected_historical_direct_frame_sandbox_interface_inventory"
                 if sandbox else
                 "selected_body_interface_inventory_not_control_qualification"
                 if selected else "unselected_candidate_interface_inventory"),
        "pointer": str(POINTER.relative_to(ROOT)),
        "pointer_sha256": sha(POINTER),
        "selected_design_id": pointer.get("selected_design_id"),
        "registry": str(REGISTRY.relative_to(ROOT)),
        "registry_sha256": sha(REGISTRY),
        "target_body": target,
        "source_bodies": sources,
        "bc_observation_source_sha256": {
            path: sha(ROOT / path) for path in BC_SOURCE_FILES},
        "policy_count": len(rows),
        "policies": rows,
        "interpretation": (
            "Name-based rows are candidate channel mappings only. Five saved "
            "20-action/69-feature actors were trained on a separate 20-actuator "
            "source and subsequently replayed on a 22-actuator body. Their direct "
            "training-to-target map differs from their replay-to-target map. "
            "Inferred ordered term widths match checkpoint inputs but do not "
            "establish runtime values, flattening, reset offsets, action "
            "normalization/history or control timing. "
            "Mass, contacts and actuator response require a fresh paired native "
            "rollout. " + ("No selected new body exists at this audit."
                           if not selected else
                           "A pointer match does not qualify learned behavior.")),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    matrix = {
        "schema": "robot-learning-lab.ape-policy-control-interface-matrix.v3",
        "role": report["role"],
        "source_report": str(out_path.relative_to(ROOT)),
        "source_report_sha256": sha(out_path),
        "case_count": len(rows),
        "cases": [{
            "id": row["id"],
            "source_body": sources[row["source_body_sha256"]]["path"],
            "source_body_label": Path(sources[row["source_body_sha256"]]["path"]).stem,
            "retained_replay_body": sources[row["retained_replay_body_sha256"]]["path"],
            "declared_training_dof": row["declared_training_source_dof"],
            "compiled_source_actuators": row["compiled_source_body_actuator_count"],
            "compiled_replay_actuators": row["compiled_retained_replay_body_actuator_count"],
            "actor_input_features": row["checkpoint_actor_shape"]["observation_features"],
            "actor_output_actions": row["checkpoint_actor_shape"]["action_channels"],
            "actor_observation_terms": row["config_actor_observation_terms"],
            "inferred_actor_input_layout": [
                f"{item['term']}[{item['start']}:{item['stop_exclusive']}]"
                for item in row["inferred_actor_observation_layout"]],
            "inferred_layout_width_matches_checkpoint":
                row["inferred_layout_width_matches_checkpoint"],
            "actor_observation_terms_basis": row["actor_observation_terms_basis"],
            "observation_handoff": row["observation_handoff"],
            "target_actuators": target["nu"],
            "moved_actuator_indices": row["interface"]["moved_index_count"],
            "moved_joint_qpos_addresses":
                row["interface"]["moved_joint_qpos_address_count"],
            "removed_actuators": row["interface"]["removed_from_source"],
            "new_target_actuators": row["interface"]["new_in_target"],
            "qualification": row["transfer_status"],
        } for row in rows],
    }
    out_path.with_name("matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Audited {len(rows)} saved policy interfaces against {target['path']}")


if __name__ == "__main__":
    main()
