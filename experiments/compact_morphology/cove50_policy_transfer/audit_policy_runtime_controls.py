"""Record saved policy timing and action conventions before body transfer.

This parses saved YAML nodes without constructing Python tags. It describes
config-declared controls, not first-frame runtime parity or policy behavior.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-policy-interface-audit-20261012/report.json"
POINTER = ROOT / "robot/ape-current-design.json"
REGISTRY = ROOT / "skills/ape/registry.json"
OUT = ROOT / "artifacts/ape-policy-runtime-controls-20261013"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(node: yaml.MappingNode, key: str) -> yaml.Node:
    if not isinstance(node, yaml.MappingNode):
        raise ValueError(f"Expected YAML mapping for {key}")
    hits = [value for name, value in node.value if name.value == key]
    if len(hits) != 1:
        raise ValueError(f"Expected one YAML field {key}, found {len(hits)}")
    return hits[0]


def scalar(node: yaml.Node) -> str:
    if not isinstance(node, yaml.ScalarNode):
        raise ValueError("Expected a YAML scalar")
    return node.value


def atom(node: yaml.Node) -> object:
    value = scalar(node)
    if value in ("null", "~", ""):
        return None
    if value in ("true", "false"):
        return value == "true"
    try:
        return float(value)
    except ValueError:
        return value


def source_document(path: Path, expected_sha: str) -> yaml.MappingNode:
    if sha(path) != expected_sha:
        raise ValueError(f"Saved config changed: {path}")
    document = yaml.compose(path.read_text())
    if not isinstance(document, yaml.MappingNode):
        raise ValueError(f"Expected YAML mapping: {path}")
    return document


def control_row(policy: dict) -> dict:
    identifier = policy["id"]
    config = policy["training_config"]
    shape = policy["checkpoint_actor_shape"]
    base = {
        "id": identifier,
        "actor_format": shape["format"],
        "training_action_channels": shape["action_channels"],
        "saved_actor_inputs": shape["observation_features"],
        "actor_observation_terms": policy["config_actor_observation_terms"],
        "source_body_sha256": policy["source_body_sha256"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "qualification": "source_declared_only_no_selected_body_or_runtime_parity",
    }
    if shape["format"] == "robomimic_bc":
        path = ROOT / config["config.json"]["path"]
        if sha(path) != config["config.json"]["sha256"]:
            raise ValueError(f"Saved BC config changed: {identifier}")
        return {**base, "saved_config": config["config.json"],
                "physics_dt_s": None, "decimation": None,
                "declared_control_period_s": None,
                "declared_control_rate_hz": None,
                "action_term": "source_derived_bc_actor_vector",
                "action_scale_kind": "not_resolved_from_saved_bc_config",
                "action_scale": None,
                "action_scale_nonzero_min": None,
                "action_scale_nonzero_max": None,
                "action_scale_zero_patterns": None,
                "use_default_joint_offset": None,
                "preserve_action_order": None,
                "agent_observation_normalization_declared": None,
                "agent_action_clip_declared": None,
                "runtime_checks": [
                    "capture_actual_73_feature_bc_vector_and_phase",
                    "replace_simulator_object_vectors_with_observed_geometry",
                    "resolve_control_period_and_action_scale_from_native_task",
                    "compare_first_action_and_attained_joint_target",
                ]}
    if shape["format"] != "rsl_rl":
        raise ValueError(f"Unaccounted actor format: {identifier}")
    env_cfg, agent_cfg = config["env.yaml"], config["agent.yaml"]
    env = source_document(ROOT / env_cfg["path"], env_cfg["sha256"])
    agent = source_document(ROOT / agent_cfg["path"], agent_cfg["sha256"])
    decimation = int(scalar(field(env, "decimation")))
    timestep = float(scalar(field(field(field(env, "sim"), "mujoco"),
                                  "timestep")))
    if decimation <= 0 or timestep <= 0:
        raise ValueError(f"Invalid saved control timing: {identifier}")
    actions = field(env, "actions")
    names = [name.value for name, _ in actions.value]
    if names != ["joint_pos"]:
        raise ValueError(f"Unaccounted action terms for {identifier}: {names}")
    joint = field(actions, "joint_pos")
    scale_node = field(joint, "scale")
    if isinstance(scale_node, yaml.MappingNode):
        scales = {name.value: float(scalar(value))
                  for name, value in scale_node.value}
        if len(scales) != shape["action_channels"]:
            raise ValueError(f"Scale patterns differ from actor actions: {identifier}")
        values = [value for value in scales.values() if value != 0]
        kind = "source_joint_regex_map"
        zeros = sum(value == 0 for value in scales.values())
    else:
        scales = float(scalar(scale_node))
        values = [scales] if scales != 0 else []
        kind = "scalar"
        zeros = shape["action_channels"] if scales == 0 else 0
    if not values:
        raise ValueError(f"Saved policy has no nonzero action scale: {identifier}")
    normalize = atom(field(field(agent, "actor"), "obs_normalization"))
    default_offset = atom(field(joint, "use_default_offset"))
    preserve_order = atom(field(joint, "preserve_order"))
    if not all(isinstance(value, bool)
               for value in (normalize, default_offset, preserve_order)):
        raise ValueError(f"Unresolved saved action/normalization flags: {identifier}")
    return {**base, "saved_env_config": env_cfg,
            "saved_agent_config": agent_cfg,
            "physics_dt_s": timestep, "decimation": decimation,
            "declared_control_period_s": round(timestep * decimation, 9),
            "declared_control_rate_hz": round(1 / (timestep * decimation), 9),
            "action_term": "joint_pos",
            "action_scale_kind": kind, "action_scale": scales,
            "action_scale_nonzero_min": min(values),
            "action_scale_nonzero_max": max(values),
            "action_scale_zero_patterns": zeros,
            "action_offset": atom(field(joint, "offset")),
            "use_default_joint_offset": default_offset,
            "preserve_action_order": preserve_order,
            "agent_observation_normalization_declared": normalize,
            "agent_action_clip_declared": atom(field(agent, "clip_actions")),
            "runtime_checks": [
                "compare_actual_first_observation_value_and_order",
                "load_checkpoint_normalizer_and_check_first_normalized_input",
                "compare_first_action_before_and_after_source_joint_mapping",
                "compare_scaled_joint_target_and_attained_state",
                "measure_control_period_reset_history_and_motor_response",
            ]}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    pointer = json.loads(POINTER.read_text())
    registry = json.loads(REGISTRY.read_text())
    if (source["policy_count"] != 29 or
            source["pointer_sha256"] != sha(POINTER) or
            source["registry_sha256"] != sha(REGISTRY) or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            pointer["selected_design_id"] is not None or
            len(registry["policies"]) != 29 or
            {row["id"] for row in source["policies"]} !=
            {row["id"] for row in registry["policies"]}):
        raise ValueError("Saved interface inventory or mechanical pointer changed")
    rows = [control_row(row) for row in source["policies"]]
    counts = Counter(row["actor_format"] for row in rows)
    report = {
        "schema": "robot-learning-lab.ape-policy-runtime-controls.v1",
        "role": "saved_training_config_runtime_contract_inventory_no_rollout",
        "source_sha256": {
            "script": sha(Path(__file__)), "interface_report": sha(SOURCE),
            "pointer": sha(POINTER), "registry": sha(REGISTRY)},
        "selected_design_id": None,
        "policy_count": len(rows),
        "format_counts": dict(counts),
        "rsl_declared_control_periods_s": sorted({
            row["declared_control_period_s"] for row in rows
            if row["actor_format"] == "rsl_rl"}),
        "cases": rows,
        "interpretation": (
            "The 24 RSL-RL timing, action-scale, default-offset and actor "
            "normalization fields are read from exact saved YAML nodes "
            "without executing Python tags. The five BC saved configs do not "
            "resolve these runtime fields in this audit. No first observation, "
            "normalizer state, scaled action, motor response or contact has "
            "been measured on a selected body. Source declarations cannot "
            "qualify transfer."),
    }
    matrix = {
        "schema": "robot-learning-lab.ape-policy-runtime-controls-matrix.v1",
        "role": report["role"],
        "source_report": str((OUT / "report.json").relative_to(ROOT)),
        "policy_count": len(rows),
        "cases": [{
            key: row[key] for key in (
                "id", "actor_format", "training_action_channels",
                "saved_actor_inputs", "physics_dt_s", "decimation",
                "declared_control_period_s", "declared_control_rate_hz",
                "action_scale_kind", "action_scale_nonzero_min",
                "action_scale_nonzero_max", "action_scale_zero_patterns",
                "use_default_joint_offset", "preserve_action_order",
                "agent_observation_normalization_declared", "runtime_checks",
                "qualification")}
            for row in rows],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    matrix["source_report_sha256"] = sha(OUT / "report.json")
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"Audited runtime control declarations for {len(rows)} policies "
          f"({counts['rsl_rl']} RSL-RL, {counts['robomimic_bc']} BC)")


if __name__ == "__main__":
    main()
