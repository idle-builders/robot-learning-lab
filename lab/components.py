"""Cross-domain checks against sourced parts; never substitute stall for rated load."""

from pathlib import Path
import yaml


def check_components(robot, catalog_path=Path("robot/actuator-catalog.yaml")):
    catalog = yaml.safe_load(catalog_path.read_text())["parts"]
    used = {j.family for j in robot.joints}
    results = {}
    for family in sorted(used):
        actuator = robot.config.actuators[family]
        if actuator.part_id is None:
            results[family] = {
                "status": "not_run",
                "reason": "no purchased part selected",
            }
            continue
        if actuator.part_id not in catalog:
            results[family] = {"status": "fail", "reason": "unknown part_id"}
            continue
        p = catalog[actuator.part_id]
        power = robot.config.power
        violations = []
        if not (
            p["voltage_range_v"][0]
            <= power.minimum_bus_v
            <= power.battery_max_v
            <= p["voltage_range_v"][1]
        ):
            violations.append("direct motor rail is outside the specified input range")
        if actuator.effort_limit_nm > p["rated_effort_nm"]:
            violations.append("modeled effort exceeds datasheet rated load")
        if actuator.velocity_limit_rad_s > p["no_load_speed_rad_s"]:
            violations.append("modeled speed exceeds datasheet no-load speed")
        if abs(actuator.mass_kg - p["mass_kg"]) > p["mass_tolerance_kg"] + 1e-9:
            violations.append("allocated actuator mass differs from selected part")
        if (
            not p["signal_high_range_v"][0]
            <= power.logic_voltage_v
            <= p["signal_high_range_v"][1]
        ):
            violations.append("logic high is outside the specified input range")
        results[family] = {
            "status": "fail" if violations else "pass",
            "part_id": actuator.part_id,
            "violations": violations,
            "datasheet": p,
        }
    statuses = [r["status"] for r in results.values()]
    return {
        "status": "fail"
        if "fail" in statuses
        else "not_run"
        if "not_run" in statuses
        else "pass",
        "families": results,
        "scope": "datasheet envelope compatibility on a direct shared motor rail",
        "limitations": [
            "rated torque and no-load speed are separate bounds, not a simultaneous operating point",
            "voltage/temperature derating and torque-speed response remain unmeasured",
            "generic ideal-torque simulation is not the listed servo's position/speed inner loop",
            "part placement, mounts, cable clearance and return signal levels remain unverified",
        ],
    }
