"""Independent physical-pad checks for the selected LSM6DSO circuit."""

from pathlib import Path
import json


def compiled_imu(root):
    definitions = json.loads((root / "parts.json").read_text())
    instances = json.loads((root / "components.json").read_text())
    result = {}
    for file in (root / "build/builds/default/pinout").glob("*.json"):
        row = json.loads(file.read_text())
        ref = row["designator"]
        if ref in result or ref not in instances:
            raise ValueError("unexpected compiled IMU component")
        if row["typeName"].split("::")[-1] != instances[ref]:
            raise ValueError("compiled IMU type mismatch")
        pins = {}
        for lead in row["leads"]:
            for number in lead["padNumbers"]:
                if number in pins:
                    raise ValueError("duplicate compiled pad")
                if not lead["netName"]:
                    raise ValueError("unnamed IMU net")
                pins[number] = lead["netName"]
        definition = definitions[instances[ref]]
        if set(pins) != set(definition["pins"].values()):
            raise ValueError("missing IMU pads")
        result[ref] = {**definition, "nets": pins}
    if set(result) != set(instances):
        raise ValueError("compiled IMU instance mismatch")
    return result


def verify_imu(
    circuit, carrier_interface=Path("artifacts/carrier-v13/board-interface.json")
):
    expected = {
        "U1": {
            "1": "GND",
            "2": "GND",
            "3": "GND",
            "4": "INT1",
            "5": "V3",
            "6": "GND",
            "7": "GND",
            "8": "V3",
            "9": "NC_INT2",
            "10": "NC_OCS",
            "11": "V3",
            "12": "V3",
            "13": "SCL",
            "14": "SDA",
        },
        "J1": {"1": "V3", "2": "GND", "3": "SDA", "4": "SCL", "5": "INT1"},
        "C1": {"1": "V3", "2": "GND"},
        "C2": {"1": "V3", "2": "GND"},
    }
    issues = []
    if set(circuit) != set(expected):
        issues.append("unexpected circuit population")
    for ref, pins in expected.items():
        if circuit.get(ref, {}).get("nets") != pins:
            issues.append("physical-pad mismatch " + ref)
    if circuit.get("U1", {}).get("mpn") != "LSM6DSOTR":
        issues.append("wrong IMU identity")
    for ref in ("C1", "C2"):
        if circuit.get(ref, {}).get("capacitance_f") != 1e-7:
            issues.append("wrong decoupling " + ref)
    # Only one pin may inhabit each explicitly unconnected net.
    for name in ("NC_INT2", "NC_OCS"):
        if (
            sum(net == name for p in circuit.values() for net in p["nets"].values())
            != 1
        ):
            issues.append("NC pin connected to another pad " + name)
    carrier = json.loads(Path(carrier_interface).read_text())
    host_pins = carrier["connector_pads"]["J7"]
    expected_host = {
        "1": "V3",
        "2": "GND",
        "3": "IMU_SDA_RESERVED",
        "4": "IMU_SCL_RESERVED",
        "5": "IMU_IRQ_RESERVED",
    }
    if len(host_pins) != 5 or {p["pin"]: p["net"] for p in host_pins} != expected_host:
        issues.append("carrier J7 differs from straight-through five-wire contract")
    return {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "part_count": len(circuit),
        "carrier_interface_schema": carrier["schema_version"],
        "harness_pin_map": [
            {"carrier": f"J7.{i}", "sensor": f"J1.{i}"} for i in range(1, 6)
        ],
        "scope": "mode1 physical-pad connectivity and decoupling only; no assembly/harness qualification",
    }
