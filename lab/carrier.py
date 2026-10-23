"""Cross-check the compiled carrier nets against firmware and selected parts."""

from pathlib import Path
import json
import yaml


def check_carrier(root):
    root = Path(root)
    nets = {k: set(v) for k, v in json.loads((root / "nets.json").read_text()).items()}
    parts = json.loads((root / "parts.json").read_text())
    layout = json.loads((root / "layout.json").read_text())
    controller = yaml.safe_load((root / "controller.yaml").read_text())
    issues = []

    def require(net, *pads):
        if not set(pads) <= nets.get(net, set()):
            issues.append(
                f"{net}: missing expected pads {set(pads) - nets.get(net, set())}"
            )

    for signal, gpio in controller["pins"].items():
        require(signal.upper(), "U1." + parts["Pico2W"]["pins"][f"gp{gpio}"])
    require(
        "GND",
        *[f"U1.{p}" for p in [3, 8, 13, 18, 23, 28, 33, 38]],
        "J1.1",
        "U2.3",
        "U3.3",
        "U4.3",
        "U5.3",
    )
    require("V3", "U1.36", "U2.5", "U3.5", "U4.5", "U5.5")
    require("V5_IN", "J1.2", "D1.2")
    require("VSYS", "D1.1", "U1.39")
    require("VBAT_SENSE", "J1.3", "R9.1")
    require("BATTERY_ADC", "R9.2", "R10.1", "C5.1")
    require("GND", "R10.2", "C5.2", "R7.2", "R8.2", "R11.2")
    require("STOP_CHAIN_OK", "J5.2", "R7.1")
    require("ARM_SWITCH", "J6.2", "R8.1")
    require("MOTION_OK_OUT", "J1.4", "R11.1")
    for b, (tx, rx, series, oe, bias, conn) in enumerate(
        [("U2", "U3", "R1", "R2", "R3", "J3"), ("U4", "U5", "R4", "R5", "R6", "J4")]
    ):
        require(f"BUS{b}_TX", tx + ".2")
        require(f"BUS{b}_RX", rx + ".4")
        require(f"BUS{b}_OE_N", tx + ".1", oe + ".2")
        require("V3", oe + ".1", bias + ".1")
        require(f"BUS{b}_DRIVE", tx + ".4", series + ".1")
        require(f"BUS{b}_DATA", series + ".2", rx + ".2", bias + ".2", conn + ".2")
        require("GND", conn + ".1")
    for i in range(1, 7):
        require("V3" if i != 5 else "BATTERY_ADC", f"C{i}.1")
        require("GND", f"C{i}.2")
    # A pad must not be on two logical nets; independently verify actual assignment set.
    seen = set()
    for name, pads in nets.items():
        if seen & pads:
            issues.append(f"{name}: pad assigned to multiple nets")
        seen |= pads

    def resistance(ref):
        return parts[layout["components"][ref]["part"]]["resistance_ohm"]

    divider = controller["voltage_divider"]
    if (
        resistance("R9") != divider["high_ohm"]
        or resistance("R10") != divider["low_ohm"]
    ):
        issues.append("ADC divider differs from firmware conversion")
    for r in ["R2", "R5"]:
        if resistance(r) != controller["transceivers"]["oe_pullup_ohm"]:
            issues.append("OE pull-up differs from controller contract")
    for u in ["U2", "U4"]:
        if (
            parts[layout["components"][u]["part"]]["mpn"]
            != controller["transceivers"]["tx"]
        ):
            issues.append("TX part mismatch")
    for u in ["U3", "U5"]:
        if (
            parts[layout["components"][u]["part"]]["mpn"]
            != controller["transceivers"]["rx"]
        ):
            issues.append("RX part mismatch")
    # Divider worst case at 14 V with 1% resistors, compared with a 5%-low 3.3 V rail.
    adc_max = (
        14
        * (resistance("R10") * 1.01)
        / (resistance("R9") * 0.99 + resistance("R10") * 1.01)
    )
    if adc_max >= 3.3 * 0.95:
        issues.append("ADC range exceeds minimum logic rail")
    return {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "part_count": len(layout["components"]),
        "adc_at_14v_worst_case_v": adc_max,
        "scope": "net assignments and selected constraints; not comprehensive ERC, SI, EMC, power-off or physical validation",
    }
