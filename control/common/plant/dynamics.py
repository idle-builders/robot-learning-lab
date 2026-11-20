"""Best-estimate actuator dynamics (r4 sidecar) for the R3 boundary and spec.

resolve() turns the sidecar into the flat parameters the boundary and the spec
use: the servo `body` on the 20 C047 joints (its catalog law, the Feetech
firmware duty loop, rotor inertia and Coulomb friction) and the XM335 neck
(R3 law, rotor inertia and friction), and best-estimate ranges for the joints
whose R3 source range is unspecified (the six hips). Without it the runtime is
exactly R3.

A name in ALLOCATIONS gives each joint group its own servo (mixed sizing): the
result then also carries per-joint law, duty loop, rotor inertia, friction and
mass under "joints"; the flat fields hold the allocation's default servo.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIDECAR = ROOT / "robot/model/dynamics/actuator-dynamics.json"
CATALOG = ("stall_torque_Nm", "no_load_speed_rad_s", "rated_torque_Nm", "idle_current_A", "stall_current_A")
GROUPS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")
# Mixed sizing (agility/allocation.py, record cove50-wbc-servo-allocation-20261105): the
# ST3025 on the loaded leg joints, the STS3250 (C047 case) on the arms, hip yaw and ankle roll.
ALLOCATIONS = {
    "st3025-pitch-roll-sts3250": dict({g: "sts3250" for g in GROUPS}, **{g: "st3025" for g in ("hip_pitch", "hip_roll", "knee", "ankle_pitch")}),
    # The same with a 24 V brushless knee (CubeMars AK45-10 V3, sidecar revision 3; agility/allocation.py demand check).
    "st3025-pitch-roll-sts3250-ak45knee": dict({g: "sts3250" for g in GROUPS}, **{g: "st3025" for g in ("hip_pitch", "hip_roll", "ankle_pitch")}, knee="ak45-10"),
    # The same on the shared 12 V rail: the ST3025 knee through an assumed 1:1.5 speed-up stage (sidecar revision 3.2).
    "st3025-pitch-roll-sts3250-geared-knee": dict({g: "sts3250" for g in GROUPS}, **{g: "st3025" for g in ("hip_pitch", "hip_roll", "ankle_pitch")}, knee="st3025-x1.5"),
    # Whole-body 24 V brushless sets (sidecar revision 4): arms and hip yaw on the light GIM3505-8, legs on
    # EduLite 05 or on Damiao DM-J3507 with a DM-J4310 hip roll.
    "qdd-edulite-gim": dict({g: "gim3505-8" for g in GROUPS}, **{g: "edulite-05" for g in ("hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll")}),
    "qdd-damiao-gim": dict({g: "gim3505-8" for g in GROUPS}, **{g: "dm-j3507" for g in ("hip_pitch", "knee", "ankle_pitch", "ankle_roll")}, hip_roll="dm-j4310"),
    # The locked three-tier brick set (robot/model/motor-map/motor-map.json): ST3025 on the fast leg
    # joints, STS3250 on hip roll and the arms, the C047 on hip yaw and ankle roll. The neck keeps the XM335 law here.
    "tier3-st3025-sts3250-c047": dict({g: "c047" for g in GROUPS}, **{g: "st3025" for g in ("hip_pitch", "knee", "ankle_pitch")},
        **{g: "sts3250" for g in ("hip_roll", "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")}),
}


def resolve(body="c047", levels=None):
    if body in ALLOCATIONS:
        groups = ALLOCATIONS[body]
        default = max(sorted(set(groups.values())), key=list(groups.values()).count)
        out = dict(resolve(default, levels), body=body, allocation=groups)
        per = {m: resolve(m, levels) for m in set(groups.values())}
        out["joints"] = {f"{side}_{g}": dict(model=m, mass_kg=per[m]["mass_kg"], law=per[m]["law"], duty=per[m]["duty"],
            armature=per[m]["armature"]["c047"], frictionloss=per[m]["frictionloss"]["c047"]) for g, m in groups.items() for side in ("R", "L")}
        return out
    return _resolve(body, levels)


def _resolve(body, levels):
    """levels: {parameter: 'low'|'nominal'|'high'} for sensitivity; default nominal throughout.

    parameter is one of armature, frictionloss, duty_gain, rate_limit, backlash, neck_armature, rail, hip_range.
    """
    sidecar = json.loads(SIDECAR.read_text())
    models = sidecar["models"]
    m, neck = models[body], models["xm335"]
    levels = dict(levels or {})
    unknown = set(levels) - {"armature", "frictionloss", "duty_gain", "rate_limit", "backlash", "neck_armature", "rail", "hip_range"}
    if unknown:
        raise ValueError(f"unknown sensitivity parameters {sorted(unknown)}")

    def pick(field, key):
        value = field.get(levels.get(key, "nominal"))
        return field["nominal"] if value is None else value

    rail = sidecar.get("supply", {}).get("rail_V")
    ranges = sidecar.get("joint_ranges_deg", {}).get("joints", {})
    return dict(version=sidecar["version"], revision=sidecar.get("revision", 1), body=body, levels=levels, mass_kg=m["mass_kg"],
        rail_V=pick(rail, "rail") if rail else 12.,
        law={k: m[k] for k in CATALOG},
        duty=dict(gain=pick(m["duty_gain_per_rad"], "duty_gain"), max=m["max_duty"]["nominal"],
            rate=pick(m["target_rate_limit_rad_s"], "rate_limit"), backlash=pick(m["backlash_rad"], "backlash")),
        armature=dict(c047=pick(m["armature_kg_m2"], "armature"), xm335=pick(neck["armature_kg_m2"], "neck_armature")),
        frictionloss=dict(c047=pick(m["frictionloss_Nm"], "frictionloss"), xm335=neck["frictionloss_Nm"]["nominal"]),
        ranges_deg={f"{side}_{axis}": list(pick(r, "hip_range")) for axis, r in ranges.items() for side in ("R", "L")})
