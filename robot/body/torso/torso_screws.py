"""Torso and pelvis screw connections (Idle T-nh6ytf): parts rebuilt for the mount check, manifest patched.

What the mount check measured on the selected body (2026-11-20, robot/body/checks run.py --section Torso):

  stage 3 power carrier   the board bosses stopped 0.51 mm (power supervisor) / 1.74 mm (main precharge) short of the PCB, and the
                          supervisor sleeves (4.0 mm) covered 4.0 of the 6.45 mm boss + frame, so the clamped stack had layer gaps
                          of 0.52 / 1.74 mm (board to boss) and 2.29 mm (sleeve to the rear washer); the precharge sleeves had a
                          2.0 mm bore for a 2.0 mm M2 shank.
  torso_mate x4           M3x16 screws, washers and nuts from the earlier waist-pelvis mate (artifact mate-r7, no script): the nut
                          sits in the honeycomb web, 1.5-2.0 mm of shank in solid chassis, heads inside the stage 4 plate. They
                          clamp nothing of the present stages (those use the stage bores of common.stage_interface_specs).

This script rebuilds the stage 3 carrier and its sleeves from build_torso_carriers.py (the same builder as the selected STEP:
17364.17 mm3 before the edit) with bosses that reach the PCB faces and full-thickness flush M2 sleeves (bore 2.2 mm,
standard-parts.json sleeve.bore_for; head on the outer face), and drops the stale torso_mate parts. Positions come from
common.stage_interface_specs and common.STAGE3_*; nothing is typed here.

  recreate:  .venv/bin/python -B robot/body/torso/torso_screws.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/torso/torso_screws.py apply   (patch the roll-flip variant manifest)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(ROOT / "robot/body"))

OUT = ROOT / "artifacts/robot/body/torso/torso-screws"
VARIANT = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
STAGE3 = "stage3_front_power_electronics_carrier"
PART, SLEEVES = f"combined__{STAGE3}", f"combined__{STAGE3}_steel_sleeves"
STALE = ("torso_mate_",)      # torso_mate_<n>_M3x16_socket_screw / _rear_M3_washer / _front_M3_washer / _M3_hex_nut


def build():
    import build_torso_carriers as B
    import common
    import motor_refit as MR
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    MR.PARTS = OUT / "parts"
    body, _ = B.BUILDERS[STAGE3]("truss_box")        # COMBINED_WINNERS: stage 3 is the truss_box family
    sleeves, _, _ = common.build_stage_sleeves(STAGE3)
    recs = {PART: MR.save(PART, body.wrapped, "printed"),
            SLEEVES: MR.save(SLEEVES, sleeves.wrapped, "purchased")}
    recs[SLEEVES]["mass_kg"] = round(recs[SLEEVES]["volume_mm3"] * 7.93e-6, 6)
    recs[SLEEVES]["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    (OUT / "records.json").write_text(json.dumps(dict(records=recs), indent=1) + "\n")
    print("wrote", OUT / "records.json", {k: v["volume_mm3"] for k, v in recs.items()}, flush=True)


def swap_parts(parts):
    """The manifest part list with the stage 3 carrier + sleeves replaced by the rebuilt ones and the stale torso_mate
    hardware dropped. Without the build output the manifest stays buildable (unedited stage 3, stale parts kept) and says so."""
    if not (OUT / "records.json").exists():
        print(f"WARNING torso screws: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED stage 3 carrier and the "
              "stale torso_mate parts stay in the manifest; run `.venv/bin/python -B robot/body/torso/torso_screws.py build` "
              "(T-nh6ytf) first")
        return parts
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        if p["name"].startswith(STALE):
            continue
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("native_step_offset_mm", None); q.pop("placement_offset_mm", None)
            q["torso_screws"] = "rebuilt by robot/body/torso/torso_screws.py (T-nh6ytf)"
            res.append(q)
        else:
            res.append(p)
    return res


def apply(variant: Path = VARIANT, out: Path | None = None):
    """Patch a roll-flip variant manifest in place (or into `out`)."""
    man = json.loads(variant.read_text())
    n0 = len(man["parts"])
    man["parts"] = swap_parts(man["parts"])
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, n0, "->", len(man["parts"]), "parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
