"""V281: correct raised-can support contact and repeat lower/upper task views."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import run_color_grounded_entry_v151 as entry
import run_upper_task_view_v274 as upper
from audit_direct_structural_entry_v271 import V149, V150
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from run import ROOT, Vision, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v281"
SEED = 130
MODES = ("present", "absent", "identical_lookalike")
SUPPORTED_CENTER_Z_M = .430


class SupportedRaisedRoom(DirectStructuralRoom):
    def layout(self, seed: int, absent: bool, *, furnished: bool = False) -> None:
        super().layout(seed, absent, furnished=furnished)
        for name in ("TomatoSauce", "Imposter"):
            self.model.body_pos[self.model.body(f"loop_{name}").id, 2] = (
                SUPPORTED_CENTER_Z_M)
        self._forward()


def preflight(mode: str) -> dict:
    room = SupportedRaisedRoom(kind="raised", mode=mode, side=-1)
    try:
        room.layout(SEED, absent=mode != "present", furnished=True)
        top = room.model.geom("loop_table_top").id
        item = room.model.geom("loop_TomatoSauce_geom" if mode == "present"
                               else "loop_Imposter_geom").id
        gap = None if mode == "absent" else float(mujoco.mj_geomDistance(
            room.model, room.data, item, top, 10., None))
        return {"mode": mode, "visible_item_table_top_signed_gap_m": gap,
                "evaluator_start_body_clearance_m":
                    room._current_obstacle_clearance()[0],
                "qualified_support_contact": gap is None or abs(gap) < .001}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    preflights = [preflight(mode) for mode in MODES]
    if not all(row["qualified_support_contact"] for row in preflights):
        raise AssertionError("Raised object is not on the table")
    (OUT / "support-preflight.json").write_text(json.dumps({
        "scope": "V281 correction before actor: V270 raised item mesh was 2.06cm above the table. Set object center z=.430m on the same Direct Frame and scene to leave less than1mm signed table-top gap; validate before repeating the frozen entry/upper-view rule.",
        "source_sha256": sha(SOURCE), "rows": preflights}, indent=2) + "\n")
    vision = Vision("generic")
    exemplar = np.load(V149)
    owner_hist = np.asarray(json.loads(V150.read_text())[
        "owner_color"]["hue_histogram"])
    old = (entry.OUT, entry.MovedPropRoom,
           upper.OUT, upper.DirectStructuralRoom)
    rows = []
    try:
        for mode in MODES:
            entry.OUT = OUT / "entry" / mode
            entry.MovedPropRoom = lambda positions, current_mode=mode: (
                SupportedRaisedRoom(positions, kind="raised",
                                    mode=current_mode, side=-1))
            variant = {"variant": f"supported-raised-{mode}",
                       "cream_cheese_xy_m": (.95, -.16),
                       "parmesan_xy_m": (.98, .18)}
            lower = entry.episode(SEED, variant, mode != "present", 1.,
                                  vision, exemplar, owner_hist)
            upper.OUT = OUT
            upper.DirectStructuralRoom = SupportedRaisedRoom
            result = upper.evaluate(SEED, "raised", mode,
                                    vision, exemplar, owner_hist, lower)
            rows.append({"mode": mode, "lower": lower, "upper": result})
            print(mode, "lower accepted", sum(
                x["accepted_box"] is not None for x in lower["rows"]),
                "upper", result["status"], flush=True)
    finally:
        (entry.OUT, entry.MovedPropRoom,
         upper.OUT, upper.DirectStructuralRoom) = old
    report = {"scope": "V281 exact selected Direct Frame and geometrically supported table can (<1mm gap), present/ordinary absent/identical lookalike. Repeat frozen V151 lower-head entry followed by V274 guarded upper-head task view only after no lower candidate. Same owner-trained RGB proposal/appearance/hue rule, hypothetical ideal co-aligned upper 8-150cm depth for raised-support evidence, scripted owner exact-instance yes/no, no base movement or reach claim. This corrects V270's 2.06cm floating target; V270/V274 historical results remain unchanged.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "support_preflight_sha256": sha(OUT / "support-preflight.json"),
              "rows": rows}
    (OUT / "supported-raised-view-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
