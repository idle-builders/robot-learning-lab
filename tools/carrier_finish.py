#!/usr/bin/env python3
"""Import the local router session without changing Atopile pad-to-net assignments."""

import json
from pathlib import Path
import sys
import pcbnew as pcb
import wx

root = Path(sys.argv[1])
app = wx.App(False)
board = pcb.LoadBoard(str(root / "carrier-unrouted.kicad_pcb"))
if not pcb.ImportSpecctraSES(board, str(root / "carrier.ses")):
    raise RuntimeError("session import failed")
board.BuildConnectivity()
pcb.SaveBoard(str(root / "carrier.kicad_pcb"), board)
nets = {}
for fp in board.GetFootprints():
    for pad in fp.Pads():
        if pad.GetNumber() and pad.GetNetname():
            nets.setdefault(pad.GetNetname(), []).append(
                f"{fp.GetReference()}.{pad.GetNumber()}"
            )
expected = json.loads((root / "nets.json").read_text())
if {k: sorted(v) for k, v in nets.items()} != {
    k: sorted(v) for k, v in expected.items()
}:
    raise ValueError("router changed electrical pad assignments")
print("Router session imported; all pad-to-net assignments preserved")

footprints = {fp.GetReference(): fp for fp in board.GetFootprints()}
keepouts = [z.GetZoneName() for z in footprints["U1"].Zones()]
if not {
    "RF Copper Keep Out",
    "Antenna Copper Keep Out",
    "Board Keep Out Suggestion RF",
} <= set(keepouts):
    raise ValueError("Pico RF keepouts did not survive circuit/layout/routing")
connector_nets = {}
layout = json.loads((root / "layout.json").read_text())
for ref, fp in footprints.items():
    if ref.startswith("J"):
        connector_nets[ref] = [
            dict(
                pin=p.GetNumber(),
                net=p.GetNetname(),
                xy_mm=[pcb.ToMM(p.GetPosition().x), pcb.ToMM(p.GetPosition().y)],
            )
            for p in fp.Pads()
            if p.GetNumber()
        ]
(root / "board-interface.json").write_text(
    json.dumps(
        dict(
            schema_version=2,
            coordinate_system="KiCad PCB 2D millimetres: outline upper-left origin, x right, y down; no z coordinate implied",
            step_frame={
                "axes": "right-handed: x right, y toward board top, z toward components",
                "xy_from_pcb": [[1, 0], [0, -1]],
                "datum": "STEP substrate bottom at z=0; component models retain KiCad offsets; nominal stack thickness is not STEP core thickness",
            },
            outline_mm=layout["size_mm"],
            thickness_mm=pcb.ToMM(board.GetDesignSettings().GetBoardThickness()),
            mounting_holes_mm=layout["mounting_holes_mm"],
            mounting_hole_diameter_mm=layout["mounting_hole_diameter_mm"],
            connector_pads=connector_nets,
            pico_keepouts=keepouts,
            trace_via_count=len(list(board.GetTracks())),
            enclosure_installation="not_run; STEP exported separately",
        ),
        indent=2,
    )
    + "\n"
)
# Deliberate power-to-ground short must be caught by independent KiCad DRC.
pads = {p.GetNumber(): p for p in footprints["J1"].Pads()}
short = pcb.PCB_TRACK(board)
short.SetNet(pads["1"].GetNet())
short.SetWidth(pcb.FromMM(0.3))
short.SetLayer(pcb.F_Cu)
short.SetStart(pads["1"].GetPosition())
short.SetEnd(pads["2"].GetPosition())
board.Add(short)
pcb.SaveBoard(str(root / "carrier-shorted.kicad_pcb"), board)
