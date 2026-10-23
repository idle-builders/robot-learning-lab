#!/usr/bin/env python3
"""Place Atopile-produced footprints using KiCad's bundled Python; export DSN."""

import json
from pathlib import Path
import sys
import pcbnew as pcb
import wx


def main():
    app = wx.App(False)  # noqa: F841 -- keep the native application alive for PCB IO
    root = Path(sys.argv[1])
    layout = json.loads((root / "layout.json").read_text())
    parts = json.loads((root / "parts.json").read_text())
    board = pcb.LoadBoard(str(root / "layouts/default/default.kicad_pcb"))
    footprints = {f.GetReference(): f for f in board.GetFootprints()}
    if set(footprints) != set(layout["components"]):
        raise ValueError("Atopile / layout component set mismatch")
    for path in (root / "build/builds/default/pinout").glob("*.json"):
        pinout = json.loads(path.read_text())
        ref = pinout["designator"]
        if pinout["typeName"].split("::")[-1] != layout["components"][ref]["part"]:
            raise ValueError(f"Atopile / BOM part mismatch: {ref}")
    # Atopile 0.15.8 drops KiCad 10 footprint zones/groups during import.
    # Keep its electrical assignments, but restore exact vendored physical footprints
    # through KiCad before placement/routing so RF and pad keepouts survive.
    for ref, old in list(footprints.items()):
        definition = parts[layout["components"][ref]["part"]]
        loaded = pcb.FootprintLoad(
            str(root / "parts"), Path(definition["footprint"]).stem
        )
        fresh = pcb.FOOTPRINT(loaded)
        assignments = {p.GetNumber(): p.GetNet() for p in old.Pads() if p.GetNumber()}
        if set(assignments) != {p.GetNumber() for p in fresh.Pads() if p.GetNumber()}:
            raise ValueError(f"footprint pad set changed: {ref}")
        for pad in fresh.Pads():
            if pad.GetNumber():
                pad.SetNet(assignments[pad.GetNumber()])
        fresh.SetReference(ref)
        fresh.SetFPID(old.GetFPID())
        board.Remove(old)
        board.Add(fresh)
        footprints[ref] = fresh
    for ref, config in layout["components"].items():
        fp = footprints[ref]
        part = parts[config["part"]]
        x, y, angle = config["at_mm"]
        fp.SetOrientationDegrees(angle)
        fp.SetPosition(pcb.VECTOR2I(pcb.FromMM(x), pcb.FromMM(y)))
        fp.SetValue(part["value"])
        fp.Value().SetVisible(False)
        fp.Reference().SetTextSize(pcb.VECTOR2I(pcb.FromMM(1), pcb.FromMM(1)))
        fp.Reference().SetTextThickness(pcb.FromMM(0.12))
        # KiCad's stock references on tiny passives can overlap neighbours; placement remains reviewable.
        fp.Reference().SetPosition(pcb.VECTOR2I(pcb.FromMM(x), pcb.FromMM(y - 2)))
        fp.Reference().SetTextAngle(pcb.EDA_ANGLE(0, pcb.DEGREES_T))
        if ref.startswith(("R", "C")):
            fp.Reference().SetLayer(pcb.F_Fab)
        else:
            fp.Reference().SetPosition(pcb.VECTOR2I(pcb.FromMM(x - 3.5), pcb.FromMM(y)))
        if "reference_at_mm" in config:
            rx, ry = config["reference_at_mm"]
            fp.Reference().SetPosition(pcb.VECTOR2I(pcb.FromMM(rx), pcb.FromMM(ry)))
        for model in fp.Models():
            model.m_Filename = model.m_Filename.replace(
                "${KICAD10_3DMODEL_DIR}",
                str(
                    Path(
                        ".tools/KiCad/KiCad.app/Contents/SharedSupport/3dmodels"
                    ).resolve()
                ),
            )
    # Notched outline removes FR4/copper beneath the antenna.
    width, height = layout["size_mm"]
    x0, y0, x1, _ = layout["antenna_cutout_mm"]
    outline = [
        (0, 0),
        (width, 0),
        (width, height),
        (x1, height),
        (x1, y0),
        (x0, y0),
        (x0, height),
        (0, height),
    ]
    for a, b in zip(outline, outline[1:] + outline[:1]):
        line = pcb.PCB_SHAPE(board)
        line.SetShape(pcb.SHAPE_T_SEGMENT)
        line.SetLayer(pcb.Edge_Cuts)
        line.SetWidth(pcb.FromMM(0.05))
        line.SetStart(pcb.VECTOR2I(pcb.FromMM(a[0]), pcb.FromMM(a[1])))
        line.SetEnd(pcb.VECTOR2I(pcb.FromMM(b[0]), pcb.FromMM(b[1])))
        board.Add(line)
    lib = str(
        Path(
            ".tools/KiCad/KiCad.app/Contents/SharedSupport/footprints/MountingHole.pretty"
        ).resolve()
    )
    for i, (x, y) in enumerate(layout["mounting_holes_mm"], 1):
        fp = pcb.FootprintLoad(lib, "MountingHole_2.7mm_M2.5")
        fp.SetReference("H" + str(i))
        fp.SetPosition(pcb.VECTOR2I(pcb.FromMM(x), pcb.FromMM(y)))
        fp.Reference().SetVisible(False)
        fp.Value().SetVisible(False)
        board.Add(fp)
    settings = board.GetDesignSettings()
    settings.SetBoardThickness(pcb.FromMM(layout["thickness_mm"]))
    settings.m_MinClearance = pcb.FromMM(0.2)
    settings.m_TrackMinWidth = pcb.FromMM(0.2)
    settings.m_MinThroughDrill = pcb.FromMM(0.3)
    settings.m_ViasMinSize = pcb.FromMM(0.6)
    board.SetCopperLayerCount(2)
    for route in layout.get("preroutes", []):
        pts = route["points_mm"]
        net = board.FindNet(route["net"])
        if net is None:
            raise ValueError("preroute references absent net")
        for a, b in zip(pts, pts[1:]):
            track = pcb.PCB_TRACK(board)
            track.SetNet(net)
            track.SetLayer(board.GetLayerID(route["layer"]))
            track.SetWidth(pcb.FromMM(route["width_mm"]))
            track.SetStart(pcb.VECTOR2I(pcb.FromMM(a[0]), pcb.FromMM(a[1])))
            track.SetEnd(pcb.VECTOR2I(pcb.FromMM(b[0]), pcb.FromMM(b[1])))
            board.Add(track)
    for config in layout.get("preroute_vias", []):
        x, y = config["at_mm"]
        via = pcb.PCB_VIA(board)
        via.SetNet(board.FindNet(config["net"]))
        via.SetPosition(pcb.VECTOR2I(pcb.FromMM(x), pcb.FromMM(y)))
        via.SetWidth(pcb.FromMM(config["diameter_mm"]))
        via.SetDrill(pcb.FromMM(config["drill_mm"]))
        via.SetViaType(pcb.VIATYPE_THROUGH)
        via.SetLayerPair(pcb.F_Cu, pcb.B_Cu)
        board.Add(via)
    pcb.SaveBoard(str(root / "carrier-unrouted.kicad_pcb"), board)
    (root / "fp-lib-table").write_text(
        '(fp_lib_table (version 7) (lib (name "parts") (type "KiCad") (uri "${KIPRJMOD}/parts") (options "") (descr "Vendored KiCad library footprints")) (lib (name "MountingHole") (type "KiCad") (uri "'
        + lib
        + '") (options "") (descr "KiCad mounting holes")))'
    )
    if not pcb.ExportSpecctraDSN(board, str(root / "carrier.dsn")):
        raise RuntimeError("DSN export failed")
    nets = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNumber() and pad.GetNetname():
                nets.setdefault(pad.GetNetname(), []).append(
                    f"{fp.GetReference()}.{pad.GetNumber()}"
                )
    (root / "nets.json").write_text(json.dumps(nets, indent=2) + "\n")
    bom = []
    for ref, c in layout["components"].items():
        bom.append(
            dict(
                reference=ref, **parts[c["part"]], quantity=1, price_status="not_quoted"
            )
        )
    (root / "bom.json").write_text(json.dumps(bom, indent=2) + "\n")
    print(
        f"{len(footprints)} populated footprints, {len(nets)} named nets; routing pending"
    )


if __name__ == "__main__":
    main()
