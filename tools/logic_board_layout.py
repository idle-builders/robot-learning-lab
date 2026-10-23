#!/usr/bin/env python3
"""Place and explicitly route the compiled logic supply using native KiCad."""

import json
from pathlib import Path
import sys
import pcbnew as pcb
import wx


def build(root):
    app = wx.App(False)  # noqa: F841
    layout = json.loads((root / "layout.json").read_text())
    output_name = layout.get("output_name", "logic-supply")
    definitions = json.loads((root / "parts.json").read_text())
    instances = json.loads((root / "components.json").read_text())
    board = pcb.LoadBoard(str(root / "layouts/default/default.kicad_pcb"))
    old_parts = {f.GetReference(): f for f in board.GetFootprints()}
    if set(old_parts) != set(instances) or set(instances) != set(
        layout["placement_mm"]
    ):
        raise ValueError("compiled/placement component set mismatch")
    native = Path(".tools/KiCad/KiCad.app/Contents/SharedSupport").resolve()
    pads = {}
    for ref, old in old_parts.items():
        part = definitions[instances[ref]]
        fresh = pcb.FOOTPRINT(
            pcb.FootprintLoad(str(root / "parts"), Path(part["footprint"]).stem)
        )
        assignments = {p.GetNumber(): p.GetNet() for p in old.Pads() if p.GetNumber()}
        if set(assignments) != {p.GetNumber() for p in fresh.Pads() if p.GetNumber()}:
            raise ValueError("footprint pad set changed: " + ref)
        for pad in fresh.Pads():
            if pad.GetNumber():
                pad.SetNet(assignments[pad.GetNumber()])
        fresh.SetReference(ref)
        fresh.SetValue(part["mpn"])
        fresh.SetFPID(old.GetFPID())
        fresh.Value().SetVisible(False)
        fresh.Reference().SetLayer(pcb.F_Fab)
        x, y, angle = layout["placement_mm"][ref]
        fresh.SetOrientationDegrees(angle)
        fresh.SetPosition(pcb.VECTOR2I(pcb.FromMM(x), pcb.FromMM(y)))
        for model in fresh.Models():
            model.m_Filename = model.m_Filename.replace(
                "${KICAD10_3DMODEL_DIR}", str(native / "3dmodels")
            ).replace("${KIPRJMOD}", str(root.resolve()))
        board.Remove(old)
        board.Add(fresh)
        for pad in fresh.Pads():
            if pad.GetNumber():
                pads[ref + "." + pad.GetNumber()] = pad

    def point(value, net=None):
        if isinstance(value, str):
            pad = pads[value]
            if net and pad.GetNetname() != net:
                raise ValueError("route/pad net mismatch: " + value)
            return pad.GetPosition()
        return pcb.VECTOR2I(pcb.FromMM(value[0]), pcb.FromMM(value[1]))

    width, height = layout["size_mm"]
    corners = [[0, 0], [width, 0], [width, height], [0, height]]
    for a, b in zip(corners, corners[1:] + corners[:1]):
        edge = pcb.PCB_SHAPE(board)
        edge.SetShape(pcb.SHAPE_T_SEGMENT)
        edge.SetLayer(pcb.Edge_Cuts)
        edge.SetWidth(pcb.FromMM(0.05))
        edge.SetStart(point(a))
        edge.SetEnd(point(b))
        board.Add(edge)
    mountlib = native / "footprints/MountingHole.pretty"
    for i, xy in enumerate(layout["mounting_holes_mm"], 1):
        fp = pcb.FootprintLoad(str(mountlib), "MountingHole_2.7mm_M2.5")
        fp.SetReference("H" + str(i))
        fp.SetPosition(point(xy))
        fp.Reference().SetVisible(False)
        fp.Value().SetVisible(False)
        board.Add(fp)
    settings = board.GetDesignSettings()
    settings.SetBoardThickness(pcb.FromMM(layout["thickness_mm"]))
    settings.m_MinClearance = pcb.FromMM(layout.get("clearance_mm", 0.2))
    settings.m_TrackMinWidth = pcb.FromMM(0.15)
    settings.m_MinThroughDrill = pcb.FromMM(0.3)
    settings.m_ViasMinSize = pcb.FromMM(0.6)
    board.SetCopperLayerCount(2)
    for label in layout.get("labels", []):
        text = pcb.PCB_TEXT(board)
        text.SetText(label["text"])
        text.SetLayer(pcb.F_SilkS)
        text.SetPosition(point(label["at_mm"]))
        text.SetTextSize(pcb.VECTOR2I(pcb.FromMM(0.9), pcb.FromMM(0.9)))
        text.SetTextThickness(pcb.FromMM(0.12))
        board.Add(text)
    route_metrics = []
    for route in layout["routes"]:
        net = route["net"]
        pts = [point(v, net) for v in route["points"]]
        length = 0
        if board.FindNet(net) is None:
            raise ValueError("unknown route net")
        for a, b in zip(pts, pts[1:]):
            t = pcb.PCB_TRACK(board)
            t.SetNet(board.FindNet(net))
            t.SetLayer(board.GetLayerID(route.get("layer", "F.Cu")))
            t.SetWidth(pcb.FromMM(route["width_mm"]))
            t.SetStart(a)
            t.SetEnd(b)
            board.Add(t)
            length += ((pcb.ToMM(a.x - b.x)) ** 2 + (pcb.ToMM(a.y - b.y)) ** 2) ** 0.5
        route_metrics.append(
            dict(
                name=route["name"],
                net=net,
                length_mm=length,
                width_mm=route["width_mm"],
                layer=route.get("layer", "F.Cu"),
            )
        )
    for entry in layout["vias"]:
        v = pcb.PCB_VIA(board)
        v.SetNet(board.FindNet(entry["net"]))
        v.SetPosition(point(entry["at_mm"]))
        v.SetWidth(pcb.FromMM(0.6))
        v.SetDrill(pcb.FromMM(0.3))
        v.SetViaType(pcb.VIATYPE_THROUGH)
        v.SetLayerPair(pcb.F_Cu, pcb.B_Cu)
        board.Add(v)
    zone = pcb.ZONE(board)
    zone.SetNet(board.FindNet("GND"))
    zone.SetLayer(pcb.B_Cu)
    zone.SetLocalClearance(pcb.FromMM(layout.get("clearance_mm", 0.2)))
    zone.SetPadConnection(pcb.ZONE_CONNECTION_FULL)
    outline = zone.Outline()
    outline.NewOutline()
    for xy in [
        [0.4, 0.4],
        [width - 0.4, 0.4],
        [width - 0.4, height - 0.4],
        [0.4, height - 0.4],
    ]:
        outline.Append(int(pcb.FromMM(xy[0])), int(pcb.FromMM(xy[1])))
    board.Add(zone)
    board.BuildConnectivity()
    pcb.ZONE_FILLER(board).Fill(board.Zones())
    pcb.SaveBoard(str(root / (output_name + ".kicad_pcb")), board)
    records = {
        k: dict(
            net=p.GetNetname(),
            xy_mm=[pcb.ToMM(p.GetPosition().x), pcb.ToMM(p.GetPosition().y)],
        )
        for k, p in pads.items()
    }
    interface = dict(
        schema_version=1,
        scope=layout.get(
            "scope",
            "candidate buck board; no enclosure placement, thermal or EMI qualification",
        ),
        coordinate_system="PCB millimetres: upper-left origin, x right, y down",
        step_frame=dict(
            xy_from_pcb=[[1, 0], [0, -1]],
            datum="substrate bottom z=0; components toward positive z",
        ),
        outline_mm=layout["size_mm"],
        thickness_mm=layout["thickness_mm"],
        mounting_holes_mm=layout["mounting_holes_mm"],
        mounting_hole_diameter_mm=2.7,
        pads=records,
        routes=route_metrics,
        ground_plane="B.Cu",
        track_via_count=len(list(board.GetTracks())),
        missing_3d_models=[
            ref
            for ref in instances
            if not len(
                list(
                    next(
                        f for f in board.GetFootprints() if f.GetReference() == ref
                    ).Models()
                )
            )
        ],
    )
    (root / "board-interface.json").write_text(json.dumps(interface, indent=2) + "\n")
    # Preserve the correct board, then inject a deliberate input short as a detector test.
    short = pcb.PCB_TRACK(board)
    short.SetNet(pads["J1.1"].GetNet())
    short.SetWidth(pcb.FromMM(0.3))
    short.SetLayer(pcb.F_Cu)
    short.SetStart(pads["J1.1"].GetPosition())
    short.SetEnd(pads["J1.2"].GetPosition())
    board.Add(short)
    pcb.SaveBoard(str(root / (output_name + "-shorted.kicad_pcb")), board)
    # KiCad netclass clearance overrides the board minimum. Keep both aligned
    # when a circuit explicitly allocates a different manufacturing rule.
    if "clearance_mm" in layout:
        for name in (output_name, output_name + "-shorted"):
            project = root / (name + ".kicad_pro")
            data = json.loads(project.read_text())
            for netclass in data["net_settings"]["classes"]:
                netclass["clearance"] = layout["clearance_mm"]
            project.write_text(json.dumps(data, indent=2) + "\n")
    (root / "fp-lib-table").write_text(
        '(fp_lib_table (version 7) (lib (name "parts") (type "KiCad") (uri "${KIPRJMOD}/parts") (options "") (descr "Vendored footprints")) (lib (name "MountingHole") (type "KiCad") (uri "'
        + str(mountlib)
        + '") (options "") (descr "Mounts")))'
    )
    print("Explicit routes, ground plane and mechanical interface exported")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
