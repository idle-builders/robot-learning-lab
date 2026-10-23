Standard capacitor, resistor, connector and TSOT footprints were copied unmodified from the locally installed KiCad 10.0.6 official libraries. See LICENSE-KiCad.md.

L_Wuerth_7447714100.kicad_mod is a project-authored footprint based on the manufacturer datasheet page 1, rotated 90 degrees: pad centers at X = ±4.5 mm, pad size 2 × 3.5 mm, 7 mm inner gap and 11 mm outer span. The nominal body is 10 × 10 mm; maximum body dimensions are 10.3 × 10.5 × 5 mm. Source: https://www.we-online.com/components/products/datasheet/7447714100.pdf

The unused generic L_Wuerth_WE-PD-Typ-M-Typ-S footprint is retained for traceability but does not fit this selected larger inductor. Package compatibility, assembly clearance and switch-mode PCB layout still require review.

The board builder also generates a conservative 10.5 × 10.5 × 5 mm inductor body STEP. This is a project-authored maximum-size envelope, not vendor CAD.

The XH output-connector footprint is also copied from the installed KiCad 10.0.6 Connector_JST library: `JST_XH_B2B-XH-A_1x02_P2.50mm_Vertical.kicad_mod`. It remains under the accompanying KiCad library license. Its standard STEP is used as candidate geometry, not a tolerance-qualified vendor model.
