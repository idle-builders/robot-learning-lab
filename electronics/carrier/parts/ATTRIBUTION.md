# Footprint provenance

The `.kicad_mod` files originate from the KiCad 10.0.6 application footprint library, installed locally from the official KiCad release. The upstream collection is https://gitlab.com/kicad/libraries/kicad-footprints . See LICENSE-KiCad.md for CC BY-SA 4.0 and the electronic-design exception.

`RobotLab_Pico2W_SMD.kicad_mod` is derived from the retained `RaspberryPi_Pico_W_SMD.kicad_mod`. The footprint header name changes and two paste-only apertures at x=±8.89 mm, y=24.13 mm change from 3.8×2.2 to 3.8×1.5 mm. This avoids their overlap with the library's RF keepout beginning at y=25 mm. Copper, pin numbers and keepouts are unchanged. These two apertures have less paste than the original over-paste recommendation; solder volume and the module reflow process still require assembly review/validation. No manufactured result is claimed.

3D models are resolved from the same installed KiCad library during export. They are generic package geometry; notably the Pico W model is not an exact Pico 2 W component-level model.
