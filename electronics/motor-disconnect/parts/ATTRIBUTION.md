# Footprint provenance

Footprints come unchanged from the installed official KiCad 10.0.6 library. See LICENSE-KiCad.md for the library license and design exception. The capacitor/resistor files and selected 100 nF / 100 kΩ candidates reuse the existing logic-supply design.

The LFPAK56 library maps the mounting-base drain to pad 5, and source leads to pads 1–3. The controller WSON footprint maps the exposed pad to 13. This candidate deliberately leaves that pad electrically unconnected, following the LM7480-Q1 pin table. Neither a library match nor successful compilation establishes land-pattern or assembly qualification.
