# KingKong TS20 — recovered native geometry and partial shape mapping

Idle **T-enxahn**, 13 October 2026. This additive record follows the [Jumper reference audit](../../references/kingkong-jumper/README.md); its earlier incomplete download remains historical evidence. Both TS20-50 and TS20-100 now refer to the manufacturer's **complete shared STEP**. No ape architecture is selected.

The [official STEP](https://kingkong.tech/files/servo/TS20-50-TS20-100.step) was recovered in four HTTP byte ranges: 1,199,645 bytes, SHA256 `0dae0856425a08bab4114a06f9a4fa27e2054db27f684227ecf6050575c9300d`. Response ranges and hashes are retained in `source/download.json`. The file declares millimetres. It imports without healing as **10 valid solids and 688 faces**: one large body plus nine small detail solids. This is not ten independently moving actuator components.

![Actual source surfaces](output/native-views.png)

The [native report](output/native-report.json) retains cylinder axes, bounds and hole identities. The [two shape records](output/shape-records.json) map both variants to the nominal `axial_annular` family, with the actual lobed mounting ears and source geometry kept separately. The source-output datum is established; the canonical stationary-case datum and internal moving/fixed solid partition remain unresolved, so this is a **partial parameter release**. Null parameters and service volumes are not zero. It does not yet alter the canonical 267-row catalog/map.

| Feature | Source-backed result | Limit |
|---|---|---|
| Body | Nominal Ø20 barrel; complete native axial extent 34.5000001 mm | Whole source extent includes output/rear details; it is not a released stationary-case length. |
| Fixed mounts | Four M2-6H through holes, PCD24, in 3 mm thick ears; nominal radial envelope Ø27 | Native circular hole cores are Ø1.6, not helical threads. Tool and mating structure absent. |
| Load mounts | Four M2-6H depth2 holes on PCD9; negative source-X end | Mount ownership comes from manufacturer drawing annotations, not articulated STEP metadata. |
| Bore | Nominal Ø1.7; the native R0.85 bore axis is offset 0.0309001 mm from the barrel axis | Do not silently replace it with a centered Ø1.7 hole. No wire routing or installed passage qualified. |
| Service | Maker specifies side cable exit | Connector mate, bend radius, fastening access, heat path and installed support remain unknown. |

The fixed-hole centers sit at source X=5.1661706…8.1661706 mm, 9.25…12.25 mm behind the load face X=−4.0838294 mm. Four load-hole axes lie on PCD9; their straight core portions are shorter than the drawing's 2 mm thread depth because of entrance/details. Face IDs and coordinates are preserved rather than converting hole-core length into a different engagement specification.

The bore distinction is checked with full-length native solid intersections. A **Ø1.6 coaxial cylinder** passes through the complete source with zero material intersection and 0.0190999 mm minimum nominal distance. A Ø1.7 cylinder on the *barrel* axis intersects 1.73340 mm³; a Ø2 positive control intersects 28.75060 mm³. This is a geometry result for these exact virtual probes, not a tolerance-qualified physical bore gauge.

The [GLB](output/TS20-source.glb) displays the same source surfaces in metres. Its local origin is the load mounting face, local +Z points outward along source −X; this display frame must not be mistaken for an independently established stationary-case datum. The manufacturer's drawing and the first reference audit remain authoritative for claimed ratings; the Jumper simulation motor remains unidentified.

`build.py --out /tmp/ts20-native-replay` verifies the source and reference pins and regenerates the report, partial shape records, native image and GLB. Rendering is a surface visualization, not collision simplification. Next integration needs a guarded new-ID catalog join, resolved case/output partition and source-qualified cable/service envelope; it must preserve both variants' different ratings.
