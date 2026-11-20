# Ape shell — sketch study

The supplied image in `reference/sketch.png` is the only form reference used for
this study. This folder is independent of the repository's other robot designs.

The committed study contains the parametric source, reference sketch, rendered
previews, viewer template and pinned dependencies, and verification record.
Generated CAD exports and the bundled viewer live in
[`artifacts/sketch-ape-shell/`](../../artifacts/sketch-ape-shell/), which is ignored
by Git like the repository's other generated artifacts. Rebuild them with the
commands below after a fresh checkout.

![CAD assembly](previews/assembly.png)

## Open the model

- [`preview.html`](../../artifacts/sketch-ape-shell/preview.html) is the generated,
  self-contained interactive viewer. Open it in a browser.
  Orbit, switch between front/side/rear views, explode the assembly, hide groups,
  or cut through the centre to inspect the hollow panels.
- [350 mm STEP](../../artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step)
  is the named, coloured B-rep assembly.
- [500 mm STEP](../../artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step)
  is the independently regenerated 500 mm assembly.
- Each export folder contains 67 individually named STL parts in assembly coordinates
  and a `manifest.json` with dimensions, volumes, colours, and solid validity.
- `build.py` is the editable parametric CAD source. STEP retains the separate
  solids and assembly names; it does **not** carry the Python parameter history.
- `viewer.html` is the source template; use the generated `preview.html` to view
  the model. The source template alone does not contain CAD geometry.

## Rebuild the delivered study

From the repository root, using its existing Python environment:

```sh
.venv/bin/python robot/shell/sketch/build.py --height 350
.venv/bin/python robot/shell/sketch/build.py --height 500
.venv/bin/python robot/shell/sketch/bundle_viewer.py
.venv/bin/python robot/shell/sketch/verify_exports.py
```

For a separate environment, install this folder's `requirements.txt` first.
Exports go to `artifacts/sketch-ape-shell/exports/<height>mm/`. The bundler writes
`artifacts/sketch-ape-shell/preview.html`; the verification script updates the
small `verification.json` record in this source folder.

## Resize / edit

From the repository root:

```sh
.venv/bin/python robot/shell/sketch/build.py --height 425 --wall 2 --gap 0.65
```

`--height` accepts any value from **350 to 500 mm**. It scales all external form
proportions from the 350 mm master. `--wall` is the independent nominal shell
thickness in millimetres, and `--gap` is the front/back panel separation in
millimetres. Defaults: 350 mm height, 2 mm nominal wall, 0.65 mm split gap.
Use `--output PATH` to choose a destination, or `--no-mesh` for STEP only.

The shape is defined by named, dimensioned elliptical loft stations, ellipsoids,
and spline curves in `build.py`. Edit these directly to adjust the silhouette.
Units are millimetres. X is left/right, Y is rearward, Z is up. The floor is Z=0.
The master crown is Z=350. L/R part names refer to the front drawing's sides.

The viewer's height slider is a uniform visual preview. Regenerate CAD with the
command above to retain the independent wall thickness at a new height.

## Assembly breakdown

| Group | Parts |
| --- | ---: |
| Torso front/back | 2 |
| Pelvis front/back | 2 |
| Cranium front/back, face mask, muzzle, eye bezels and lenses | 8 |
| Upper arms and forearms, front/back on each side | 8 |
| Palm front/back, four finger caps and thumb on each side | 14 |
| Thighs and shins, front/back on each side | 8 |
| Dorsal foot, three toes and sole on each side | 10 |
| Neck, shoulders, elbows, wrists, knees and ankles | 11 |
| Indicator, mouth and nostril inlays | 4 |
| **Total** | **67** |

The design follows the sketch's broad rounded torso, small domed head with a
warm face inset and projecting muzzle, long arms, short bowed legs, broad
three-toed feet, and softly curled fingers. Front/back splits make the body
and major limb cavities accessible. Joint-coloured pieces are cosmetic covers.

This is a shell form and part-split study. It does not include motors, an internal
frame, fastening features, locating tabs, or a joint-motion/clearance design.
Wall thickness is nominal: subtracting smaller elliptical sections gives a
variable surface-normal thickness on steep curves and near caps. Print process,
minimum wall thickness, fits, and attachment details still need to be resolved
before manufacture.

## Verification and reproducibility

Both delivered heights were rebuilt, each part checked as one valid positive-volume
B-rep solid, and the resulting assembly height measured. The exported STEP files
were reimported for further solid-count, height, and volume checks. All delivered
STL parts were also reopened and checked for watertightness. Small tessellation
seam duplicates are welded during export; exact CAD geometry remains in STEP. Front, side,
three-quarter, and cutaway views were inspected from the actual CAD tessellation.
The previews are CAD renders, not generated concept images.

Python dependencies are the repository environment's `build123d==0.11.1`, NumPy
and trimesh. The offline viewer bundles Three.js 0.180.0; its MIT license is in
`vendor/LICENSE.three`. Run `bundle_viewer.py` after rebuilding the 350 mm model
to refresh the generated `preview.html`.
