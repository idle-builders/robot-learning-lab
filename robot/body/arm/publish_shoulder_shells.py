"""Package and publish the parametric shoulder shells (Idle T-tph7xb) on the robot design dashboard.

The upper-arm covers and torso shells around the shoulder U connectors, built by robot/body/arm/shoulder_shells.py apply:
the cover is a few smooth parametric functions (superellipse rings, spline control values) that encloses the green U, its top
is trimmed about the roll axis, and the torso shells get one round opening about each pitch axis (owner 2026-11-22).

  python3 infra/tools/memguard.py --limit-gb 6 --timeout-s 2400 -- .venv/bin/python -B robot/body/arm/publish_shoulder_shells.py package
  .venv/bin/python -B robot/body/arm/publish_shoulder_shells.py record
  .venv/bin/python infra/dashboard/results_hub.py publish artifacts/robot/body/arm/shoulder-clevis/shells/publication.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ART = ROOT / "artifacts/robot/body/arm/shoulder-clevis/shells"
MANIFEST = ART / "assembly-parts-shoulder-u-shells.json"
REPORT = ART / "report.json"
VIEWS = ROOT / ".cache/work/shoulder-views"
RENDERS = ("shoulder-shells-before-after.png", "shoulder-shells-range-poses.png", "shoulder-shells-cover-mount.png")
STUDY = "shoulder-shells-parametric"
GLB = ART / "assembly-shoulder-shells.glb"


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def package():
    (ART / "renders").mkdir(parents=True, exist_ok=True)
    for n in RENDERS:
        if (VIEWS / n).exists():
            shutil.copy2(VIEWS / n, ART / "renders" / n)
    sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    from package import scene  # motor-refit packager: one named node per part, metres
    scene(json.loads(MANIFEST.read_text())).export(str(GLB))
    print("packaged", rel(ART))


def _params():
    """The parameter tables of shoulder_shells.py, as written in the script."""
    sys.path.insert(0, str(HERE))
    import shoulder_shells as S
    out = {}
    for k in ("COVER", "CAP", "BUMPS", "TRIM", "TORSO_OPENING", "WALL", "GAP", "TOP_OFFSET_LOOK"):
        if hasattr(S, k):
            out[k] = getattr(S, k)
    return json.loads(json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def record():
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    renders = [ART / "renders" / n for n in RENDERS if (ART / "renders" / n).exists()]
    assets = [dict(id="assembly", title="Complete body with the parametric shoulder shells", kind="model", path=rel(GLB)),
              dict(id="parts", title="Named parts and joints (selected body)", kind="data", path=rel(MANIFEST)),
              dict(id="report", title="Shell build report (growth, openings, clearances)", kind="data", path=rel(REPORT))]
    assets += [dict(id=p.stem, title=p.stem.replace("-", " "), kind="image", path=rel(p)) for p in renders]
    params = _params()
    obs = ("The upper-arm cover is now a few gentle parametric functions (superellipse rings whose half-widths and exponents follow "
           "a handful of spline control values along the arm; fit to today's cover rms 0.71 mm) and encloses the green U: two smooth C2 "
           "swellings, max outward growth 6.3 mm at the back-inboard corner below the roll servo, elsewhere <= 1.6 mm. Its top is trimmed "
           "by two planes about the roll axis (10 mm corner round); the torso shells get one round opening per side about the pitch axis "
           "(R 34.6 mm, 1.5 mm rounded edge). No opening follows the motion sweep.")
    meas = ("Blue U to cover >= 4.3 mm over roll -15..95 (point sweep 4.54 mm); blue U to torso shells >= 4.1 mm over pitch -60..150; "
            "green U, its sleeves and screws fully inside the cover (min 1.71 mm to the cover material); cover 12092 mm3 per side "
            "(today 15336); torso opening removes about 2023 mm3 (front) and 2141 mm3 (back) per side. Cover mount: two bosses on the "
            "green U with a pressed sleeve and a captive M3 nut, heads seated, nut engagement 2.25 / 2.5 mm.")
    lim = ("At the roll -15 end the cover's lower inboard wall still enters the torso side (95 + 23 mm3; today's body 293 + 224 mm3): a "
           "roll-limit or torso-shape question. The torso opening leaves about 4 mm of shell above it at the shoulder top. The trim shows "
           "the upper roll servo (the blue U rolls over it at +95). Cover pads are clamped plastic without a sleeve. Mesh-defined shells, "
           "nothing printed.")
    pub = dict(id=STUDY, study_id=STUDY, title="Shoulder · parametric shells around the U connectors",
               source=["robot/body/arm/shoulder_shells.py", "robot/body/arm/publish_shoulder_shells.py"],
               dashboard="design", workspace="Body", robot_sections=["Shoulder", "Torso / shell"],
               observation=obs, measurements=meas, limitations=lim,
               provenance=["Idle T-tph7xb", "owner decisions 2026-11-22: cover over the green U, no sweep-following openings, gentle parametric functions"],
               assets=assets, clips=[],
               design=dict(id=STUDY, title="Parametric shoulder shells", summary="Smooth parametric upper-arm cover enclosing the green U, simple openings about the joint axes.",
                           visual_kind="native_CAD_assembly", maturity="Exploratory CAD; physical behavior unmeasured",
                           image_path=rel(renders[0]) if renders else None,
                           cad=dict(model_units="m", views=[dict(label="Complete body with the parametric shoulder shells", model_path=rel(GLB))],
                                    parts_path=rel(MANIFEST), report_path=rel(REPORT), hide_shell_toggle=True, show_shells=True,
                                    motion_note="Joint sliders are inspection commands over the manifest ranges, not qualified travel."),
                           evidence=[dict(id="parameters", title="Shape parameters (shoulder_shells.py)", kind="analysis", value=params)]))
    (ART / "publication.json").write_text(json.dumps(pub, indent=1, default=str) + "\n")
    print(rel(ART / "publication.json"))


if __name__ == "__main__":
    {"package": package, "record": record}[sys.argv[1]]()
