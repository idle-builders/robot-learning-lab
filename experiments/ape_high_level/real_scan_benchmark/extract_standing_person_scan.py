"""Keep the person, and discard the street, from one CC-BY photogrammetry scan.

This is an appearance asset screen. It does not create articulated motion or
human physical clearance, and its source metric scale is inferred visually.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "assets/real_indoor_scenes/scansceneverse/glbs/b555f1fbaef64752a7b4b42e04fd6e7d.glb"
OUTPUT = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/standing_person_isolated.glb"


def main() -> None:
    source = trimesh.load(SOURCE, process=False)
    parts = source.dump() if isinstance(source, trimesh.Scene) else [source]
    if len(parts) != 1 or parts[0].visual.kind != "texture":
        raise ValueError("Expected one textured photogrammetry mesh")
    components = parts[0].split(only_watertight=False)
    kept = []
    for component in components:
        vertices = np.asarray(component.vertices)
        if (vertices[:, 1].max() > .08 and
                np.max(np.abs(vertices[:, [0, 2]])) < .2):
            kept.append(component)
    if not kept:
        raise ValueError("No plausible person components")
    person = trimesh.util.concatenate(kept)
    if person.visual.kind != "texture":
        raise ValueError("Texture was lost during person extraction")
    bounds = person.bounds
    if not (.60 <= bounds[1, 1] - bounds[0, 1] <= .70):
        raise ValueError("Unexpected source person height")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    person.export(OUTPUT)
    provenance = json.loads(SOURCE.with_suffix(".source.json").read_text())
    manifest = {"schema": "ape.standing-photogrammetry-person-extraction.v1",
        "source_glb": str(SOURCE.relative_to(ROOT)),
        "source_url": provenance["source_url"],
        "license": provenance["license"],
        "author": provenance["author"],
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "output_glb": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "source_component_count": len(components),
        "kept_component_count": len(kept),
        "kept_face_count": len(person.faces),
        "source_height_m": float(bounds[1, 1] - bounds[0, 1]),
        "provisional_scale_to_person_metres": 2.5,
        "visual_screen_only": True,
        "extraction_rule": "retain textured connected components fully within 0.2 source x/z metres of the standing body and reaching above y=0.08"}
    OUTPUT.with_suffix(".source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
