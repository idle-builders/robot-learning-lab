"""Matched opaque mesh-projection change from the approved ankle shell.

This is a visual-area cross-check, not a replacement for exact BRep volume.
Each rendered panel uses the same 4.5 pixel/mm orthographic scale and camera.
"""
from __future__ import annotations

import json

import numpy as np

from boot_collar_case import OUT, case_shapes, make_scene, panel

BACKGROUND=np.array((220,217,207),dtype=np.uint8)
PIXELS_PER_MM=4.5


def mask(scene,view):
    pixels=np.asarray(panel(scene,view))
    return np.any(pixels!=BACKGROUND,axis=2)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    case=case_shapes()
    rows=[]
    for angle in (0,-20,20):
        source=make_scene(case,angle,source=True)
        candidate=make_scene(case,angle)
        for view in ("front","side"):
            a=mask(source,view)
            b=mask(candidate,view)
            added=int(np.count_nonzero(b&~a))
            missing=int(np.count_nonzero(a&~b))
            rows.append(dict(pitch_deg=angle,view=view,
                source_projected_mm2=round(int(np.count_nonzero(a))/PIXELS_PER_MM**2,2),
                candidate_projected_mm2=round(int(np.count_nonzero(b))/PIXELS_PER_MM**2,2),
                added_pixels=added,missing_pixels=missing,
                added_projected_mm2=round(added/PIXELS_PER_MM**2,2),
                missing_projected_mm2=round(missing/PIXELS_PER_MM**2,2)))
    report=dict(method="matched opaque mesh projection; 4.5 px/mm, fixed camera",
                limitation="Pixel area measures visual change, not 3D shell clearance or volume.",
                rows=rows)
    (OUT/"silhouette-projection.json").write_text(json.dumps(report,indent=2)+"\n")
    for row in rows:print(row)


if __name__=="__main__":main()
