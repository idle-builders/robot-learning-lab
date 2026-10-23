"""Bounded exact-BRep location scan for the 500 mm rated knee cavity.

This only ranks neutral packages. It does not accept a moving joint or choose a
new anatomical axis without a cover and full-leg motion screen.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import build123d as b

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/"artifacts/rated-leg-skin-v2-20261005"
sys.path.insert(0,str(ROOT/"experiments/compact_morphology/ape_design/swarm_rated_leg_drive"))
from rated_leg_coupon import maybe_intersection, package
from front_belt_screen import capsule, cylinder_x, PITCH_RADIUS

SOURCE=ROOT/"artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"


def shapes(kx,ky,kz,dy):
    knee=(kx,ky,kz)
    drive=(kx,ky+dy,kz+math.sqrt(31.5**2-dy**2))
    motor=b.Pos(kx-17.5,drive[1],drive[2]+12.2)*b.Box(35,24.72,45.22)
    belt=capsule(PITCH_RADIUS+1.5,kx+3,kx+18,knee,drive)-capsule(PITCH_RADIUS-1,kx+2.9,kx+18.1,knee,drive)
    shaft=cylinder_x(4,kx-30,kx+26,ky,kz)
    return knee,drive,{"motor":motor,"belt":belt,"shaft":shaft}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source={x.label:x for x in b.import_step(str(SOURCE)).children}
    thigh=[source[k] for k in ("R_thigh_front","R_thigh_back")]
    cuff=source["R_knee_cuff"]
    neighbor={m["name"]:m for m in package(500/350) if m["name"] in ("hip_roll","ankle_pitch")}
    rows=[]
    for kx,ky,kz,dy in itertools.product((58,60,64,68),(-6,-2,2,6),(138,142,146),(-8,-4,0)):
        knee,drive,parts=shapes(kx,ky,kz,dy)
        clash={name:round(sum(maybe_intersection(shape,wall) for wall in thigh),3)
               for name,shape in parts.items()}
        cuff_hits={name:round(maybe_intersection(shape,cuff),3) for name,shape in parts.items()}
        neighbor_hits={name:round(sum(maybe_intersection(shape,q) for m in neighbor.values() for q in m["parts"].values()),3)
                       for name,shape in parts.items()}
        # Wall violation priority, then silhouette-preserving axis offset.
        score=sum(clash.values())+sum(cuff_hits.values())+10*sum(neighbor_hits.values())
        score+=20*(abs(kx-60)+abs(ky+2)+abs(kz-140))
        rows.append(dict(knee_mm=knee,motor_output_mm=drive,thigh_hits_mm3=clash,
                         cuff_hits_mm3=cuff_hits,neighbor_hits_mm3=neighbor_hits,score=round(score,3)))
    rows.sort(key=lambda x:x["score"])
    report={"source":"artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step; nominal 2 mm wall",
            "scope":"neutral package only; 4 x 4 x 3 x 3 candidate grid; no output support or moving cover in score",
            "candidate_count":len(rows),"ranked":rows}
    (OUT/"scan.json").write_text(json.dumps(report,indent=2)+"\n")
    for row in rows[:10]:print(row)


if __name__=="__main__":main()
