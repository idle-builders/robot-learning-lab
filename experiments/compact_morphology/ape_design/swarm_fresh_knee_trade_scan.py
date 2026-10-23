"""Second-round trade scan for the fresh barrel knee, using the new shell.

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_fresh_knee_trade_scan.py
The tested grid is finite; this script rejects this configuration, not every
possible knee mechanism or all off-axis transmissions.
"""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import build123d as bd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from swarm_fresh_knee_coupon import ROOT,OUT,AXIS,overlap,turn


def alternative_barrel(start_deg):
    aa=list(range(start_deg,261,4))
    if aa[-1]!=260: aa.append(260)
    p=lambda ry,rz,a:(26,3+ry*math.cos(math.radians(a)),86+rz*math.sin(math.radians(a)))
    w=bd.Wire.make_polygon([p(17,7,a) for a in aa]+[p(15,5,a) for a in reversed(aa)],close=True)
    return bd.extrude(bd.Face(w),amount=32,dir=(1,0,0))


def main():
    sys.path.insert(0,str(ROOT/'experiments/sketch_ape_shell'))
    import build as study
    stations=[(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
              (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
    inner=study.loft([(z,x,y,rx-2,ry-2) for z,x,y,rx,ry in stations])
    moving=bd.import_step(OUT/'moving-shin-and-yoke.step')
    posed=turn(moving,120)
    motors=[]
    for z in (124,128,132,136):
        for y in (-12,-9,-6,-3,0,3):
            case=bd.Pos(39,y,z)*bd.Rot(0,-12,0)*bd.Box(35,24.72,45.22)
            motors.append({'case_center_y_mm':y,'case_center_z_mm':z,
                'nominal_thigh_interior_fit_pct':round(100*overlap(case,inner)/case.volume,2),
                'shin_vs_case_at_120_mm3':round(overlap(posed,case),2)})
    apertures=[]
    for start in (40,60,80,100,120,140):
        hood=alternative_barrel(start)
        apertures.append({'front_rigid_arc_deg':260-start,
                          'rear_opening_deg':360-(260-start),
                          'shin_vs_barrel_at_120_mm3':round(overlap(posed,hood),2)})
    clear=[m for m in motors if m['shin_vs_case_at_120_mm3']==0]
    report={'source_new_shell':'artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step',
            'source_moving':'artifacts/swarm-fresh-knee-new-shell-20261004/moving-shin-and-yoke.step',
            'motor':'Official-size Feetech ST-3250-C001 35x24.72x45.22 mm box, tilted -12 deg about Y',
            'motor_official_url':'https://www.feetechrc.com/en/562636.html',
            'motor_grid':motors,'barrel_aperture_scan':apertures,
            'best_fit_with_zero_120_motor_overlap_pct':max(m['nominal_thigh_interior_fit_pct'] for m in clear),
            'any_grid_motor_fit_at_least_95pct_and_zero_120_overlap':any(m['nominal_thigh_interior_fit_pct']>=95 and m['shin_vs_case_at_120_mm3']==0 for m in motors),
            'judgment':'Reject this barrel/high-thigh motor configuration for the requested 0-120 deg motion. The motor can clear the shin only by protruding far outside the new thigh cavity; removing the barrel clash needs a 200+ degree rear aperture that abandons the continuous side enclosure.',
            'limits':'Finite motor grid and 10-degree first-round motion sweep; these are geometric envelope tests, not proof that no other knee architecture is possible. Cable load, connector, fasteners, bearings, continuous clearance and hip/other-leg coupling remain unverified.'}
    (OUT/'trade_scan.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('best_fit_with_zero_120_motor_overlap_pct','any_grid_motor_fit_at_least_95pct_and_zero_120_overlap','judgment')},indent=2))
if __name__=='__main__':main()
