"""Small placement screen for an ST3250/123-3MGT-15 35 cm source-shell knee."""
from __future__ import annotations
import json, math
import build123d as bd
from swarm_belt_knee_coupon import SOURCE, box, cyl_x, compound, overlap, volume

R = 60/(2*math.pi)

def capsule(radius, x0, x1, knee, drive):
    dy,dz=drive[1]-knee[1],drive[2]-knee[2]
    angle=math.degrees(math.atan2(-dy,dz))
    bar=bd.Pos((x0+x1)/2,(knee[1]+drive[1])/2,(knee[2]+drive[2])/2)*bd.Rot(angle,0,0)*bd.Box(x1-x0,2*radius,31.5)
    return cyl_x(radius,x0,x1,knee[1],knee[2]).fuse(cyl_x(radius,x0,x1,drive[1],drive[2]),bar)

def turn(shape,knee,a):
    return bd.Pos(*knee)*bd.Rot(a,0,0)*bd.Pos(*(-v for v in knee))*shape

def scan():
    p={v.label:v for v in bd.import_step(SOURCE).children}
    thigh=compound([p['R_thigh_front'],p['R_thigh_back']])
    cuff=p['R_knee_cuff']
    shin=compound([p['R_shin_front'],p['R_shin_back']])
    foot=compound([p[f'R_{n}'] for n in ('foot_dorsal','toe_1','toe_2','toe_3','sole')])
    moving=compound([shin,foot])
    source_outer=bd.Solid.make_loft([bd.Wire.make_ellipse(rx,ry,bd.Plane(origin=(x,y,z)))
        for z,x,y,rx,ry in ((87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),(145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10))])
    rows=[]
    for x,y,z,dy in ((42,18,105,-27),(43,18,105,-27),
                     (44,18,105,-27),(42,20,105,-27),
                     (42,18,108,-27),(42,18,103,-27)):
        knee=(x,y,z)
        drive=(x,y+dy,z+math.sqrt(31.5**2-dy**2))
        # Rotate the same 35 x 24.72 x 45.22 case around its X output shaft:
        # 45.22 now runs fore-aft, output 12.2 mm anterior of box centre.
        motor=box((x-8.5,drive[1]+12.2,drive[2]),(35,45.22,24.72))
        belt=capsule(R+1.5,x+9,x+24,knee,drive)-capsule(R-1,x+8.9,x+24.1,knee,drive)
        pulley=cyl_x(9,x+9,x+24,knee[1],knee[2])
        rows.append(dict(knee=knee,drive=drive,motor_outside_source_outer_mm3=round(volume(motor-source_outer),2),
                         belt_outside_source_outer_mm3=round(volume(belt-source_outer),2),
                         pulley_outside_source_outer_mm3=round(volume(pulley-source_outer),2),
                         motor_original_skin_overlap_mm3=round(overlap(motor,thigh),2),
                         belt_original_skin_overlap_mm3=round(overlap(belt,thigh),2),
                         shin_motor_overlap_mm3={a:round(overlap(turn(moving,knee,a),motor),2) for a in (0,60,120)},
                         shin_belt_overlap_mm3={a:round(overlap(turn(moving,knee,a),belt),2) for a in (0,60,120)},
                         shin_source_fixed_overlap_mm3={a:round(overlap(turn(moving,knee,a),compound([thigh,cuff])),2) for a in (0,60,120)}))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':scan()
