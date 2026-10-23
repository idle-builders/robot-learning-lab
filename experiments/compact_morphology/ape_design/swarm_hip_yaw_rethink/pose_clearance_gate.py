"""Bounded exact 500 mm static-clearance and coupled yaw/squat shell pose gate."""
import json
from pathlib import Path
import build123d as b
import hip_yaw_rethink as h

K=500/350; OUT=h.OUT/'pose-clearance.json'
def turn(shape,axis,deg,p):
    r={'x':b.Rot(deg,0,0),'y':b.Rot(0,deg,0),'z':b.Rot(0,0,deg)}[axis]
    return b.Pos(*p)*r*b.Pos(*(-v for v in p))*shape
rows=[]
for wall in (1,2):
    st=bld=h.Study(500,wall,.65);st.build(); parts={x['name']:x['shape'] for x in st.parts}
    skin={n:s for n,s in parts.items() if n in {'torso_front','torso_back','pelvis_front','pelvis_back'}}
    common=[];motors=[];fixed=[]
    for side in (-1,1):
        hip=(side*h.HIP_X*K,0,h.HIP_Z*K)
        m=h.rsf((side*13.5,-10,h.HIP_Z),K);motors.append((f'{side:+} RSF body',m));common.append((f'{side:+} RSF body',m))
        shaft=h.cyl('z',3,16,(hip[0],0,hip[2]-8));fixed.append((f'{side:+} output shaft',shaft))
        for dz in (-10,-2):
            ring=h.cyl('z',6.5,3.5,(hip[0],0,hip[2]+dz))-h.cyl('z',3,4,(hip[0],0,hip[2]+dz))
            fixed.append((f'{side:+} W618/6 envelope z{dz}',ring))
        flange=h.cyl('z',5,4,(hip[0],0,hip[2]-14));fixed.append((f'{side:+} yaw flange',flange))
        belt=h.slot_belt((side*13.5*K,-10*K),(hip[0],0),hip[2]-2,2,3);fixed.append((f'{side:+} belt envelope',belt))
    fixed.append(('fixed cross bridge',h.box((0,0,h.HIP_Z*K-10),(66,8,3.5))))
    for side in (-1,1):fixed.append((f'{side:+} bearing cheek',h.box((side*32*K,0,h.HIP_Z*K-5),(4,8,8))))
    fixed += [('battery',h.box((0,0,200*K),(60,30,30.5))),('power-service reserve',h.box((0,0,225*K),(25,20,14)))]
    allitems=common+fixed
    static=[]
    for n,shape in allitems:
        shell_hits=[{'shell':sn,'mm3':h.overlaps(shape,ss)} for sn,ss in skin.items() if h.overlaps(shape,ss)>.001]
        distances=[shape.distance_to(ss) for ss in skin.values()]
        static.append({'component':n,'shell_hits':shell_hits,'minimum_exact_skin_surface_gap_mm':round(min(distances),4)})
    # Source thigh shells are rotated as complete outer skin around the hip yaw
    # datum; this tests shell motion only, not bearing force or a full leg chain.
    moving=[]
    poses=[('neutral',0,0),('yaw_left',20,0),('yaw_right',-20,0),('coupled_stride',20,-35),('squat',0,-45)]
    for side,link in [(-1,'L'),(1,'R')]:
        pivot=(side*h.HIP_X*K,0,h.HIP_Z*K)
        halves=[parts[f'{link}_thigh_front'],parts[f'{link}_thigh_back']]
        for name,yaw,pitch in poses:
            moved=[turn(turn(s,'z',yaw,pivot),'x',pitch,pivot) for s in halves]
            hits=[]
            for i,m in enumerate(moved):
                for sn,ss in skin.items():
                    v=h.overlaps(m,ss)
                    if v>.01:hits.append({'moving_half':i,'fixed_shell':sn,'mm3':round(v,3)})
                for n,c in common:
                    v=h.overlaps(m,c)
                    if v>.01:hits.append({'moving_half':i,'fixed_component':n,'mm3':round(v,3)})
            moving.append({'side':link,'pose':name,'yaw_deg':yaw,'hip_pitch_deg':pitch,'exact_intersections':hits})
    rows.append({'height_mm':500,'wall_mm':wall,'remote_round_layout_static':static,'thigh_shell_motion':moving})
OUT.write_text(json.dumps({'note':'Exact BRep; thigh movement is a skin-only proxy, no full leg or force qualification.','rows':rows},indent=2))
for r in rows:
 print('wall',wall,'static shell hits',sum(len(x['shell_hits']) for x in static),'min wall gap',min(x['minimum_exact_skin_surface_gap_mm'] for x in static))
 print('motion overlap poses',[(x['side'],x['pose'],sum(i['mm3'] for i in x['exact_intersections'])) for x in moving if x['exact_intersections']])
