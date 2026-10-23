"""Robot-specific removable shunt tray candidate. Torso-local millimetres."""
from pathlib import Path
import yaml
from build123d import Box, Cylinder, Pos, Rot, Polygon, RegularPolygon, extrude, import_step

def hex_nut(af,height,bore_radius=1.0):
    return extrude(RegularPolygon(af/2,6,major_radius=False),amount=height)-Pos(0,0,height/2)*Cylinder(bore_radius,height+.2)

def along_x(x,y,z,r,length):
    return Pos(x,y,z)*Rot(0,90,0)*Cylinder(r,length)

def geometry(path=Path('electronics/regen-mount.yaml')):
    c=yaml.safe_load(path.read_text());t=c['tray'];a=c['arches'];h=c['hardware']
    placement=Pos(*c['module_center_mm'])*Rot(*c['module_rotation_deg'])*Pos(*[-v for v in c['supplier_center_mm']])
    board_bottom=c['module_center_mm'][2]-c['supplier_center_mm'][2]
    m=yaml.safe_load(Path(c['module_mounts_source']).read_text())['interface']
    board_top=board_bottom+m['pcb_size_mm'][2]
    # The single-axis 180-degree placement maps supplier XY directly.
    centers=[[c['module_center_mm'][0]+c['supplier_center_mm'][0]-x,c['module_center_mm'][1]+c['supplier_center_mm'][1]-y] for x,y in m['mount_holes_mm']]
    x0,x1=t['x_limits_mm'];y0,y1=t['y_limits_mm'];floor=t['bottom_mm'];top=floor+t['thickness_mm']
    tray=Pos((x0+x1)/2,(y0+y1)/2,(floor+top)/2)*Box(x1-x0,y1-y0,top-floor)
    nut=hex_nut(h['nut_af_mm'],h['nut_height_mm'])
    parts={'module':placement*import_step(c['module_source'])}
    module_nuts=[]
    for i,(x,y) in enumerate(centers):
        collar_bottom=board_bottom-t['contact_collar_height_mm']
        tray+=Pos(x,y,(top+collar_bottom)/2)*Cylinder(t['lower_boss_radius_mm'],collar_bottom-top)
        tray+=Pos(x,y,(collar_bottom+board_bottom)/2)*Cylinder(t['contact_collar_radius_mm'],board_bottom-collar_bottom)
        zb=t['nut_pocket_bottom_mm'];zt=t['nut_pocket_top_mm']
        tray-=Pos(x,y,zb)*extrude(RegularPolygon(t['nut_pocket_af_mm']/2,6,major_radius=False),amount=zt-zb)
        direction=1 if y>0 else -1; travel=t['nut_insertion_travel_mm']
        tray-=Pos(x,y+direction*travel/2,(zb+zt)/2)*Box(t['nut_insertion_slot_width_mm'],travel,zt-zb)
        tray-=Pos(x,y,(board_bottom+t['blind_bore_bottom_mm'])/2)*Cylinder(t['screw_clearance_radius_mm'],board_bottom-t['blind_bore_bottom_mm']+.02)
        parts[f'module_nut_{i}']=Pos(x,y,zt-h['nut_height_mm'])*nut
        parts[f'module_screw_{i}']=Pos(x,y,board_top+h['head_height_mm']/2)*Cylinder(h['head_diameter_mm']/2,h['head_height_mm'])+Pos(x,y,board_top-h['module_screw_length_mm']/2)*Cylinder(h['thread_diameter_mm']/2,h['module_screw_length_mm'])
        module_nuts.append({'center_mm':[x,y],'nut_bottom_mm':zt-h['nut_height_mm'],'nut_top_mm':zt,'screw_tip_mm':board_top-h['module_screw_length_mm'],'screw_seat_mm':board_top})
    holes=[]
    for i,y in enumerate(a['y_centers_mm']):
        rear=a['rear_x_mm'];front=a['beam_rear_face_x_mm'];z=a['bolt_z_mm'];seat=a['screw_seat_x_mm']
        tray+=Pos(rear+a['upright_thickness_mm']/2,y,(floor+a['bridge_top_mm'])/2)*Box(a['upright_thickness_mm'],a['width_mm'],a['bridge_top_mm']-floor)
        tray+=Pos((rear+front)/2,y,(a['bridge_top_mm']+a['bridge_bottom_mm'])/2)*Box(front-rear,a['width_mm'],a['bridge_top_mm']-a['bridge_bottom_mm'])
        tray-=along_x((rear+front)/2,y,z,t['screw_clearance_radius_mm'],front-rear+.2)
        tray-=along_x((rear-.1+seat)/2,y,z,a['counterbore_radius_mm'],seat-rear+.1)
        parts[f'beam_screw_{i}']=along_x(seat-h['head_height_mm']/2,y,z,h['head_diameter_mm']/2,h['head_height_mm'])+along_x(seat+h['beam_screw_length_mm']/2,y,z,h['thread_diameter_mm']/2,h['beam_screw_length_mm'])
        parts[f'beam_nut_{i}']=Pos(a['beam_front_face_x_mm'],y,z)*Rot(0,90,0)*nut
        holes.append(along_x((front+a['beam_front_face_x_mm'])/2,y,z,t['screw_clearance_radius_mm'],a['beam_front_face_x_mm']-front+2))
    # Relieve the rear underside against the rounded belly, preserving the upper plate.
    if t.get('rear_bottom_bevel_width_mm',0):
        wedge=Polygon((x0,floor),(x0+t['rear_bottom_bevel_width_mm'],floor),(x0,floor+t['rear_bottom_bevel_height_mm']),align=None)
        tray-=Pos(0,y1+1,0)*Rot(90,0,0)*extrude(wedge,amount=y1-y0+2)
    parts['tray']=tray
    p=c['capacitor_clearance_probe'];loc=Pos(*p['base_mm'])*Rot(*p['rotation_deg'])
    parts['capacitor']=loc*import_step(p['source']);parts['vent']=loc*import_step(p['vent_source'])
    parts['capacitor_working_space']=loc*Pos(0,0,10.5)*Cylinder(9.75,27)
    return c,parts,holes,{'board_bottom_mm':board_bottom,'board_top_mm':board_top,'module_fasteners':module_nuts,'module_hole_centers_mm':centers,'beam_hole_centers_mm':[[a['beam_rear_face_x_mm'],y,a['bolt_z_mm']] for y in a['y_centers_mm']]}
