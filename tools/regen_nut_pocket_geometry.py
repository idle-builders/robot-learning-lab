"""Crossbeam hex-nut recess hypothesis; no new printed component."""
from pathlib import Path
import yaml
from build123d import Pos,Rot,RegularPolygon,extrude
from tools.regen_mount_geometry import geometry,along_x,hex_nut

def pocket_geometry(path=Path('electronics/regen-nut-pockets.yaml')):
    cfg=yaml.safe_load(path.read_text());mount,parts,cuts,interface=geometry(Path(cfg['mount_source']))
    a=mount['arches'];h=mount['hardware'];p=cfg['pocket'];pockets=[]
    for i,y in enumerate(a['y_centers_mm']):
        x=p['nut_bearing_face_x_mm'];z=a['bolt_z_mm'];seat=a['screw_seat_x_mm']
        parts[f'beam_nut_{i}']=Pos(x,y,z)*Rot(0,90,0)*hex_nut(h['nut_af_mm'],h['nut_height_mm'])
        parts[f'beam_screw_{i}']=along_x(seat-h['head_height_mm']/2,y,z,h['head_diameter_mm']/2,h['head_height_mm'])+along_x(seat+cfg['beam_screw_length_mm']/2,y,z,h['thread_diameter_mm']/2,cfg['beam_screw_length_mm'])
        pockets.append(Pos(x,y,z)*Rot(0,90,0)*extrude(RegularPolygon(p['across_flats_mm']/2,6,major_radius=False),amount=p['depth_mm']+.01))
    return cfg,mount,parts,cuts,pockets,interface
