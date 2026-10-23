"""Two-nut counterhold plate; existing M2 nuts bear directly on the crossbeam."""
from pathlib import Path
import yaml
from build123d import Box,Pos,Rot,RegularPolygon,extrude
from tools.regen_mount_geometry import geometry,along_x

def holder_geometry(path=Path('electronics/regen-nut-holder.yaml')):
    cfg=yaml.safe_load(path.read_text())
    mount,parts,cuts,interface=geometry(Path(cfg['mount_source']))
    p=cfg['holder'];a=mount['arches'];x=p['rear_face_x_mm'];t=p['thickness_mm']
    ymin,ymax=p['y_limits_mm'];zmin,zmax=p['z_limits_mm']
    holder=Pos(x+t/2,(ymin+ymax)/2,(zmin+zmax)/2)*Box(t,ymax-ymin,zmax-zmin)
    for y in a['y_centers_mm']:
        pocket=Pos(x-.01,y,a['bolt_z_mm'])*Rot(0,90,0)*extrude(RegularPolygon(p['pocket_af_mm']/2,6,major_radius=False),amount=p['pocket_depth_mm']+.01)
        holder-=pocket
        holder-=along_x(x+t/2,y,a['bolt_z_mm'],p['screw_clearance_radius_mm'],t+.2)
    return cfg,mount,parts,holder
