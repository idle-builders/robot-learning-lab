"""Torso-local capacitor saddle overlay using existing native CAD recipes."""
from pathlib import Path
import yaml
from build123d import Box,Cylinder,Pos,Rot
from tools.regen_nut_pocket_geometry import pocket_geometry

def geometry(path=Path('electronics/capacitor-saddle.yaml')):
 c=yaml.safe_load(path.read_text());cfg,m,parts,cuts,pockets,interface=pocket_geometry(Path(c['parent']))
 s=c['saddle'];t=c['strap'];x=s['x_center_mm'];z=s['center_z_mm'];w=s['axial_width_mm'];r=s['bore_radius_mm'];ro=s['outer_radius_mm']
 def cylinder(radius,width):return Pos(x,0,z)*Rot(0,90,0)*Cylinder(radius,width)
 low=z-ro; high=s['top_z_mm']
 assert low < high <= z
 lower=Pos(x,0,(low+high)/2)*Box(w,2*ro,high-low)
 cradle=(cylinder(ro,w)&lower)-cylinder(r,w+2)
 xmin=s['join_x_min_mm'];xmax=x+w/2;floor=s['floor_z_mm'];top=s['support_top_z_mm']
 bridge=Pos((xmin+xmax)/2,0,(floor+top)/2)*Box(xmax-xmin,2*s['support_y_half_mm'],top-floor)
 bridge-=cylinder(r,2*(xmax-xmin)+10)
 extension=cradle+bridge
 # D-shaped strap: semicircular upper run and flat-bottomed lower run.
 # This is only a nominal packaging envelope, not a flexible-body solution.
 def d_shape(radius,bottom,width):
  upper=cylinder(radius,width)&(Pos(x,0,z+radius/2)*Box(width+2,2*radius+2,radius))
  return upper+Pos(x,0,(bottom+z)/2)*Box(width,2*radius,z-bottom)
 outer=t['outer_radius_mm'];bottom=t['bottom_z_mm'];th=t['thickness_mm']
 strap=d_shape(outer,bottom,t['width_mm'])-d_shape(outer-th,bottom+th,t['width_mm']+1)
 cl=t['slot_clearance_mm']
 channel=d_shape(outer+cl,bottom-cl,t['width_mm']+2*cl)-d_shape(outer-th-cl,bottom+th+cl,t['width_mm']+2*cl+1)
 extension-=channel
 tray=parts['tray']+extension
 head=Pos(x,outer+t['head_height_mm']/2,z+t['head_length_mm']/2)*Box(t['head_width_mm'],t['head_height_mm'],t['head_length_mm'])
 parts['original_tray']=parts['tray'];parts['tray']=tray;parts['strap']=strap;parts['strap_head']=head
 return c,m,parts,cuts,pockets,interface
