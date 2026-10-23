"""Terminal-block drawing envelopes on the supplier shunt power ports."""
from pathlib import Path
import json,yaml,numpy as np
from build123d import Box,Cylinder,Pos
from tools.capacitor_saddle_geometry import geometry as saddle_geometry

def geometry(path=Path('electronics/regen-terminal.yaml')):
 c=yaml.safe_load(path.read_text());_,m,parts,*_=saddle_geometry(Path(c['parent']))
 ports=json.loads(Path(c['port_source']).read_text())['ports']
 # Current board transform is explicitly Rz(180); fail if it changes.
 assert m['module_rotation_deg']==[0,0,180]
 centers={p['name']:[m['module_center_mm'][0]+m['supplier_center_mm'][0]-p['center_mm'][0],m['module_center_mm'][1]+m['supplier_center_mm'][1]-p['center_mm'][1],m['module_center_mm'][2]-m['supplier_center_mm'][2]+p['center_mm'][2]] for p in ports}
 x,y,z=np.mean(list(centers.values()),axis=0)
 assert abs(centers['VIN'][0]-centers['GND'][0])<1e-8
 assert abs(abs(centers['VIN'][1]-centers['GND'][1])-c['common']['pitch_mm'])<.0001
 cases={}
 for kind,t in c['candidates'].items():
  for direction in [1,-1]:
   # Direction describes the deeper body side; side-entry wires enter here.
   shift=direction*(t['front_distance_from_pin_mm']-t['body_depth_mm']/2)
   center=[float(x+shift),float(y),float(z+t['body_height_mm']/2)]
   body=Pos(*center)*Box(t['body_depth_mm'],t['body_width_mm'],t['body_height_mm'])
   a=c['probe']['body_allowance_mm']
   # Seating plane is deliberate contact; expand lateral/top space, not into PCB.
   allowance=Pos(center[0],center[1],center[2]+a/2)*Box(t['body_depth_mm']+2*a,t['body_width_mm']+2*a,t['body_height_mm']+a)
   pins={};drivers={};solder={}
   for name,(_,py,_) in centers.items():
    length=t['pin_length_mm']+t['pin_length_tolerance_mm']
    pins[name]=Pos(float(x),py,float(z-length/2))*Cylinder(c['common']['pin_diameter_mm']/2,length)
    bottom=m['module_center_mm'][2]-m['supplier_center_mm'][2]
    depth=c['probe']['solder_below_board_mm']
    solder[name]=Pos(float(x),py,bottom-depth/2)*Cylinder(c['probe']['solder_radius_mm'],depth)
    start=z+t['body_height_mm'];top=c['probe']['screwdriver_top_z_mm']
    drivers[name]=Pos(float(x),py,float((start+top)/2))*Cylinder(c['probe']['screwdriver_radius_mm'],float(top-start))
   cases[kind+('-forward' if direction==1 else '-rearward')]={'body':body,'allowance':allowance,'pins':pins,'solder':solder,'drivers':drivers,'placement':{'body_center_mm':center,'body_entry_direction':direction,'body_height_mm':t['body_height_mm'],'port_centers_mm':centers}}
 return c,m,parts,cases
