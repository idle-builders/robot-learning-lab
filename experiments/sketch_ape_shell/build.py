"""Sketch-only ape shell study. Units: mm, X right, Y rear, Z up.

Run with the repository's .venv/bin/python. All sculpting stations below are
dimensioned at 350 mm. --height uniformly scales the silhouette; --wall is an
independent nominal radial wall thickness in output millimetres. No mechanisms.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import trimesh
from build123d import (
    Solid, Wire, Edge, Face, Plane, Axis, Vector, Vertex, Compound, Color,
    export_step, import_step,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT.parents[1] / "artifacts" / "sketch-ape-shell"
IVORY = '#dfd8c6'
FACE = '#bcae93'
MUZZLE = '#e8deca'
JOINT = '#6b675c'
EYE = '#201f1a'
RIM = '#827968'
ORANGE = '#cd784c'


def ellipsoid(center, radii):
    return Solid.make_sphere(1).scale(tuple(radii)).translate(center)


def cranial_loft(center, radii):
    # Explicit pole vertices avoid a STEP reader ambiguity on trimmed,
    # affinely transformed spherical surfaces. Stations follow an ellipsoid.
    x,y,z=center
    rx,ry,rz=radii
    sections=[Vertex(x,y,z-rz)]
    for angle in [-80,-65,-45,-20,0,20,45,65,80]:
        a=math.radians(angle)
        sections.append(Wire.make_ellipse(rx*math.cos(a),ry*math.cos(a),
                        Plane(origin=(x,y,z+rz*math.sin(a)))))
    return Solid.make_loft(sections+[Vertex(x,y,z+rz)])


def box(x0, x1, y0, y1, z0=-20, z1=380):
    return Solid.make_box(x1-x0, y1-y0, z1-z0, Plane(origin=(x0,y0,z0)))


def loft(stations):
    # Each station is z, centre-x, centre-y, x-radius, y-radius.
    return Solid.make_loft([
        Wire.make_ellipse(rx, ry, Plane(origin=(x,y,z)))
        for z,x,y,rx,ry in stations
    ])


def tube_path(points, radius):
    path = Edge.make_spline(points)
    return Solid.sweep(Wire.make_circle(radius, Plane(origin=points[0],
        z_dir=Vector(points[1])-Vector(points[0]))), path)


def prism_xz(points, y0=-90, length=100):
    edge = Edge.make_spline([(x,y0,z) for x,z in points], periodic=True)
    return Solid.extrude(Face(Wire([edge])), (0,length,0))


def above_waist(delta=0):
    """Curved belly/pelvis seam read from the front sketch."""
    pts=[(x,-100,163+.004*x*x+delta) for x in [-160,-80,-60,-40,0,40,60,80,160]]
    edges=[Edge.make_spline(pts), Edge.make_line(pts[-1],(160,-100,400)),
           Edge.make_line((160,-100,400),(-160,-100,400)),
           Edge.make_line((-160,-100,400),pts[0])]
    return Solid.extrude(Face(Wire(edges)),(0,200,0))


class Study:
    def __init__(self, height=350, wall=2, gap=.65):
        if not 350 <= height <= 500:
            raise ValueError('Height must be within the design study range: 350–500 mm')
        if not 1 <= wall <= 3.5:
            raise ValueError('Nominal wall must be between 1 and 3.5 mm')
        self.height, self.s = height, height/350
        self.wall, self.t, self.gap = wall, wall/(height/350), gap/(height/350)
        self.parts = []

    def add(self, name, shape, color=IVORY, group='shell', explode=(0,0,0)):
        shape = shape.clean()
        solids = shape.solids()
        if not shape.is_valid or len(solids) != 1 or shape.volume <= 0:
            raise ValueError(f'{name}: invalid or disconnected part ({len(solids)} solids)')
        shape = shape.scale(self.s)
        shape.label, shape.color = name, Color(color)
        self.parts.append(dict(name=name, shape=shape, color=color, group=group,
                               explode=explode))
        print(f'  {name}', flush=True)

    def split(self, name, shell, color=IVORY, group='shell', y=0, outward=0):
        g = self.gap/2
        self.add(name+'_front', shell & box(-180,180,-100,y-g), color, group,
                 (outward,-23,0))
        self.add(name+'_back', shell & box(-180,180,y+g,100), color, group,
                 (outward,23,0))

    def hollow_loft(self, stations, crown=0):
        if crown:
            z,x,y,_,_=stations[-1]
            outer=Solid.make_loft([Wire.make_ellipse(rx,ry,Plane(origin=(cx,cy,cz)))
                                  for cz,cx,cy,rx,ry in stations]+[Vertex(x,y,z+crown)])
        else:
            outer = loft(stations)
        inner = loft([(z,x,y,max(.8,rx-self.t),max(.8,ry-self.t))
                      for z,x,y,rx,ry in stations])
        return outer-inner, outer

    def hollow_ellipsoid(self, c, r):
        return ellipsoid(c,r)-ellipsoid(c,tuple(a-self.t for a in r))

    def build(self):
        t = self.t
        # Torso: broad shoulders, rounded chest, taper towards the hip bowl.
        torso, torso_outer = self.hollow_loft([
            (160,0,2,42,25), (173,0,2,49,29), (199,0,3,58,34),
            (229,0,4,63,35), (253,0,5,63,30), (264,0,5,60,27),
            (274,0,5,35,22), (278,0,5,23,18)])
        neck_hole = Solid.make_cylinder(20,30,Plane(origin=(0,4,267)))
        torso = (torso-neck_hole) & above_waist()
        # A tiny vertical lozenge follows the sketch's single chest light.
        light = ellipsoid((0,-27,244),(1.75,3,5.4)) & torso_outer
        torso = torso-ellipsoid((0,-27,244),(1.95,4.5,5.8))
        self.split('torso',torso,y=4,group='torso')
        self.add('chest_indicator',light,ORANGE,'details',(0,-28,0))

        thigh_station = [(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
                         (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)]
        thigh, thigh_outer = self.hollow_loft(thigh_station)
        pelvis, pelvis_outer = self.hollow_loft([
            (130,0,5,9,11),(135,0,5,20,17),(145,0,5,34,23),
            (155,0,4,43,26),(164,0,3,46,26),(178,0,3,49,29)])
        pelvis=pelvis-above_waist(-self.gap)
        for side in [-1,1]:
            cutter = thigh_outer.scale(1.025,about=(35,3,140))
            if side < 0: cutter = cutter.mirror(Plane.YZ)
            pelvis = pelvis-cutter
        self.split('pelvis',pelvis,group='pelvis',y=4)

        neck = self.hollow_ellipsoid((0,4,276),(21,18,7))
        neck = neck & box(-40,40,-40,40,270,280)
        self.add('neck_collar',neck,JOINT,'joint_covers',(0,0,13))

        # Ellipsoidal head with an explicit crown and heart-shaped aperture.
        head_c=(0,0,313)
        head_r=(39,35,37)
        face_outline=[(0,328.5),(15,331),(24,329),(30,321),(31,313),
                      (27,303),(23,298),(22,287),(16,279),(0,276.7),
                      (-16,279),(-22,287),(-23,298),(-27,303),(-31,313),
                      (-30,321),(-24,329),(-15,331)]
        face_prism=prism_xz(face_outline,length=86)
        head=cranial_loft(head_c,head_r)-cranial_loft(head_c,tuple(r-t for r in head_r))
        head=head-face_prism
        self.split('head_cranium',head,group='head',y=2)
        face_outer=cranial_loft(head_c,(39,35.25,37))
        face_inner=cranial_loft(head_c,(39-t,35.25-t,37-t))
        mask=(face_outer-face_inner) & face_prism
        muzzle_c=(0,-31,292)
        muzzle_r=(23,16,17)
        muzzle_outer=ellipsoid(muzzle_c,muzzle_r)
        mask=mask-muzzle_outer.scale(1.017,about=muzzle_c)
        for side in [-1,1]:
            cutter=Solid.make_cylinder(7.25,70,Plane(origin=(side*18,-60,312),z_dir=(0,1,0)))
            mask=mask-cutter
        self.add('face_mask',mask,FACE,'head',(0,-19,7))
        muzzle=self.hollow_ellipsoid(muzzle_c,muzzle_r)-face_inner
        # A narrow smile and two discreet nostril dimples in the actual CAD.
        mouth=[]
        for x in np.linspace(-14.5,14.5,11):
            z=284.3+2.8*(x/14.5)**2
            y=-31-16*math.sqrt(1-(x/23)**2-((z-292)/17)**2)
            mouth.append((float(x),y-.1,z))
        mouth_tool=tube_path(mouth,.36)
        muzzle=muzzle-mouth_tool
        self.add('mouth_inlay',tube_path(mouth,.2),JOINT,'details',(0,-29,7))
        for side in [-1,1]:
            c=(side*5,-44.6,300.1)
            nostril=ellipsoid(c,(1.5,1.25,.9))
            muzzle=muzzle-nostril
            self.add(('L' if side<0 else 'R')+'_nostril_inlay',
                     ellipsoid((side*5,-44.5,300.1),(1.2,.8,.65)),JOINT,'details',(0,-29,7))
        self.add('muzzle_shell',muzzle,MUZZLE,'head',(0,-28,7))
        for side in [-1,1]:
            label='L' if side<0 else 'R'
            eye_c=(side*18,-28.7,312)
            bezel=ellipsoid(eye_c,(7.05,4.4,7.05))-ellipsoid(eye_c,(5.35,6,5.75))
            bezel=bezel & box(-40,40,-60,-29,300,325)
            self.add(label+'_eye_bezel',bezel,RIM,'head',(side*3,-26,7))
            lens=ellipsoid((side*18,-29.5,312),(5.25,3.4,5.65))
            self.add(label+'_eye_lens',lens,EYE,'head',(side*3,-31,7))

        # Mirrored limb exteriors; these are cosmetic joint covers, no mechanics.
        for side in [-1,1]:
            label='L' if side<0 else 'R'
            def mirrored(shape):
                return shape.mirror(Plane.YZ) if side<0 else shape
            def split_limb(name,shape,group,y=2):
                self.split(label+'_'+name,mirrored(shape),group=group,y=y,outward=side*18)
            upper,_=self.hollow_loft([
                (199,93,2,14,17),(207,94,2,17,20),(228,89,3,20,22),
                (249,80,4,19,21),(260,73,4,14,17),(265,70,4,6,9)],crown=2.8)
            split_limb('upper_arm',upper,'arms')
            fore,_=self.hollow_loft([
                (123,99,1,15,16),(133,102,1,18,19),(155,103,1,21,22),
                (178,101,1,18,20),(190,97,1,13,15)])
            split_limb('forearm',fore,'arms',y=1)
            elbow=self.hollow_ellipsoid((96,2,194.5),(12.7,14.5,8))
            elbow=elbow & box(75,115,-30,30,189,200)
            self.add(label+'_elbow_cuff',mirrored(elbow),JOINT,'joint_covers',(side*22,0,0))
            def shoulder_ring(reduction):
                return Solid.make_loft([Wire.make_ellipse(ry-reduction,rz-reduction,
                    Plane(origin=(x,4,245),x_dir=(0,1,0),z_dir=(1,0,0)))
                    for x,ry,rz in [(63,17,19),(65.5,17,19),(68,16,18)]])
            shoulder=shoulder_ring(0)-shoulder_ring(t)
            self.add(label+'_shoulder_trim',mirrored(shoulder),JOINT,'joint_covers',(side*10,0,0))
            wrist=self.hollow_ellipsoid((99,1,122),(12,13.5,5))
            self.add(label+'_wrist_cuff',mirrored(wrist),JOINT,'joint_covers',(side*22,0,0))
            palm=self.hollow_ellipsoid((99,-1,106),(16,13,20))
            palm=palm & box(70,122,-25,23,93,120)
            split_limb('palm',palm,'hands',y=0)
            # Four broad, softly curled finger caps and an opposable thumb.
            for i,(x,bottom,radius) in enumerate([(85,81,4.9),(94,77,5.3),(103,78.5,5.1),(111,83,4.6)],1):
                center=(x-1,-10,(bottom+100)/2)
                radii=(radius,radius+1,(100-bottom)/2+1)
                finger=self.hollow_ellipsoid(center,radii)
                finger=finger.rotate(Axis(center,(1,0,0)),-28)
                finger=finger.rotate(Axis(center,(0,1,0)),23)
                self.add(label+f'_finger_{i}_cap',mirrored(finger),IVORY,'hands',(side*25,-10,-8))
            thumb_st=[(92,78,-6,3,4),(95,78,-9,5,6),(104,80,-9,6,6.5),
                      (113,85,-4,5.5,6),(116,88,-1,3,4)]
            thumb,_=self.hollow_loft(thumb_st)
            self.add(label+'_thumb_cap',mirrored(thumb),IVORY,'hands',(side*17,-13,-3))
            split_limb('thigh',thigh,'legs',y=3)
            knee=self.hollow_ellipsoid((42,3,86),(15,17,7))
            self.add(label+'_knee_cuff',mirrored(knee),JOINT,'joint_covers',(side*12,0,-7))
            calf,_=self.hollow_loft([
                (23,46,3,21,20),(36,46,3,23,23),(57,45,4,22,23),
                (75,43,4,19,20),(83,42,3,16,17)])
            split_limb('shin',calf,'legs',y=3)
            ankle=self.hollow_ellipsoid((46,3,24),(19,18,5))
            self.add(label+'_ankle_cuff',mirrored(ankle),JOINT,'joint_covers',(side*12,0,-12))
            # A low, broad three-toed foot, with a separate flat sole.
            foot,foot_outer=self.hollow_loft([
                (2,46,-8,30,34),(5,46,-8,31,35),(12,46,-7,29,33),
                (20,46,-1,24,25),(26,46,3,19,19)])
            # Toe panel seams run along Y, restricted to the front of each boot.
            toe_zone=box(10,82,-65,-15,0,35)
            boot=foot-toe_zone
            self.add(label+'_foot_dorsal',mirrored(boot),IVORY,'feet',(side*15,5,-15))
            for i,(a,b) in enumerate([(10,36),(36,56),(56,82)],1):
                toe=foot & box(a+self.gap/2,b-self.gap/2,-65,-15-self.gap,0,35)
                self.add(label+f'_toe_{i}',mirrored(toe),IVORY,'feet',(side*15,-16,-15))
            sole=loft([(0,46,-8,29.4,33.4),(1.4,46,-8,30,34)])
            self.add(label+'_sole',mirrored(sole),JOINT,'feet',(side*15,0,-23))
        return self

    def export(self, destination, mesh=True):
        destination.mkdir(parents=True,exist_ok=True)
        assembly=Compound(label=f'Ape shell study — {self.height:g} mm',
                          children=[p['shape'] for p in self.parts])
        export_step(assembly,destination/'ape_shell_assembly.step')
        # Mesh the reopened exchange geometry. STEP healing normalizes tiny
        # trim edges that OCC's direct in-memory mesher can leave ambiguous.
        reopened=import_step(destination/'ape_shell_assembly.step')
        exchange_parts={part.label:part for part in reopened.children}
        if len(reopened.solids())!=len(self.parts):
            raise ValueError('STEP round-trip changed the assembly solid count')
        manifest={'height_mm':self.height,'nominal_wall_mm':self.wall,
                  'seam_gap_mm':self.gap*self.s,'units':'mm',
                  'reference':'experiments/sketch_ape_shell/reference/sketch.png','part_count':len(self.parts),
                  'parts':[]}
        mesh_parts=[]
        stls=destination/'stl'
        if mesh: stls.mkdir(exist_ok=True)
        for p in self.parts:
            shape=p['shape']
            bb=shape.bounding_box()
            entry={k:v for k,v in p.items() if k!='shape'}
            entry.update(volume_mm3=round(shape.volume,3),valid=shape.is_valid,
                         solids=len(shape.solids()),bounds_mm=[list(bb.min),list(bb.max)])
            manifest['parts'].append(entry)
            if mesh:
                vertices,faces=exchange_parts[p['name']].tessellate(.1*self.s,.12)
                m=trimesh.Trimesh(vertices=np.array([tuple(v) for v in vertices]),
                                  faces=np.array(faces),process=False)
                printable=m.copy()
                printable.vertices=np.asarray(printable.vertices,dtype=np.float32)
                printable.merge_vertices(digits_vertex=5)
                printable.update_faces(printable.nondegenerate_faces())
                printable.update_faces(printable.unique_faces())
                if not printable.is_watertight:
                    # OCC occasionally leaves a sub-tolerance polygon at a trim
                    # intersection. Patch only tiny closed boundary loops.
                    edges,counts=np.unique(printable.edges_sorted,axis=0,return_counts=True)
                    boundary=edges[counts==1]
                    if len(boundary) and np.all(counts<=2):
                        import networkx as nx
                        graph=nx.Graph();graph.add_edges_from(boundary)
                        vertices=printable.vertices.tolist();faces=printable.faces.tolist()
                        directed={tuple(sorted(e)):tuple(e) for e in printable.edges}
                        for nodes in nx.connected_components(graph):
                            local=graph.subgraph(nodes)
                            points=printable.vertices[list(nodes)]
                            if all(d==2 for _,d in local.degree()) and np.linalg.norm(np.ptp(points,axis=0))<.3*self.s:
                                center=len(vertices);vertices.append(points.mean(axis=0).tolist())
                                for a,b in local.edges:
                                    u,v=directed[tuple(sorted((a,b)))];faces.append((v,u,center))
                        printable=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
                if not printable.is_watertight:
                    raise ValueError(f"STL boundary remains open: {p['name']}")
                printable.export(stls/(p['name']+'.stl'))
                mesh_parts.append({**{k:v for k,v in p.items() if k!='shape'},
                    'positions':np.asarray(m.vertices).round(4).flatten().tolist(),
                    'normals':np.asarray(m.vertex_normals).round(5).flatten().tolist(),
                    'indices':np.asarray(m.faces).flatten().tolist()})
        bb=assembly.bounding_box()
        manifest['bounds_mm']=[list(bb.min),list(bb.max)]
        manifest['dimensions_mm']=list(bb.size)
        (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
        if mesh:
            (destination/'mesh.json').write_text(json.dumps({'height':self.height,'parts':mesh_parts},separators=(',',':')))
        print(json.dumps({'parts':len(self.parts),'dimensions_mm':list(bb.size),
                          'all_valid':all(p['shape'].is_valid for p in self.parts)}),flush=True)
        return manifest


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--height',type=float,default=350)
    ap.add_argument('--wall',type=float,default=2)
    ap.add_argument('--gap',type=float,default=.65)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--no-mesh',action='store_true')
    args=ap.parse_args()
    dest=args.output or OUTPUT_ROOT/'exports'/f'{args.height:g}mm'
    Study(args.height,args.wall,args.gap).build().export(dest,not args.no_mesh)
