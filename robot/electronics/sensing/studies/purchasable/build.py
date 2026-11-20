"""V17: purchased manufacturer modules on source V5/V16, no PCB authoring."""
from pathlib import Path
import sys, copy, json, math
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'integrated-attachments'))

# Imported helper module above has the same basename: load it explicitly instead.
import importlib.util
spec=importlib.util.spec_from_file_location('attachment_helpers',Path(__file__).resolve().parents[1]/'integrated-attachments/build.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
for key in ['read','write','rel','sha','bb','cx','cy','common','placed','module','ensure']:
    globals()[key]=getattr(h,key)
from build123d import Axis,Box,Cylinder,Compound,RegularPolygon,extrude,export_step,export_stl,import_step
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[5]
BASE=ROOT/'artifacts/cove50-sensor-integration/v4'
OUT=ROOT/'artifacts/cove50-sensor-integration/v17'
SOURCES=ROOT/'artifacts/cove50-sensor-integration/purchasable-sources-20261024'

def main():
    OUT.mkdir(exist_ok=True);(OUT/'parts').mkdir(exist_ok=True)
    baseline=read(BASE/'assembly-parts.json');index={p['name']:p for p in baseline['parts']}
    shapes={};rows=[];scene=trimesh.Scene();interfaces=[]
    def add(name,s,link,role,material,color,basis=None,sku=None):
        s=Compound(children=list(s.solids()),label=name);shapes[name]=s
        path=OUT/'parts'/f'{name}.step';meshpath=path.with_suffix('.stl')
        export_step(s,path);export_stl(s,meshpath,tolerance=.06,angular_tolerance=.2)
        m=trimesh.load(meshpath,force='mesh',process=False);m.apply_scale(.001);m.visual.face_colors=color
        scene.add_geometry(m,node_name=name,geom_name=name)
        rows.append(dict(name=name,owning_link=link,role=role,material=material,physical_part=True,inspection_reference=False,physical_part_id=name,native_step=rel(path),native_step_sha256=sha(path),mesh=rel(meshpath),mesh_sha256=sha(meshpath),mesh_units='mm',bounds_mm=bb(s).tolist(),native_volume_mm3=s.volume,native_solids=len(s.solids()),native_validity=[bool(q.is_valid)for q in s.solids()],mass_kg=None,source_basis=basis,purchased_sku=sku))
        print('exported',name,len(s.solids()),flush=True)
    source_head=import_step(ROOT/index['upper__integrated_engineered_head']['native_step'])
    head=source_head
    # Keep original camera receiver, neck datums and outside silhouette. USB
    # exit is the established source-sized slot; no camera strap recess added.
    cutter=Box(10,50,8).translate((15.55,45,468.25))+cy(5,20,70,15.55,464.25)+cy(5,20,70,15.55,472.25)
    head=head-cutter
    head=head+Box(22,18,2.4).translate((0,0,496.2))
    head=head-Box(18,14,3).translate((0,0,494.5))-Box(4,8,1.2).translate((0,11,495.4))
    # Exact catalogue transforms, recorded as source -> world rigid operations.
    def crown(s):return s.translate((-10.16,-13.97,0)).rotate(Axis.X,-90).translate((-18,20,465))
    def mic(s):return s.translate((-8.255,-7.822,0)).rotate(Axis.Y,-90).translate((50.5,0,460))
    def imu(s):return s.translate((-12.7,-8.89,0)).rotate(Axis.X,-90).translate((-12,42,304))
    def adc(s):return s.translate((-15.24,-8.89,.1304)).rotate(Axis.X,-90).translate((-12,42,339))
    tilt=math.degrees(math.atan(.1475))
    def back(s):return s.rotate(Axis.X,tilt).translate((0,46,360))
    def back_reader(s):return back(s.translate((-10.16,-13.97,0)).rotate(Axis.X,90))
    raw={k:import_step(SOURCES/(k+'.step'))for k in ['1374','4517','6049','6494']}
    boards={'crown_reader_1374':(crown(raw['1374']),'head','1374'), 'microphone_6049':(mic(raw['6049']),'head','6049'), 'body_IMU_4517':(imu(raw['4517']),'torso','4517'), 'foot_ADC_6494':(adc(raw['6494']),'torso','6494'), 'back_reader_1374':(back_reader(raw['1374']),'torso','1374')}
    # Ray sample the real head's outside surface around each screw position;
    # stop integral posts inside that skin, rather than projecting through it.
    source_mesh=trimesh.load(ROOT/index['upper__integrated_engineered_head']['mesh'],force='mesh',process=False)
    if index['upper__integrated_engineered_head'].get('mesh_units')=='m':source_mesh.apply_scale(1000)
    def outer_y(x,z):
        origins=np.array([[x+dx,20,z+dz]for dx,dz in [(0,0),(2.4,0),(-2.4,0),(0,2.4),(0,-2.4)]])
        hits,ids,_=source_mesh.ray.intersects_location(origins,np.tile([0,1,0],(5,1)),multiple_hits=True)
        return min(max(hits[ids==i,1])for i in range(5))-.2
    crown_holes=[]
    for i,(x0,y0) in enumerate([(2.54,2.54),(17.78,2.54),(2.54,25.4),(17.78,25.4)]):
        x=x0-10.16-18;z=465-(y0-13.97);end=outer_y(x,z)
        head=head+cy(2.4,21.57,end,x,z)
        head=head-cy(.8,21.3,27,x,z)
        screw=cy(1,20,25,x,z)+cy(1.9,18,20,x,z)
        add(f'crown_M2x5_{i}',screw,'head','fastener','A2 DIN912',[160,165,175,255],sku='Orbital 2040820')
        crown_holes.append(dict(center_mm=[x,20,z],post_end_Y_mm=end,screw='M2x5',pilot_mm=1.6))
    # Bottom-port microphone: use source port (8.255,7.822), not package centre.
    head=head-Box(3.05,13.3,17.2).translate((49.125,-1.472,460))
    mic_holes=[]
    for i,x0 in enumerate([2.54,13.97]):
        y=10.16-7.822;z=460+x0-8.255
        head=head+cx(2.4,50.57,53.2,y,z)
        head=head-cx(.8,50.3,53.25,y,z)
        add(f'mic_M2x4_{i}',cx(1,48.93,52.93,y,z)+cx(1.9,46.93,48.93,y,z),'head','fastener','A2 DIN912',[160,165,175,255],sku='Orbital 2040810')
        mic_holes.append(dict(center_mm=[49,y,z],screw='M2x4',pilot_mm=1.6))
    head=head-cx(1.25,50.4,65,0,460)
    # Frame is derived directly from original V5, with two through holes and
    # two tapped back-cap holes. No accepted source file is changed.
    upper=read(ROOT/'artifacts/cove50-engineering-competition/upper-body-20261023/lead/v5/parts.json')
    frame_row=next(p for p in upper['parts']if p['name']=='A_main_spine_split_neck_r4')
    spine=import_step(ROOT/frame_row['native_step'])
    carrier=Box(36,2,58).translate((-12,39,322))
    for i,x in enumerate([-19,-5]):
        spine=spine-cy(1.35,30,38,x,318)
        carrier=carrier+cy(3.2,36,38.1,x,318)
        carrier=carrier-cy(1.35,35,41,x,318)
        add(f'carrier_M2p5x12_{i}',cy(1.25,28,40,x,318)+cy(2.25,40,42.5,x,318),'torso','fastener','A2 DIN912',[160,165,175,255],sku='Orbital M2.5x12')
        nut=extrude(RegularPolygon(5/math.sqrt(3),6),amount=2).rotate(Axis.X,90).translate((x,32,318))-cy(1.05,29,33,x,318)
        add(f'carrier_M2p5_nut_{i}',nut,'torso','fastener','Class8 BZP DIN934',[160,165,175,255],sku='Orbital 1770020')
    # Retain the lower-body torso mating screw at(-16,297.5); its
    #5.5mm head needs a clearance/tool window, accessible with IMU removed.
    carrier=carrier-cy(3.25,37.5,40.5,-16,297.5)
    module_holes={}
    for key,holes,fn,seat in [('IMU',[(2.54,2.54),(22.86,2.54),(2.54,15.24),(22.86,15.24)],imu,43.57),('ADC',[(2.54,2.54),(27.94,2.54),(2.54,15.24),(27.94,15.24)],adc,43.3278)]:
        centers=[]
        for i,(x0,y0)in enumerate(holes):
            sourcecenter=(12.7 if key=='IMU'else 15.24,8.89)
            x=x0-sourcecenter[0]-12;z=(304 if key=='IMU'else 339)-(y0-sourcecenter[1])
            carrier=carrier+cy(2.4,39.9,42,x,z)
            carrier=carrier-cy(.8,37,42.2,x,z)
            add(f'{key}_M2x6_{i}',cy(1,seat-6,seat,x,z)+cy(1.9,seat,seat+2,x,z),'torso','fastener','A2 DIN912',[160,165,175,255],sku='Orbital 2040830')
            centers.append([x,seat,z])
        module_holes[key]=centers
    # Removable dielectric back cap. Its purchased touch board is mounted to
    # its inside; the copper electrode is bonded in a separate relieved pocket.
    panel=back(Box(44,2,38).translate((0,3,0)))
    for i,(x,zlocal)in enumerate([(-7.62,-11.43),(7.62,-11.43),(-7.62,11.43),(7.62,11.43)]):
        panel=panel+back(cy(2.4,0,2.1,x,zlocal))
        panel=panel-back(cy(.8,-.1,3,x,zlocal))
        add(f'back_reader_M2x4_{i}',back(cy(1,-1.57,2.43,x,zlocal)+cy(1.9,-3.57,-1.57,x,zlocal)),'torso','fastener','A2 DIN912',[160,165,175,255],sku='Orbital 2040810')
    # Foil at inner cap face: relief is cut after the supporting posts.
    panel=panel-back(Box(22,.2,17).translate((0,2.05,0)))
    panel=panel-back(Box(4,.8,9).translate((0,2,-12)))
    back_fasteners=[]
    for i,(x,z)in enumerate([(-16,355),(16,375)]):
        outer_y=46+4/math.cos(math.radians(tilt))-.1475*(z-360)
        seat=45.
        panel=panel+cy(3.4,36,outer_y-.1,x,z)
        panel=panel-cy(1.35,35,outer_y+2,x,z)-cy(2.5,seat,outer_y+3,x,z)
        spine=spine-cy(1.025,30,40,x,z)
        length=12
        add(f'back_cap_M2p5x{length}_{i}',cy(1.25,seat-length,seat,x,z)+cy(2.25,seat,seat+2.5,x,z),'torso','fastener','A2 DIN912',[160,165,175,255],sku=f'Orbital M2.5x{length}')
        back_fasteners.append(dict(center_mm=[x,seat,z],length_mm=length,tip_Y_mm=seat-length))
    add('sensor_head_integrated_v17',head,'head','printed_head','PA12',[217,201,155,255],'Original V5 + crown platform, actual1374 posts, actual6049 posts/acoustic duct, USB slot')
    add('sensor_spine_integrated_v17',spine,'torso','frame','Inherited aluminium',[155,168,181,255],frame_row['native_step'])
    add('IMU_ADC_carrier_v17',carrier,'torso','produced_holder','PA12',[83,153,195,255],'Integral posts match two official manufacturer boards;2 original spine through-holes')
    add('back_touch_cap_v17',panel,'torso','produced_holder','PA12',[83,153,195,255],'Integral1374 board posts, independent foil pocket and two spine supports')
    add('crown_touch_electrode',Box(16,12,.07).translate((0,0,495.965)),'head','cut_stock','Adafruit4607 copper+conductive PSA',[203,116,64,255],sku='4607')
    add('back_touch_electrode',back(Box(20,.07,15).translate((0,2.115,0))),'torso','cut_stock','Adafruit4607 copper+conductive PSA',[203,116,64,255],sku='4607')
    for name,(s,link,sku)in boards.items():add(name,s,link,'purchased_module','Manufacturer assembled module',[45,145,90,255],rel(SOURCES/(sku+'.step')),f'Adafruit {sku}')
    # Fit selected0.5mm silicone sheet instead of unpurchasable0.6 allocation.
    # Existing FSR/backing/adhesive poses and sole/ground datums are retained.
    cfg=read(ROOT/'robot/electronics/sensing/foot-contact-coupon.json')
    for side in ['R','L']:
        old=import_step(ROOT/index[side+'_sensor_pocket_pad']['native_step']);pad=old
        for site in cfg['right_sites']:
            x,y=site['xy_mm'];x=x if side=='R'else-x
            oldpuck=index[side+'_'+site['id']+'_actuator'];b=np.array(oldpuck['bounds_mm']);top=b[1,2];bottom=top-.5
            #0.1mm floor restoration, joined at existing pocket base; leaves
            # manufacturer sensor package and vent/tail cutouts untouched.
            pad=pad+Cylinder(9.7,.1002).translate((x,y,b[0,2]+.05-.0001))
            add(side+'_'+site['id']+'_actuator_v17',Cylinder(5,.5).translate((x,y,(bottom+top)/2)),side+'_foot','cut_stock','Polymax1001526 silicone60ShA',[65,155,215,255],sku='Polymax1001526')
        add(side+'_sensor_pocket_pad_v17',pad,side+'_foot','produced_holder','TPU95A',[145,145,150,255],'Subtractive V16 pad derivative with0.1mm pocket-floor restoration for0.5mm silicone')
    export_step(Compound(children=list(shapes.values())),OUT/'attachment-package.step');scene.export(OUT/'attachment-package.glb')
    write(OUT/'parts.json',dict(version='v17',parts=rows,joints=[],units='mm'))
    write(OUT/'placement.json',dict(crown_reader_holes=crown_holes,microphone_port_world_mm=[50.5,0,460],microphone_holes=mic_holes,body_module_holes=module_holes,back_cap_fasteners=back_fasteners,transform_note='Full manufacturer geometry retained through rigid transforms; no mirrored purchased sensor or substituted package envelope.',body_datums='Source upperV5/lowerV16,22 axes.'))
    # Complete body: remove legacy custom IMU and the two old pad/puck sets.
    complete_scene=trimesh.load(BASE/'assembly.glb',force='scene',process=False)
    retained=[]
    for p in baseline['parts']:
        n=p['name'];remove=n in ['upper__integrated_engineered_head','sensor_spine_two_hole_proposal','R_sensor_pocket_pad','L_sensor_pocket_pad']or n.startswith('imu_')or n.endswith('_actuator')
        if remove:complete_scene.delete_geometry(complete_scene.graph[n][1])
        else:retained.append(copy.deepcopy(p))
    for p in rows:complete_scene.add_geometry(placed(scene,p['name']),node_name=p['name'],geom_name=p['name'])
    complete_scene.export(OUT/'assembly-with-reference.glb')
    mechanical=complete_scene.copy()
    for p in retained:
        if p.get('inspection_reference'):mechanical.delete_geometry(mechanical.graph[p['name']][1])
    mechanical.export(OUT/'assembly.glb')
    manifest=dict(baseline,version='v17',parts=retained+rows,scope='Purchase-ready body sensor selection and mechanical proposal; hands delegated T-9gspf2. No PCB design. Exact current compute/battery packaging remains a separate concurrent task.')
    write(OUT/'assembly-parts.json',manifest)
    # Readback retained source identity, and full-body native export.
    complete=[];missing=[];required=[Path(__file__),SOURCES/'sources.json',ROOT/frame_row['native_step'],ROOT/'robot/design.json',BASE/'assembly-parts.json']+[SOURCES/(k+'.step')for k in raw]
    for p in retained+rows:
        if p.get('inspection_reference')or p.get('physical_part')is False:continue
        if p['name']in shapes:q=shapes[p['name']]
        elif p.get('native_step'):
            path=ROOT/p['native_step'];ensure(path)
            if p.get('native_step_sha256')and sha(path)!=p['native_step_sha256']:raise ValueError('Source changed '+p['name'])
            q=import_step(path);required.append(path)
        else:missing.append(p['name']);continue
        q.label=p['name'];complete.append(q)
    export_step(Compound(children=complete),OUT/'assembly-native.step')
    write(OUT/'source-seal.json',dict(files=[dict(path=rel(p),sha256=sha(p),bytes=p.stat().st_size)for p in dict.fromkeys(required)]))
    report=dict(version='v17',source_upper='V5',source_lower='V16',context_groups=len(retained+rows),body_axes=len(manifest['joints']),native_groups=len(complete),native_missing=missing,attachment_groups=len(rows),head_native_solids=len(head.solids()),carrier_native_solids=len(carrier.solids()),back_cap_native_solids=len(panel.solids()),purchased_modules={n:dict(sku=sku,source=rel(SOURCES/(sku+'.step')),source_solids=len(raw[sku].solids()),installed_solids=len(s.solids()),volume_error_mm3=s.volume-raw[sku].volume)for n,(s,_,sku)in boards.items()},limits=['Native mechanical proposal on exact accepted sections; fabrication, tolerance, print threads, preload and sensing remain unmeasured.','Six local FSR channels do not measure total ground force or full-foot CoP; existing TPU bypass contact retained.','Magnetometer near motors is extra, not a required reliable heading source.','Current6049 module remains orderable stock, but its ICS43434 chip is discontinued; preserve exactSKU/CAD, no silent substitution.','Power/host I2C,I2S/GPIO firmware and complete moving harness routing belong to the existing compute/power integration. Sensor-side leads and interface requirements are listed in BOM.','Original torso skin is reference geometry; the backcap is a local proposed shell attachment. Concurrent compute/battery placement is not silently qualified.','Hands/wrists and their sensing belong to T-9gspf2.'])
    write(OUT/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
