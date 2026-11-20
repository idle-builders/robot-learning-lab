"""Exercise native RGB/depth frames and capture-time localization semantics."""
import copy
import xml.etree.ElementTree as ET
import mujoco
from PIL import Image
from common import *
from runtime import capture_transform,metric_point
from build import element,save_xml

def head_pose(model,data,t):
    data.qpos[model.jnt_qposadr[model.joint('neck_yaw').id]]=.4*np.sin(2*np.pi*.5*t)
    data.qpos[model.jnt_qposadr[model.joint('neck_pitch').id]]=.13*np.sin(2*np.pi*.5*t+.3)
    mujoco.mj_forward(model,data)

def main():
    root=ET.parse(OUT/'cove50-rigid.xml').getroot();world=root.find('worldbody')
    for name,pos,size,color in [('back_wall',[1.5,0,.7],[.02,.9,.7],[.65,.70,.72,1]),
        ('red_box',[.65,-.20,.14],[.08,.08,.14],[.85,.12,.08,1]),
        ('blue_box',[.95,.24,.23],[.10,.1,.23],[.08,.25,.85,1]),
        ('green_landmark',[1.46,0,.60],[.025,.08,.08],[.10,.8,.2,1])]:
        element(world,'geom',name='scene_'+name,type='box',pos=vec(pos),size=vec(size),rgba=vec(color))
    save_xml(OUT/'perception-world.xml',root)
    m=mujoco.MjModel.from_xml_path(str(OUT/'perception-world.xml'));d=mujoco.MjData(m)
    contract=read(OUT/'sensors.json');cams=[s for s in contract['sensors']if s['kind']in['rgb','depth']]
    report=dict(source=ref(OUT/'cove50-rigid.xml'),world=ref(OUT/'perception-world.xml'),
        scene='Authored synthetic calibration/occlusion scene; object poses are evaluator data only',
        measured_physical_calibrations=0,projection_roundtrip=[],latency_conditions=[],native_intrinsics=[],packet_conditions=[])
    captures=[];points=np.array([[.65,-.2,.2],[.95,.24,.35],[1.45,0,.6],[.8,0,.05]])
    for s in cams:
        cid=m.camera(s['name']+'_camera').id;K=np.array(s['nominal_K'])
        intrinsic=m.cam_intrinsic[cid];size=m.cam_sensorsize[cid]
        report['native_intrinsics'].append(dict(sensor=s['name'],resolution=m.cam_resolution[cid].tolist(),
            compiled_focal_m=intrinsic[:2].tolist(),compiled_sensor_size_m=size.tolist(),
            hfov_deg=float(np.rad2deg(2*np.arctan(size[0]/2/intrinsic[0]))),
            vfov_deg=float(np.rad2deg(2*np.arctan(size[1]/2/intrinsic[1]))),
            basis='Nominal supplier field rectangle reproduced by native camera; no calibrated physical intrinsics'))
    for t in np.arange(0,2.00001,1/30):
        head_pose(m,d,float(t));record=dict(capture_time_s=float(t))
        for s in cams:
            T=capture_transform(m,d,s['name']);K=np.array(s['nominal_K']);pc=(np.linalg.inv(T)@np.c_[points,np.ones(len(points))].T).T[:,:3]
            pixel=(K@pc.T).T;pixel=pixel[:,:2]/pixel[:,2,None]
            recovered=(np.linalg.inv(K)@np.c_[pixel,np.ones(len(pixel))].T).T*pc[:,2,None]
            world_points=(T@np.c_[recovered,np.ones(len(recovered))].T).T[:,:3]
            report['projection_roundtrip'].append(dict(time_s=float(t),sensor=s['name'],max_error_m=float(np.max(np.linalg.norm(world_points-points,axis=1)))))
            record[s['name']]=T.tolist()
        captures.append(record)
    for delay in [.02,.05,.1,.2]:
        errors=[];correct=[]
        for capture in captures:
            t=capture['capture_time_s'];Tc=np.array(capture['depth_optical']);head_pose(m,d,t+delay);Ta=capture_transform(m,d,'depth_optical')
            pc=(np.linalg.inv(Tc)@np.c_[points,np.ones(len(points))].T).T
            wrong=(Ta@pc.T).T[:,:3];right=(Tc@pc.T).T[:,:3]
            errors.extend(np.linalg.norm(wrong-points,axis=1));correct.extend(np.linalg.norm(right-points,axis=1))
        report['latency_conditions'].append(dict(delay_s=delay,head_yaw_amplitude_rad=.4,head_pitch_amplitude_rad=.13,frequency_Hz=.5,
            arrival_pose_position_error_m=dict(zip(['p10','p50','p90','p99'],np.quantile(errors,[.1,.5,.9,.99]).tolist())),
            capture_pose_max_roundtrip_error_m=float(max(correct)),
            interpretation='Known simulated capture pose supplied to transform test; no SLAM/localization-estimator accuracy claim'))
    calibration='synthetic-model-r1';T=np.array(captures[10]['depth_optical']);t=captures[10]['capture_time_s']
    packet=dict(health='available',calibration_id=calibration,clock_epoch='sim0',capture_time_s=t)
    pose=dict(capture_time_s=t,T_world_optical=T.tolist())
    cases=[('available',{},pose,t+.05),('missing_calibration',dict(calibration_id=None),pose,t+.05),
        ('clock_reset',dict(clock_epoch='sim1'),pose,t+.05),('late_sample',{},pose,t+1.),
        ('completion_pose',{},dict(capture_time_s=t+.05,T_world_optical=T.tolist()),t+.05),
        ('missing_pose',{},None,t+.05),('missing_signal',dict(health='lost'),pose,t+.05)]
    for label,delta,p,now in cases:
        result=metric_point({**packet,**delta},[0,0,1],calibration_id=calibration,pose_at_capture=p,clock_epoch='sim0',now_s=now,max_age_s=.15)
        report['packet_conditions'].append(dict(condition=label,result=result))
    for label, params in [
        ('absent_identity_on_both_sides', dict(packet=dict(health='available',calibration_id=None,clock_epoch=None,capture_time_s=t),calibration_id=None,clock_epoch=None,pose_at_capture=pose,now_s=t+.05,max_age_s=.15)),
        ('nonfinite_capture_time',dict(packet={**packet,'capture_time_s':float('nan')},calibration_id=calibration,clock_epoch='sim0',pose_at_capture=pose,now_s=t+.05,max_age_s=.15)),
        ('nonfinite_pose_time',dict(packet=packet,calibration_id=calibration,clock_epoch='sim0',pose_at_capture={**pose,'capture_time_s':float('nan')},now_s=t+.05,max_age_s=.15)),
        ('nonfinite_age_envelope',dict(packet=packet,calibration_id=calibration,clock_epoch='sim0',pose_at_capture=pose,now_s=t+.05,max_age_s=float('inf'))),
        ('nonfinite_now',dict(packet=packet,calibration_id=calibration,clock_epoch='sim0',pose_at_capture=pose,now_s=float('nan'),max_age_s=.15))]:
        report['packet_conditions'].append(dict(condition=label,result=metric_point(point_optical=[0,0,1],**params)))
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0]
    renderer=mujoco.Renderer(m,height=480,width=640)
    media=[]
    for index,t in enumerate([0.,.5,1.]):
        head_pose(m,d,t);renderer.update_scene(d,camera='rgb_optical_camera',scene_option=opt)
        path=OUT/f'rgb-look-{index}.png';Image.fromarray(renderer.render()).save(path);media.append(ref(path))
    renderer.close();s=next(s for s in cams if s['kind']=='depth');w,h=s['resolution'];head_pose(m,d,.5)
    renderer=mujoco.Renderer(m,height=h,width=w);renderer.enable_depth_rendering();renderer.update_scene(d,camera='depth_optical_camera',scene_option=opt)
    depth=renderer.render().copy();renderer.close()
    rng=np.random.default_rng(3753);valid=np.isfinite(depth)&(depth>=.2)&(depth<=4)&(rng.random(depth.shape)>.02)
    noisy=depth+rng.normal(size=depth.shape)*np.maximum(.005,.01*depth);noisy[~valid]=np.nan
    yy,xx=np.mgrid[:h,:w];K=np.array(s['nominal_K']);pc=(np.linalg.inv(K)@np.array([xx.ravel(),yy.ravel(),np.ones(w*h)])).T*noisy.ravel()[:,None]
    np.savez_compressed(OUT/'synthetic-depth-packet.npz',optical_z_m=noisy,valid=valid,points_optical_m=pc.reshape(h,w,3),
        capture_time_s=.5,T_world_optical=capture_transform(m,d,'depth_optical'),nominal_K=K,clock_epoch='sim0',calibration_id=calibration)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5));im=ax.imshow(noisy,vmin=.2,vmax=2,cmap='viridis');ax.set_title('Synthetic ToF · optical Z depth · capture t=0.5s');ax.set_xlabel('pixel column');ax.set_ylabel('pixel row');fig.colorbar(im,ax=ax,label='m');fig.tight_layout();fig.savefig(OUT/'depth-look.png',dpi=140);plt.close(fig)
    report['depth_packet']=dict(source=ref(OUT/'synthetic-depth-packet.npz'),valid_pixels=int(valid.sum()),total_pixels=int(valid.size),
        depth_kind='Synthetic optical-axis Z depth from rasterization. Physical B0410 SDK axial-versus-radial semantics unverified; no silent equivalence.',
        noise='Assumed sigma=max5mm,1%depth and2% random invalidity; no multipath/amplitude/exposure/ambient model')
    report['media']=media+[ref(OUT/'depth-look.png')]
    report['limitations']=['Kinematic prescribed head scan, no learned gaze controller or walking localization',
        'Fixed CAD camera transforms, IMU die mapping and pixel roll need physical calibration',
        'Synthetic truth supplied only to evaluator transform checks; no claim a real estimator has this capture pose',
        'No domain-matched appearance, lens distortion, rolling-shutter readout or four-phase ToF motion/IR model',
        'Recorded age limits are a caller-supplied experiment envelope, not a universal safety threshold']
    write(OUT/'perception-validation.json',report);write(OUT/'synthetic-camera-transforms.json',dict(captures=captures))
    print(dict(captures=len(captures),max_projection_roundtrip_m=max(r['max_error_m']for r in report['projection_roundtrip']),depth_valid_pixels=int(valid.sum())))

if __name__=='__main__':main()
