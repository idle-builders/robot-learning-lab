"""Source-backed actuator, sensing and timing sidecars for portable models."""
from common import *

def main():
    m=read(CAD/'assembly-parts.json'); o=origins(m)
    c047=dict(model='FEETECH ST-3215-C047', quantity=20, nominal_voltage_V=12., voltage_range_V=[4.,14.],
        gear_ratio=345., mass_kg=.055, mass_tolerance_kg=.001,
        stall_torque_Nm=30*.0980665, rated_torque_Nm=10*.0980665,
        no_load_speed_rad_s=np.pi/3/.222, no_load_current_A=.180, idle_current_A=.030,
        stall_current_A=2.7, rated_current_A=.9, backlash_upper_bound_rad=np.deg2rad(.5),
        encoder_counts=4096, minimum_position_update_period_s=.001,
        supplier_terminal_resistance_ohm=1., supplier_Kt_Nm_A=11*.0980665,
        torque_speed_curve=dict(torque_Nm=[x*.0980665 for x in [0,3,6,9,12,15,18,21,24,27,30]],
            speed_rad_s=[45*2*np.pi/60*(1-x/30)for x in [0,3,6,9,12,15,18,21,24,27,30]],
            basis='Approximate supplier page6 plotted straight line through45rpm/0 and0rpm/30kgcm; ±10% catalog tolerance, not identified sample dynamics'),
        source_refs=[ref(OUT/'sources/c047-datasheet.pdf')],
        unknown=['Closed-loop PID gains/bandwidth and firmware response','Rotor/gear reflected inertia and efficiency','Coulomb/viscous friction and load-dependent backlash','Installed continuous thermal duty and shutdown recovery','Actual bus/power path and current measurement calibration'],
        interpretation='Gear ratio already inside output torque/speed; never multiply output effort by345 again. Terminal resistance and listed output Kt have ambiguous measurement planes; do not combine into a fictitious bus-winding circuit. Rated point is conditional catalog data, not an installed sustained rating.')
    xm=dict(model='ROBOTIS XM335-T323-T', quantity=2, nominal_voltage_V=12., recommended_voltage_V=11.1,
        voltage_range_V=[6.5,12.], gear_ratio=323.04, mass_kg=.027,
        supplier_inertia_reference_mass_kg=.030,
        stall_torque_Nm=1.12, rated_torque_Nm=None, no_load_speed_rad_s=57*2*np.pi/60,
        stall_current_A=.88, idle_current_A=.017, encoder_counts=4096,
        voltage_points=[dict(V=9.,stall_Nm=.85,stall_A=.61,rpm=43),dict(V=11.1,stall_Nm=1.03,stall_A=.80,rpm=53),dict(V=12.,stall_Nm=1.12,stall_A=.88,rpm=57)],
        performance_curve=dict(torque_Nm=[.05,.08,.16,.24,.32,.40,.48,.56,.64,.71],
            speed_rad_s=[x*2*np.pi/60 for x in [58.5,56.5,51.5,46.,40.,34.,27.5,20.5,12.,4.5]],
            current_A=[.065,.085,.13,.18,.235,.295,.365,.445,.545,.69],
            basis='Hand-read points from official performance image; approx±1.5rpm,±0.015Nm,±0.025A, graph voltage not explicitly printed. Retain image; do not extrapolate endpoint to stall1.12Nm.'),
        source_refs=[ref(OUT/'sources/xm335-manual.html'),ref(OUT/'sources/xm335-performance.jpg'),ref(OUT/'sources/xm335-inertia.pdf')],
        inertia_reference=dict(mass_kg=.030, COM_m=(np.array([-.58132712,-8.952185,-11.436848])/1000).tolist(),
            tensor_kg_m2=(np.array([[10230.473,-277.85249,-239.39013],[-277.85249,5748.8064,-3340.0323],[-239.39013,-3340.0323,5961.2787]])*1e-9).tolist(),
            application='Retained reference only:30g reference conflicts with27g e-manual/selected27g case+horn split; inertia reference point and selected native coordinate registration require reconciliation. Do not silently replace case-only native inertia with a whole assembly tensor.'),
        unknown=['Continuous torque and installed thermal response','Selected firmware/gains/control-profile/bus timing','Backlash/friction/rotor inertia','Supplier inertia reference point and delivered accessory mass split'],
        interpretation='Source yaw case25.5g plus horn1.5g represents27g assembly; supplier30g reference kept as explicit discrepancy. 3S full12.6V exceeds documented12V input: model regulated/capped neck rail12V, track actual supply in power follow-up.')
    act=[]
    for index,j in enumerate(m['joints']):
        typ='xm335' if j['name'].startswith('neck') else 'c047'
        act.append(dict(index=index,joint=j['name'],model=typ,source_axis=j,
            simulation_axis_model=(R@np.array(j['axis_world_neutral'])).tolist(),
            command_units='Output joint radians relative exact neutral CAD',
            hardware_motor_sign=j.get('motor_command_sign'),hardware_zero_counts=None,
            hardware_mapping='Uncalibrated electrical zero/direction. Source motor sign annotation is retained evidence, not a deployable calibration.',
            external_ratio=1.,simulation_kp_Nm_rad=8. if typ=='c047' else 2.,
            simulation_kd_Nm_s_rad=.16 if typ=='c047'else.05,
            servo_gain_basis='Unidentified impedance assumption for bounded simulation; not copied from supplier digital PID register units'))
    write(OUT/'actuators.json',dict(schema='cove50.actuators.v1',models=dict(c047=c047,xm335=xm),actuators=act,
        native_actuator='MJCF motors accept Nm; use runtime.py for delayed bounded position-target control and catalog speed/current sensitivity',
        default_torque_limit='C047 catalog rated0.980665Nm at12V; neck0.20Nm explicit assumption because no continuous rating. Peak curve mode is a separate sensitivity.',
        hardware_transfer='No installed actuator identification. Curves are supplier evidence; all estimated gains/damping/delay remain exposed.'))
    sensors=[]
    # Explicit proper optical axes; actual sensor pixel roll still requires a
    # checkerboard calibration. CAD lens fronts are entrance-pupil proxies.
    optical_CAD=np.column_stack([[-1.,0,0],[0,0,-1.],[0,-1.,0]])
    def sensor(name,link,point_CAD_mm,rotation_CAD,**kwargs):
        point=R@(np.array(point_CAD_mm)/1000)
        sensors.append(dict(name=name,owning_link=link,origin_CAD_mm=point_CAD_mm,
            origin_link_m=(point-o[link]).tolist(), rotation_link_sensor=(R@rotation_CAD).tolist(),
            T_direction='p_link = T_link_sensor @ p_sensor', **kwargs))
    depth_origin=[-20.8499,-21.06+1.32888-4.4,455.8981]
    rgb_origin=[20.7928,-22.1668+1.2578-4.4,456.]
    for name,part,point,h,v,res in [('depth_optical','B0410_supplier_native_v23',depth_origin,62.8,37.9,[240,180]),
                                  ('rgb_optical','B0196_supplier_native_v23',rgb_origin,60.,47.,[640,480])]:
        w,ht=res; fx=w/2/np.tan(np.deg2rad(h)/2); fy=ht/2/np.tan(np.deg2rad(v)/2)
        sensor(name,'head',point,optical_CAD,part_name=part,kind='depth' if name.startswith('depth')else'rgb',
            placement_basis='V20 native barrel-axis proxy + exact V22/V23 source translations; accepted R23 retains camera geometry',
            calibration_status='Nominal CAD optics only. Pixel roll, entrance pupil, lens distortion and depth/RGB registration uncalibrated.',
            resolution=res, nominal_hfov_deg=h,nominal_vfov_deg=v,
            nominal_K=[[fx,0,(w-1)/2],[0,fy,(ht-1)/2],[0,0,1]],
            intrinsic_basis='Independent manufacturer H/V FoV rectangle; not a calibrated pinhole. RGB simulation640x480 full-aspect nominal view;1080 mode crop/FoV remains unknown. Native camera uses separate fx/fy to retain both source FoVs.',
            nominal_rate_Hz=30,rate_sensitivity_Hz=[15,30],
            latency_ms=dict(capture_transport=[10,70],inference=[15,100],queue=[0,40]),
            timing_basis='Engineering assumptions on selected ROCK4D, no installed measurement',
            source_refs=[ref(ROOT/'artifacts/cove50-sensor-integration/v20/optical-summary.json'),ref(ROOT/'artifacts/cove50-sensor-integration/original-shell-layout-20261026/interface.json')],
            depth_range_m=[.2,4.] if name.startswith('depth')else None,
            depth_noise_model='Uncalibrated sensitivity: sigma=max(0.005m,0.01*range), invalid probability0.02; multipath/ambient IR/motion not calibrated' if name.startswith('depth')else None,
            shutter='Four-phase ToF depth; no assumed global simultaneous RGB exposure' if name.startswith('depth')else'Rolling shutter; row readout/exposure duration unknown')
    rxminus90=np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
    sensor('imu_board','torso',[-12.,42.,304.],rxminus90,part_name='body_IMU_4517',kind='imu',
        die_origin_board_m=None,die_rotation_board=None,
        calibration_status='Exact board transform from source build; chip lever arm and package/die axes must be registered from PCB or physical calibration. Runtime uses explicitly assumed board-aligned die.',
        nominal_rate_Hz=208,chosen_range=dict(accel_g=4,gyro_deg_s=500),
        source_noise=dict(accel_density_m_s2_sqrt_Hz=75e-6*9.80665,gyro_density_rad_s_sqrt_Hz=np.deg2rad(.0038),
            basis='LSM6DSOX DS12814 high-performance typical density at ±4g; actual filter bandwidth/config and board vibration unknown'),
        simulation_noise=dict(equivalent_bandwidth_Hz=52,accel_bias_sensitivity_m_s2=[-.196133,.196133],gyro_bias_sensitivity_rad_s=[-np.pi/180,np.pi/180]),
        transport_latency_ms=[1,12], measured_latency_ms=None,
        magnetometer='LIS3MDL physically on module; no trusted heading observation near motors without calibration',
        source_refs=[ref(ROOT/'robot/electronics/sensing/studies/purchasable/build.py')])
    sensor('foot_adc_board','torso',[-12.,42.,339.],rxminus90,part_name='foot_ADC_6494',kind='adc',
        nominal_rate_Hz=200,rate_basis='200Hz/channel simulation acquisition choice, not demonstrated serial scan timing',calibration_status='Raw voltage/counts only; FSR force curve not calibrated')
    # Native microphone port shifted with V23; board bottom-port points +CAD X.
    mic_rotation=np.column_stack([[0,1.,0],[0,0,1.],[1.,0,0]])
    sensor('microphone_port','head',[46.7,0.,464.],mic_rotation,part_name='microphone_6049_v23',kind='microphone',
        placement_basis='Source acoustic port[49.5,0,464]mm minusV23X2.8mm; entrance/shell diffraction unmeasured',
        nominal_rate_Hz=48000,rate_basis='Simulation audio interface choice; real ALSA/I2S tuple unverified',
        audio_buffer_latency_ms=[10,40],calibration_status='No acoustic or motor-noise calibration; one microphone does not establish sound-source direction')
    for s in ['R','L']:
        sign=1 if s=='R'else-1
        for site,x,y in [('toe_lateral',80,-38),('heel_lateral',80,12),('mid_medial',49,-5)]:
            sensor(s+'_'+site+'_contact',s+'_foot',[sign*x,y,1.75],np.eye(3),part_name=s+'_'+site+'_sensor',kind='FSR',
                surface_normal_CAD=[0,0,-1], nominal_rate_Hz=200,transport_latency_ms=[2,15],
                calibration_status='Nominal disk centre, no installed load-sharing/force/hysteresis calibration',
                signal='Raw ADC event/voltage; simulated contact normal force is an evaluator oracle, not measured FSR force',
                sensitivity=dict(catalog_force_N=[.2,20],catalog_hysteresis_fraction=.10,latency_basis='ADC/middleware assumptions; supplier rise time is not complete signal latency'))
    for s in ['R','L']:
        foot=next(p for p in read(OUT/'part-mass-ledger.json')['parts'] if p['name']==f'lower__{s}_B13R14_plain_sole')
        point=np.array(foot['bounds_CAD_m']).mean(0)*1000
        point[2]=np.array(foot['bounds_CAD_m'])[0,2]*1000
        sensor(s+'_sole_contact_frame',s+'_foot',point.tolist(),np.eye(3),kind='contact_frame',calibration_status='CAD sole bottom reference; no slip estimator')
    write(OUT/'sensors.json',dict(schema='cove50.sensors.v1',sensors=sensors,
        calibrated_extrinsics=0,source_sensor_modules=11,
        pose_semantics=frame_contract(),
        timing_contract=dict(required_fields=['stream_id','sequence','clock_domain','capture_time_ns','host_receive_ns','inference_end_ns','clock_map_id','clock_uncertainty_ns','pose_at_capture','pose_source','health','source_config_hash'],
            interpretation='Readout time is not conversion time; maintain mapped capture clocks, epoch and interpolation gaps. Missing calibration/pose/clock makes metric coordinates unavailable.'),
        exclusions=['No installed lidar','No selected hand/wrist sensor model','No RGB/ToF capture on target ROCK4D claimed','Simulator global pose/segmentation withheld from deployable actor observations']))
    write(OUT/'timing.json',dict(schema='cove50.timing.v1',compute='Selected ROCK4D6GB, no separate motion MCU selected',
        simulation_control_period_s=.02,control_rate_basis='50Hz provisional target; no hard real-time Linux evidence',
        physics_timestep_s=.001,flexible_physics_timestep_s=.000002,
        actuation_delay_ms=dict(nominal=20,sensitivity=[0,20,40]),
        joint_feedback_delay_ms=dict(nominal=10,sensitivity=[0,10,30]),
        bus=dict(C047_baud=1000000,XM335_baud_assumption=1000000,
            command_wire_time_ms=1.72,feedback_wire_time_ms=6.70,
            basis='Representative20Feetech sync-write position/time/speed:8+20*7=148bytes;20*(8byte read+22byte response for16byte register window)=600bytes. TwoRobotis sync-write24bytes plus2*(14byte read+21byte response)=70bytes. Total842bytes,10bits/byte at1Mbps=8.42ms before turnaround/return delays/Linux queues. Read windows, protocol bridges and actual scheduling unverified; arithmetic scenario only.'),
        timestamps='monotonic capture/acquisition clock mapped to host; no wall-clock subtraction across unsynchronized devices',
        inference_ms=dict(vision=[15,100],localization=[2,20],high_level_planning=[20,200]),
        state='All host/scheduler/inference/bus estimates exposed; supplier1ms position update is not guaranteed end-to-end1kHz control',
        qualification='No physical timing samples; joint delayed-state and image-age behavior exercised in native synthetic evidence.'))
    print(dict(actuators=len(act),sensors=len(sensors)))

if __name__=='__main__': main()
