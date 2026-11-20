"""Compact native measurements, provenance and qualitative model assessment."""
import shutil
from common import *

def main():
    replay=read(OUT/'replay-summary.json');validation=read(OUT/'native-validation.json')
    numerical=read(OUT/'numerical-sensitivity.json')
    perception=read(OUT/'perception-validation.json');mass=read(OUT/'link-inertias.json')
    dimensions=[
        dict(dimension='Geometry and joint contract',grade=4.2,confidence='medium',trend='improving',reason='853 parts and22 source axes retained; native rigid/flexible/URDF FK agree. Source R/L convention and six unknown hip ranges explicit.'),
        dict(dimension='Mass and inertia',grade=3.0,confidence='low',trend='improving',reason='Per-part native volume moments and retained supplier allocations, with conditional3.830 kg total. About0.814 kg reference skin and many module masses estimated; no weighed robot.'),
        dict(dimension='Actuator dynamics',grade=2.5,confidence='medium',trend='improving',reason='Exact C047/XM335 supplier curves, gear/current/encoder properties retained. Real gains/friction/rotor inertia/thermal behavior unmeasured; XM335 30 g tensor discrepancy not silently applied.'),
        dict(dimension='Connector compliance',grade=3.2,confidence='medium',trend='improving',reason='18 coupled FEM matrices exported,16 native receiving interfaces active. Native spring forces match matrices. Clamp/material/damping, finite motion, fixed receiving contact proxies and numerical step sensitivity limit dynamics.'),
        dict(dimension='Collision/contact fidelity',grade=2.4,confidence='medium',trend='improving',reason='592 convex decomposed shapes remove most filled-torso artifacts; neutral native contacts up to2.042mm (including nominal floor placement), voxel/skin approximation and unmeasured friction remain.'),
        dict(dimension='Learning interface',grade=3.4,confidence='medium',trend='improving',reason='Actual MJLab 22-actuator Entity compiles; CPU 75-value timed observation/target interface exercised. Old72/21actor is incompatible; diagnostic old-trajectory mapping loses balance, no new learned controller.'),
        dict(dimension='Perception/localization model',grade=3.1,confidence='medium',trend='improving',reason='Native RGB/depth,13 frames and capture-time transforms exercised over61 head poses; calibration/noise/clock/target-host latency remain partly assumed, no walking estimator qualified.'),
        dict(dimension='Physical qualification',grade=.5,confidence='high',trend='flat',reason='No fabricated robot, bench identification, physical sensing, installed thermal or timing measurements in this task.')]
    matrix=[]
    for variant,condition in [('rigid','nominal'),('rigid','low-rail-delay'),('compliant','nominal')]:
        rows=[r for r in replay['episodes']if r['variant']==variant and r['condition']['name']==condition]
        matrix.append(dict(condition=variant+' / '+condition,episodes=len(rows),start_families=2,
            rms_tracking_error_rad=[r['rms_joint_tracking_error_rad']for r in rows],
            minimum_root_height_m=[r['minimum_root_height_m']for r in rows],
            mean_catalog_current_estimate_A=[r['mean_catalog_current_sum_A']for r in rows],
            maximum_connector_translation_mm=[r['maximum_connector_translation_m']*1000 for r in rows],
            maximum_connector_rotation_deg=[float(np.rad2deg(r['maximum_connector_rotation_rad']))for r in rows]))
    power=read(ROOT/'robot/electronics/power/battery.json')
    write(OUT/'power-and-randomization.json',dict(selected_battery=power,source=ref(ROOT/'robot/electronics/power/battery.json'),
        source_power_interfaces=ref(CAD/'interfaces-and-open-work.json'),
        battery_mass='639g cell maxima included once across9source cell bodies; mounting/circuit/harness estimates are separate part rows',
        supply='C047 raw3S planning; neck requires regulated rail at/below12V (historical11V target). Selected R33 is not a wired/qualified full power installation.',
        native_replay='12V and10V are fixed catalog-reference test rails; no battery sag/SOC/return-energy waveform is solved by these episodes',
        electronics_output_reserve_W=dict(sustained=28,peak=45),reserve_count='One aggregate reserve, never additive twice',
        estimated_runtime_minutes=None,
        runtime_limit='Current robot mass and native motor-current sensitivity do not identify a useful full-to-empty gait duty. Reconcile measured or credible new-body duty in the remaining power task.',
        uncalibrated_training_sensitivities=dict(rail_V=[10,12],command_delay_s=[0,.02,.04],feedback_delay_s=[0,.01,.03],
            Coulomb_joint_friction_Nm=[0,.02],viscous_joint_friction_Nm_s_rad=[0,.01],
            connector_stiffness_multiplier=[.5,1.,2.],assumed_modal_damping_fraction=[.02,.10,.20],
            sole_sliding_friction=[.4,.8,1.2],mass_source='Per-part engineering bounds in part-mass-ledger; unmeasured, not independent calibrated probability distributions'),
        calibration_priorities=['Weigh populated link assemblies and whole robot; record CoM/inertia identification where practical',
            'Measure motor position/current/velocity under several loads/voltages/temperatures; actual zero, signs and stop travel first',
            'Test connector source fixtures with real fasteners/contact; dynamic ring-down needed for damping',
            'Calibrate RGB/ToF intrinsics, pixel roll and inter-camera transform; register IMU die axes/lever arm and six FSR load paths',
            'Measure capture/transport/compute/application times under simultaneous perception/control workload and clock resets',
            'Establish sole friction/contact and cable forces; replace broad randomization with measured uncertainty']))
    report=dict(schema='cove50.learning-model-report.v1',version=VERSION,
        observation='A reproducible22-axis learning and perception model now matches the selected electronics CAD. Native rigid and FEM-flexible bodies load; the lossy older gait-trajectory diagnostic falls, so source-model completion does not establish a transferred walking policy.',
        source_CAD=ref(CAD/'assembly-parts.json'),source_versions=read(OUT/'source-lock.json')['selected_versions'],
        native=validation['models'],mass=mass,
        active_actuators=22,exported_FEM_matrices=18,active_FEM_interfaces=16,passive_FEM_coordinates=96,
        sensor_frames=13,physical_sensor_modules=11,measured_physical_calibrations=0,
        matrix=matrix,numerical_sensitivity=numerical,grades=dict(overall='A usable provisional model with source-bound geometry and explicit physical uncertainty; new-body control adaptation and physical calibration remain necessary.',dimensions=dimensions),
        replay_counts=dict(episodes=6,independent_start_families=2,simultaneous_world_slots=1,
            native_physics_steps=sum(r['physics_steps']for r in replay['episodes']),recorded_1ms_samples=sum(r['recorded_samples']for r in replay['episodes']),
            target_updates=240,new_actor_inferences=0,optimizer_updates=0,new_policy_checkpoints=0),
        behavior='All six prescribed-target episodes lose upright support within the0.8s horizon. Exact per-condition tracking/current/flex measurements are in matrix. Source-family differences, contact proxy, numerical step and untrained adaptation limit causal attribution. Flexible receiving offsets challenge small-deflection interpretation; no convergence or identified current claim.',
        perception=dict(captures=61,projection_roundtrip_max_m=max(r['max_error_m']for r in perception['projection_roundtrip']),
            latency_conditions=perception['latency_conditions'],physical_localization='Not measured; exact simulated capture pose supplied for transform checks'),
        important_limits=['Unknown hip travel, electrical signs/zeros, hardware servos/thermal duty and end-to-end host timing',
            'Conditional mass includes unqualified reference skins and150g unfinished-installation allowance',
            'No selected articulated hand/gripper added; wrist compliance remains data only',
            'V-HACD contacts and ideal-clamp FEM approximate physics; receiving case/horn contact hulls stay on source parent links in flex variant; severe fields cannot imply strength',
            'Compliant2us default is a numerical experiment, not a convergence claim; supplemental1us/5us traces retained separately',
            'No old policy checkpoint compatibility, stable new walking, real camera capture or SLAM qualification'],
        source_refs=[ref(OUT/f)for f in ['source-lock.json','part-mass-ledger.json','FEM-geometry-identity.json','actuators.json','compliance.json','sensors.json','timing.json','native-validation.json','replay-summary.json','trajectory-mapping.json','perception-validation.json','mjlab-entity-interface.json','numerical-sensitivity.json']])
    write(OUT/'report.json',report)
    selected=dict(schema='cove50.current-learning-model.v1',version=VERSION,status='provisional_selected_geometry_learning_model',
        source_study='cove50-electronics-integrated',source_result='cove50-electronics-integrated-r1-20261026',
        source_manifest=ref(CAD/'assembly-parts.json'),rigid_URDF=ref(OUT/'cove50.urdf'),
        rigid_MJCF=ref(OUT/'cove50-rigid.xml'),compliant_MJCF=ref(OUT/'cove50-compliant.xml'),
        actuators=ref(OUT/'actuators.json'),sensors=ref(OUT/'sensors.json'),report=ref(OUT/'report.json'),
        mass_estimate_kg=mass['total_mass_kg'],actuated_axes=22,
        scope='Current model for new-body learning/perception experiments. Historical training bodies/checkpoint guards and selected Electronics CAD remain unchanged. No physical qualification or working policy is implied.')
    write(ROOT/'robot/model/current-model.json',selected);write(OUT/'current-model.json',selected)
    write(DATA/'summary.json',dict(version=VERSION,source_manifest=selected['source_manifest'],
        current_model=ref(ROOT/'robot/model/current-model.json'),report=ref(OUT/'report.json'),
        mass_estimate_kg=mass['total_mass_kg'],matrix=matrix,grades=report['grades'],replay_counts=report['replay_counts']))
    print(dict(mass_kg=mass['total_mass_kg'],matrix=matrix,replay_counts=report['replay_counts']))

if __name__=='__main__':main()
