"""Every full/terminated/censored episode, causal inputs and actual reset state."""
import json
from collections import defaultdict


def make(native, configuration):
    import numpy as np
    import torch
    class Episodes(native.Episodes):
        def __init__(self,cfg,env):
            super().__init__(cfg,env);self.reset_serial=0;self.initial_records={}

        def cache_inputs(self,value):
            env=self._env;rt=native.runtime_package().get_runtime(env)
            action=env.action_manager.get_term('joint_pos');s=action.feedback
            s.observe(rt,env.episode_length_buf)
            values=dict(input_actor120=env.obs_buf['actor'],input_value174=env.obs_buf['critic'],
                input_value_prediction=value,input_previous_raw22=s.previous_raw,
                input_beta=s.obs[:,50:51],input_beta_dot=s.obs[:,51:52],
                input_motor_reference22_rad=s.motor_reference,input_pose_goal22_rad=s.pose_goal,
                input_sensor_valid=rt.valid,input_sensor_capture_s=rt.capture_time_s,
                input_imu_capture_s=rt.imu_capture_time_s,input_sensor_age_s=s.obs[:,119],
                input_local_control=env.episode_length_buf,input_reset_generation=s.generation)
            self.inputs={k:v.detach().cpu().numpy().copy()for k,v in values.items()}

        def record_post_reset(self,env_ids):
            super().record_post_reset(env_ids)
            env=self._env;rt=native.runtime_package().get_runtime(env)
            action=env.action_manager.get_term('joint_pos')
            for i in env_ids.tolist():
                self.reset_serial+=1
                values=dict(qpos=env.sim.data.qpos[i],qvel=env.sim.data.qvel[i],
                    bootstrap_target22_rad=action.bootstrap_target[i],
                    previous_raw22=action.feedback.previous_raw[i],
                    runtime_generator_state=rt.generator.get_state())
                for name in ('tick','pending_reset','target_history','target','initial_target','previous_command',
                        'gravity_estimate','sample_gyro','accel_bias','gyro_bias','next_imu','next_joint','last_imu_time',
                        'seq','packet_values','packet_capture','packet_imu_capture','last_values','capture_time_s',
                        'imu_capture_time_s','valid'):
                    values['runtime_'+name]=getattr(rt,name)[i]
                path=self.directory/f'initial-{self.reset_serial:06d}-slot-{i:04d}.npz'
                np.savez_compressed(path,**{k:v.detach().cpu().numpy().copy()for k,v in values.items()})
                self.initial_records[i]=dict(trace=path.name,reset=env._feedback_resets[i],
                    common_counter=env.common_step_counter,generation=int(action.feedback.generation[i]))

        def capture(self,ids):
            env=self._env;rt=native.runtime_package().get_runtime(env)
            action=env.action_manager.get_term('joint_pos');s=action.feedback
            data=env.scene['robot'].data
            values=dict(qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,
                root_velocity_body_m_s=data.root_link_lin_vel_b,
                root_ang_velocity_body_rad_s=data.root_link_ang_vel_b,
                measured_q22_rad=data.joint_pos[:,action.ids],measured_qd22_rad_s=data.joint_vel[:,action.ids],
                raw_residual22=env.action_manager.action,physical_delta22_rad=s.delta,
                physical_delta_rate22_rad_s=action.physical_delta_rate,total_request_rate22_rad_s=action.total_request_rate,
                requested_target22_rad=action.requested_target,accepted_target22_rad=rt.previous_command,applied_target22_rad=rt.target,
                torque22_Nm=rt.torque,current22_A=rt.current,saturated22=rt.saturated,
                substep_peak_abs_torque22_Nm=action.peak_abs_torque,substep_peak_current22_A=action.peak_current,
                substep_saturation_fraction22=action.saturation_sum/max(1,action.interval_substeps),
                physical_external_force_world_N=action.issued_force,feet_force_N=env.scene['feet_ground_contact'].data.force,
                nonfoot_force_N=env.scene['nonfoot_ground_contact'].data.force,
                distal_forearm_force_N=env.scene['hand_ground_contact'].data.force,
                runtime_time_s=rt.tick*rt.physics_dt_s,native_time_s=env.sim.data.time,
                reward=env.reward_buf,reward_terms=env.reward_manager._step_reward,
                source_terminated=env.reset_terminated,source_timeout=env.reset_time_outs)
            arrays={k:v[ids].detach().cpu().numpy().copy()for k,v in values.items()}
            for j,i in enumerate(ids.tolist()):
                if self.inputs is None:raise RuntimeError('Missing actual pre-action inputs')
                for key,value in self.inputs.items():self.buffers[i][key].append(value[i])
                for key,value in arrays.items():self.buffers[i][key].append(value[j])
                self.buffers[i]['captured_physics_substeps'].append(action.interval_substeps)
                self.transitions+=1

        def finish(self,i,reason):
            b=self.buffers.get(i)
            if not b or not b['qpos']:return
            self.serial+=1;name=f'episode-{self.serial:06d}-slot-{i:04d}'
            arrays={k:np.asarray(v)for k,v in b.items()}
            arrays.update(initial_qpos=self.initial[i],initial_qvel=self.initial_velocity[i])
            path=self.directory/(name+'.npz');np.savez_compressed(path,**arrays)
            row=dict(id=name,world_slot=i,trace=path.name,terminal_reason=reason,
                initialization=self.initial_records[i],case=configuration['cases'][i],
                controls=len(arrays['qpos']),duration_s=len(arrays['qpos'])*env_dt(self._env),
                source_termination_observed=bool(arrays['source_terminated'].any()),
                source_timeout_observed=bool(arrays['source_timeout'].any()),
                input_width=120,training_only_value_width=174,raw_width=22,physical_width=22,
                joint_order=configuration['joint_order'],reward_terms=self._env.reward_manager.active_terms,
                semantics='Input columns are issued-request context for every ordinary/terminal row; POST state reached after80substeps. Full noreset replays keep sourceending annotations/floor tails.')
            with self.manifest.open('a')as stream:stream.write(json.dumps(row)+'\n')
            if reason.endswith('censoring'):self.censored+=1
            else:self.completed+=1
            self.buffers[i]=defaultdict(list)
    return Episodes


def env_dt(env):return env.step_dt
