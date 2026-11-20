"""Compose reviewed current-R3 entry/physics with the fresh feedback interface.

Only called by a separately released native runner. No old policy/reward state.
"""
from copy import deepcopy
from dataclasses import dataclass
from interface import Interface
from method import REWARDS, lateral_force, reward_terms, value_observation


def observation_terms(native):
    def actor_observation(env):
        action=env.action_manager.get_term('joint_pos')
        # Pinned ObservationManager probes shapes before assignment to env and
        # before any real reset. This packet is never a physical initialization,
        # cache preparation or normalizer sample. Once installed, missing actual
        # reset remains an error from Interface.observe.
        if not hasattr(env,'observation_manager'):
            return action.feedback.obs.new_zeros((env.num_envs,120))
        return action.feedback.observe(native.runtime_package().get_runtime(env),env.episode_length_buf)
    def critic_observation(env):
        action=env.action_manager.get_term('joint_pos')
        if not hasattr(env,'observation_manager'):
            return action.feedback.obs.new_zeros((env.num_envs,174))
        return value_observation(env,actor_observation(env),action.ids)
    return actor_observation,critic_observation


def build_config(native, teacher, cfg, cases, impulse_signs, *, training, seed):
    import torch
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
    from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
    BaseCfg,_,source_joint_reset,_,_,_=teacher.components(native,cfg)

    @dataclass(kw_only=True)
    class FeedbackCfg(BaseCfg):
        def build(self,env):
            action=super().build(env)
            donor=teacher.Donor(teacher.ROOT/cfg['checkpoint']['path'],cfg['joint_order'])
            goals=teacher.Goals(teacher.ROOT/cfg['goal_table']['path'],cfg['joint_order'])
            s=Interface(env.num_envs,cfg['joint_order'],teacher,donor,goals,device=env.device)
            action.feedback=s
            action.physical_delta_rate=torch.zeros_like(action._raw)
            action.total_request_rate=torch.zeros_like(action._raw)
            action.issued_force=torch.zeros((env.num_envs,1,3),device=env.device)
            signs=torch.tensor(impulse_signs,device=env.device,dtype=torch.float32)
            torso=env.scene['robot'].body_names.index('torso')
            reset,process,apply=action.reset,action.process_actions,action.apply_actions
            def reset_feedback(env_ids=None):
                ids=torch.arange(env.num_envs,device=env.device)if env_ids is None else env_ids
                reset(ids);s.reset(ids,action.bootstrap_target[ids])
                action.physical_delta_rate[ids]=0;action.total_request_rate[ids]=0
            def process_feedback(raw):
                rt=native.runtime_package().get_runtime(env)
                s.observe(rt,env.episode_length_buf)
                previous_delta=s.delta.clone();previous_request=action.requested_target.clone()
                requested=s.issue(raw,rt,env.episode_length_buf)
                process(requested) # Identical baseline valid/bootstrap and motor pipeline.
                action._raw.copy_(raw) # Own raw22, never motor targets, remains action history.
                action.physical_delta_rate.copy_((s.delta-previous_delta)/env.step_dt)
                action.total_request_rate.copy_((action.requested_target-previous_request)/env.step_dt)
            def apply_feedback():
                apply() # Arms the unchanged R3 boundary for each actual physics step.
                force=lateral_force(env.episode_length_buf,signs,torch=torch)
                action.issued_force.copy_(force)
                action._entity.write_external_wrench_to_sim(force,torch.zeros_like(force),body_ids=[torso])
            action.reset,action.process_actions,action.apply_actions=reset_feedback,process_feedback,apply_feedback
            return action

    actor_observation,critic_observation=observation_terms(native)

    robot=native.runtime_package().robot_cfg(seed=seed)
    result=native.morphology_cfg(robot,play=not training)
    result.scene.num_envs=len(cases);result.seed=seed
    result.auto_reset=training;result.episode_length_s=10.
    result.actions={'joint_pos':FeedbackCfg(entity_name='robot')}
    result.observations={'actor':ObservationGroupCfg(terms={'feedback':ObservationTermCfg(func=actor_observation)}),
        'critic':ObservationGroupCfg(terms={'training_only':ObservationTermCfg(func=critic_observation)})}
    result.commands={};result.curriculum={};result.metrics={}
    result.rewards={name:RewardTermCfg(func=lambda env,name=name:reward_terms(env)[name],weight=spec['weight'])
        for name,spec in REWARDS.items()}
    result.scene.entities['robot'].init_state.joint_pos={name:0. for name in cfg['joint_order']}
    result.scene.entities['robot'].init_state.pos=tuple(cfg['root_position_m'])
    result.scene.sensors+=(ContactSensorCfg(name='nonfoot_ground_contact',
        primary=ContactMatch(mode='body',pattern=r'^(?!(R|L)_foot$).*$',entity='robot'),
        secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1),)
    base=result.events['reset_base'];base_reset=base.func;base_params=deepcopy(base.params)
    def each(ids,env):return torch.arange(env.num_envs,device=env.device)if ids is None else ids
    def reset_base(env,env_ids=None):
        for i in each(env_ids,env).tolist():
            params=deepcopy(base_params);params['velocity_range']={}
            params['pose_range']={k:(v,v)for k,v in zip(('roll','pitch','yaw'),cases[i]['root_rpy_rad'])}
            base_reset(env,torch.tensor([i],device=env.device),**params)
    def reset_joints(env,env_ids=None):
        if not hasattr(env,'_feedback_resets'):env._feedback_resets={}
        for i in each(env_ids,env).tolist():
            env._coordinated_case=cases[i]
            source_joint_reset(env,torch.tensor([i],device=env.device))
            env._feedback_resets[i]=deepcopy(env._coordinated_reset)
    result.events={name:result.events[name]for name in ('reset_base','reset_robot_joints')}
    result.events['reset_base'].func=reset_base;result.events['reset_base'].params={}
    result.events['reset_robot_joints'].func=reset_joints;result.events['reset_robot_joints'].params={}
    return result
