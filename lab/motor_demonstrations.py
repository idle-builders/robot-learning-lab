"""Training-only teacher and native-regressor to PPO weight conversion.

Never imported by learned motor inference. The teacher's commands pass through
the same zero-feedforward interface as the learned actor.
"""
import numpy as np

def teacher_action(interface):
    m=interface.motor
    goal=interface.goal
    pitch,_,gyro,x,velocity,_=m.motion()
    trim,feedforward=m.model_reference()
    reference=m.reference_x+goal[14]*.02
    gains=m.settings['balance']
    speed=(gains['pitch_gain_m_s']*(pitch-trim)+gains['pitch_rate_gain_m']*gyro[1]
           +gains['position_gain_s_inv']*(x-reference)
           +gains['velocity_gain']*(velocity-goal[14])+goal[14])
    action=np.zeros(16)
    # Convert joint feedforward torque to a position-servo offset. This is a
    # bounded approximation after rate limiting, not algebraic plant parity.
    action[interface.positions]=feedforward[interface.positions]/m.pack.kp[interface.positions]/.4
    for side,sign in [('left',-1),('right',1)]:
        i=m.order.index(side+'_wheel_joint')
        action[i]=(speed/.04+sign*(goal[15]-gyro[2]))/m.limits[i]
    return np.clip(action,-1,1).astype(np.float32)

def copy_regressor_to_actor(regressor,policy):
    """Standard dense tanh MLP only; preserve native PPO policy and optimizer."""
    import torch
    layers=[layer for layer in policy.mlp_extractor.policy_net if isinstance(layer,torch.nn.Linear)]
    activations=[layer for layer in policy.mlp_extractor.policy_net if not isinstance(layer,torch.nn.Linear)]
    layers.append(policy.action_net)
    if regressor.activation!='tanh' or regressor.out_activation_!='identity' or not all(isinstance(a,torch.nn.Tanh) for a in activations):
        raise ValueError('Only equivalent tanh / linear MLP conversion is supported')
    if len(layers)!=len(regressor.coefs_):
        raise ValueError('Regressor and actor layer count mismatch')
    # Validate all shapes before changing any weights.
    for layer,w,b in zip(layers,regressor.coefs_,regressor.intercepts_,strict=True):
        if tuple(layer.weight.shape)!=w.T.shape or tuple(layer.bias.shape)!=b.shape:
            raise ValueError('Regressor and actor weight shape mismatch')
    with torch.no_grad():
        for layer,w,b in zip(layers,regressor.coefs_,regressor.intercepts_,strict=True):
            layer.weight.copy_(torch.as_tensor(w.T,dtype=layer.weight.dtype))
            layer.bias.copy_(torch.as_tensor(b,dtype=layer.bias.dtype))
