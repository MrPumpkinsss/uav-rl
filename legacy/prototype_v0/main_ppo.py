
import torch
import numpy as np
from gitdb.util import mkdir

from ppo import PPOAgent
from uav_env import UAVDeploymentEnv
from torch.utils.tensorboard import SummaryWriter
import json
from noise_ppl import inference_with_ppl
import random

# 配置参数：5 UAVs，34 层模型
config = {
    'N': 5,
    'L': 32,
    'cl': [1.0] * 32,
    'ol': [1.0] * 31,
    'fn': [1e9] * 5,
    'pn': [0.1] * 5,
    'Bmax': 20.0,
    'k0': 1e-28,
    'Phov': 0.5,
    'N0': 1e-9,
    'Emax': [10.0] * 5,
    'alpha': 1e-5,
    'beta': 1.0,
    'PPL': 5.0
}

env = UAVDeploymentEnv(config)
state_dim = config['N'] * config['N']
agent = PPOAgent(state_dim=state_dim, hidden_dim=256,
                 num_layers=config['L']-1, num_uavs=config['N'])

log_dir = 'checkpoints_infocom_7_20/'
run_dir = log_dir + 'logs/'

writer = SummaryWriter(str(log_dir))
log_dict = {'rewards': [], 'latency': [], 'actions': []}

episodes = 10000
for ep in range(episodes):
    print('Episode', ep)
    state = env.reset()
    state = env.reset()
    state_tensor = torch.FloatTensor(state)
    action_tensor, log_prob, _, action_idxs, action_uavs = agent.get_action(state_tensor)

    # 还原为 LxN → one-hot → NxL
    action_flat = action_tensor.numpy().reshape(config['L']-1, config['N']*2)
    action_np = action_flat.T

    ppl_config={}
    noise_flag=0
    action_bin=np.zeros([config['N'], config['L']])
    for i,layer_idx, uav_idx in zip(range(len(action_idxs)),action_idxs, action_uavs):
        print(layer_idx, uav_idx)
        noise=random.random()
        if noise_flag==0:
            print()
            ppl_config[layer_idx]=noise
            print('layer idx: ', layer_idx, ', noise: ', noise)
        else:
            ppl_config[layer_idx]=noise_flag
            print('layer idx: ', layer_idx, ', noise: ', noise_flag)
        if i != len(action_idxs) - 1:
            next_uav_idx = action_uavs[i + 1]
            if next_uav_idx == uav_idx:
                if noise_flag == 0:
                    noise_flag = noise
            else:
                noise_flag = 0
        #construct the action_bin
        if i==0:
            action_bin[uav_idx, :layer_idx] =1
        else:
            previous_layer_idx=action_idxs[i-1]
            action_bin[uav_idx, previous_layer_idx:layer_idx] =1

    ppl_delay = inference_with_ppl(ppl_config)

    next_state, reward = env.step(action_bin,ppl_delay)
    agent.update(state_tensor.unsqueeze(0),
                 action_tensor.unsqueeze(0),
                 log_prob.unsqueeze(0), [reward])

    writer.add_scalar("Reward/Total", reward, ep)
    writer.add_scalar("Reward/StepAvg", reward / (config['L']-1), ep)
    writer.add_scalar("Latency/Inference", ppl_delay, ep)
    writer.add_scalar("Latency/StepAvg", ppl_delay / (config['L']-1), ep)

    log_dict['rewards'].append(reward)
    log_dict['latency'].append(config['PPL'])
    log_dict['actions'].append(action_np.tolist())

    if (ep + 1) % 10 == 0:
        print(f"[Episode {ep+1}] Reward: {reward:.2f}, Latency: {config['PPL']}")

torch.save(agent.policy.state_dict(), "ppo_model.pth")
with open("training_logs.json", "w") as f:
    json.dump(log_dict, f)

writer.close()
