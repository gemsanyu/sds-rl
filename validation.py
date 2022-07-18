
# note only the validation results, others are logged in main loop(?)
import pathlib

import numpy as np
import torch as T

from agent.agent import Agent
from env.sds_env import SDS_ENV
from utils import select, compute_objective

def validate(args,
            agent_state_dict, 
            agent_opt_state_dict, 
            critic_state_dict,
            critic_opt_state_dict,
            best_validation_value,
            checkpoint_path,
            last_step,
            env:SDS_ENV,
            writer,
            parallel=False):
    agent = Agent(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden,
                 tanh_clip=args.tanh_clip,
                 device=args.device)   
    agent.load_state_dict(agent_state_dict)
    agent.eval()
    mask = np.ones((args.num_validation_envs, 128, 3))
    mask[:,:,2] = 0
    print("validation init")
    features = env.reset()
    print("TEST")
    result_list = []
    is_env_done = [False for _ in range(args.num_validation_envs)]
    while len(result_list) < args.num_validation_envs:
        # rollout / solve env/ run episode -> gather experiences..
        with T.no_grad():
            agent.eval()
            features_ = T.from_numpy(features).to(agent.device).float()
            mask_ = T.from_numpy(mask).to(agent.device).float()
            check = T.logical_or(T.isnan(features_), T.isinf(features_))
            check = check.sum(dim=1)
            print(check)
            probs, entropy = agent(features_, mask_)
            actions, logprobs = select(probs)
            new_features, rewards, done_list, new_mask = env.step(actions)
            print(done_list)
            for i, done in enumerate(done_list):
                if done:
                    is_env_done[i] = True
                    result_list += [rewards[i]]
            features = new_features
            features = np.concatenate(features)
            features = features.reshape(args.num_validation_envs, -1, 11)
            mask = new_mask
            mask = np.asanyarray(mask)
            mask = mask.reshape(args.num_validation_envs, -1, 3)

    energy_usage = 0
    mean_slowdown_time = 0
    obj = 0
    for result in result_list:
        energy_usage_, mean_slowdown_time_, obj_ = result
        energy_usage += (energy_usage_/args.num_validation_envs)
        mean_slowdown_time += (mean_slowdown_time_/args.num_validation_envs)
        obj += (obj_/args.num_validation_envs)

    callback_input = agent_state_dict, agent_opt_state_dict, critic_state_dict,\
        critic_opt_state_dict, best_validation_value, checkpoint_path, energy_usage,\
        mean_slowdown_time, obj, last_step, writer
        
    if parallel:
        return callback_input
    else:
        return validation_callback(callback_input)


def validation_callback(callback_input):
    agent_state_dict, agent_opt_state_dict, critic_state_dict,\
        critic_opt_state_dict, best_validation_value, checkpoint_path, energy_usage,\
        mean_slowdown_time, obj, step, writer = callback_input
    write_validation_result(energy_usage, mean_slowdown_time, obj, step, writer)
    return save_checkpoint(agent_state_dict, agent_opt_state_dict, critic_state_dict,\
        critic_opt_state_dict, best_validation_value, obj, step, checkpoint_path)

def save_checkpoint(agent_state_dict, 
                    agent_opt_state_dict, 
                    critic_state_dict,
                    critic_opt_state_dict, 
                    best_validation_value,
                    obj,
                    step,
                    checkpoint_path:pathlib.Path):
    checkpoint = {
                    "agent_state_dict": agent_state_dict,
                    "agent_opt_state_dict": agent_opt_state_dict,
                    "critic_state_dict": critic_state_dict,   
                    "critic_opt_state_dict":critic_opt_state_dict,
                    "step":step
                }
    if best_validation_value is None or best_validation_value > obj:
        best_validation_value = obj
        checkpoint["best_val_value"] = best_validation_value
        best_checkpoint_path = checkpoint_path.parent/(checkpoint_path.name+"_best")
        T.save(checkpoint, best_checkpoint_path)
    else:
        checkpoint["best_val_value"] = best_validation_value
    T.save(checkpoint, checkpoint_path)
    return best_validation_value
    

def write_validation_result(energy_usage, mean_slowdown_time, obj, step, writer):
    # obj = weighted sum energy, turnaround
    writer.add_scalar("Energy Usage", energy_usage, step)
    writer.add_scalar("Mean Slowdown Time", mean_slowdown_time, step)
    writer.add_scalar("Obj Function", obj, step)
