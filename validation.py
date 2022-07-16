
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
            validation_env:SDS_ENV,
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
    mask = np.ones((1, 128, 3))
    mask[:,:,2] = 0
    features = validation_env.reset()
    # features = features.reshape(1, -1, 11)
    done = False
    validation_step=0
    while not done:
        with T.no_grad():
            features_ = T.from_numpy(features).to(agent.device).float()
            mask_ = T.from_numpy(mask).to(agent.device).float()
            probs, _ = agent(features_, mask_)
            actions, _ = select(probs)
            actions = actions.squeeze(0)
            new_features, _, done, new_mask = validation_env.step(actions)
            features = new_features
            features = np.concatenate(features)
            features = features.reshape(2, -1, 11)
            mask = new_mask
            mask = np.asanyarray(mask)
            mask = mask.reshape(2, -1, 3)
        validation_step += 1


    energy_usage, mean_slowdown_time, obj = compute_objective(validation_env.simulation_monitor, validation_env.simulator)
    callback_input = agent_state_dict, agent_opt_state_dict, critic_state_dict,\
        critic_opt_state_dict, best_validation_value, checkpoint_path, energy_usage,\
        mean_slowdown_time, obj, last_step, writer
        
    if parallel:
        return callback_input
    else:
        validation_callback(callback_input)


def validation_callback(callback_input):
    agent_state_dict, agent_opt_state_dict, critic_state_dict,\
        critic_opt_state_dict, best_validation_value, checkpoint_path, energy_usage,\
        mean_slowdown_time, obj, step, writer = callback_input
    write_validation_result(energy_usage, mean_slowdown_time, obj, step, writer)
    save_checkpoint(agent_state_dict, agent_opt_state_dict, critic_state_dict,\
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
        checkpoint["best_val_value"] = obj
        best_checkpoint_path = checkpoint_path.parent/(checkpoint_path.name+"_best")
        T.save(checkpoint, best_checkpoint_path)
    else:
        checkpoint["best_val_value"] = best_validation_value
    T.save(checkpoint, checkpoint_path)
    

def write_validation_result(energy_usage, mean_slowdown_time, obj, step, writer):
    # obj = weighted sum energy, turnaround
    writer.add_scalar("Energy Usage", energy_usage, step)
    writer.add_scalar("Mean Slowdown Time", mean_slowdown_time, step)
    writer.add_scalar("Obj Function", obj, step)
