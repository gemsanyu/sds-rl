import torch as T
import numpy as np
import pathlib

from env.sds_env import SDS_ENV
from config import get_args
from utils import select, learn
from setup import setup

NUM_DATASETS = 1000

def save_checkpoint(agent_state_dict, 
                    agent_opt_state_dict, 
                    critic_state_dict,
                    critic_opt_state_dict, 
                    epoch,
                    step,
                    checkpoint_path:pathlib.Path):
    checkpoint = {
                    "agent_state_dict": agent_state_dict,
                    "agent_opt_state_dict": agent_opt_state_dict,
                    "critic_state_dict": critic_state_dict,   
                    "critic_opt_state_dict":critic_opt_state_dict,
                    "epoch":epoch,
                    "step":step
                }
    T.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    args = get_args()
    agent, critic, agent_opt, critic_opt, memory, last_epoch, last_step, checkpoint_path, writer = setup(args)
    # start training
    # 1 epoch = 1 full training data,, not the epoch commonly understood (?)
    # init training environment
    step = last_step
    args.num_envs = 1
    for epoch in range(last_epoch, args.max_epoch):
        # mulai generate experience dari training environments
        env = SDS_ENV(dataset_name=args.dataset_name, batsim_verbosity="quiet", is_test=False, alpha=args.alpha, beta=args.beta)
        mask = np.ones((args.num_envs, 128, 2))
        mask[:,:,1] = 0
        features = env.reset() 
        features = features.reshape(args.num_envs, -1, 11)
        done = False
        saved_logprobs = []
        saved_rewards = []
        agent.train()
        while not done:
            features_ = T.from_numpy(features).to(agent.device).float()
            mask_ = T.from_numpy(mask).to(agent.device).float()
            if not T.any(mask_):
                env.simulator.proceed_time(time=1800)
                done = env.is_really_running
                features = env.get_features(env.simulator.current_time)
                mask = env.get_mask()
                continue
            probs, entropy = agent(features_, mask_)
            need_decision_idx = T.any(mask_, dim=2).nonzero()[:,1]
            print(need_decision_idx)
            exit()
            actions, logprobs = select(probs)
            new_features, rewards, done, info = env.step(need_decision_idx, actions)
            critic_vals = critic(features_)
            if not done:
                features = new_features
                features = np.concatenate(features)
                features = features.reshape(args.num_envs, -1, 11)
                new_mask, wasted_energy, waiting_time_since_last_dt = info
                mask = new_mask
                mask = np.asanyarray(mask)
                mask = mask.reshape(args.num_envs, -1, 3)

            #log important values
            writer.add_scalar("Reward", rewards, step)
            writer.add_scalar("Wasted Energy Reward", wasted_energy, step)
            writer.add_scalar("Waitim Time Reward", waiting_time_since_last_dt, step)
            writer.add_scalar("Consume Joules", env.host_monitor.info["consumed_joules"], step)
            writer.add_scalar("Wasted Energy", env.host_monitor.info["energy_waste"], step)
            writer.add_scalar("Time Computing", env.host_monitor.info["time_computing"], step)
            writer.add_scalar("Time Idle", env.host_monitor.info["time_idle"], step)
            writer.add_scalar("Time Switching On", env.host_monitor.info["time_switching_on"], step)
            writer.add_scalar("Time Switching Off", env.host_monitor.info["time_switching_off"], step)
            writer.add_scalar("Time Sleeping", env.host_monitor.info["time_sleeping"], step)
            writer.add_scalar("Number of Switching State", env.host_monitor.info["nb_switches"], step)
                
            if step > 0 and len(saved_logprobs) >= args.training_steps:
                
                save_checkpoint(agent.state_dict(), agent_opt.state_dict(), critic.state_dict(), critic_opt.state_dict(), epoch, step, checkpoint_path)
            step+=1
