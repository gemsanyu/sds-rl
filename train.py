from random import randint
from time import time
import gym
import torch as T
import numpy as np
from tqdm import tqdm

from env.sds_env import SDS_ENV
from config import get_args
from utils import select, learn
from setup import setup
from validation import validate

NUM_DATASETS = 1000

if __name__ == "__main__":
    args = get_args()
    agent, critic, agent_opt, critic_opt, memory, last_epoch, best_validation_value, checkpoint_path, writer = setup(args)

    # 1. edit enviromentnya
    # 2. edit setup dan save checkpoint untuk save epoch
    # 3. lanjut siapin validasi
    # 4. remove nested loop gae
    # start training
    step = 0
    for epoch in range(last_epoch, args.max_epoch):
        print("EPOCH", epoch)
        dataset_idxs = [randint(0,NUM_DATASETS) for _ in range(args.num_envs+args.num_validation_envs)]

        # 8 train, 2 validasi
        training_idxs = dataset_idxs[:args.num_envs]
        validation_idxs = dataset_idxs[args.num_envs:]
        # batching dataset
        # pisah dataset training validation
        # init training environment
        env_fns = [lambda: SDS_ENV(dataset_idx=train_idx) for train_idx in training_idxs]
        env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
    
        val_env_fns = [lambda: SDS_ENV(dataset_idx=val_idx) for val_idx in validation_idxs]
        validation_env = gym.vector.AsyncVectorEnv(val_env_fns, shared_memory=False)
    
        # mulai generate experience dari training environments
        mask = np.ones((args.num_envs, 128, 3))
        mask[:,:,2] = 0
        features = env.reset()        
        for it in range(args.max_training_steps):
            # rollout / solve env/ run episode -> gather experiences..
            
            with T.no_grad():
                agent.eval()
                features_ = T.from_numpy(features).to(agent.device).float()
                mask_ = T.from_numpy(mask).to(agent.device).float()
                probs, entropy = agent(features_, mask_)
                actions, logprobs = select(probs)
                new_features, rewards, done, new_mask = env.step(actions)
                critic_vals = critic(features_)
                memory.store_memory(features, mask, actions, logprobs, critic_vals, rewards, done)
                features = new_features
                features = np.concatenate(features)
                features = features.reshape(args.num_envs, -1, 11)
                mask = new_mask
                mask = np.asanyarray(mask)
                mask = mask.reshape(args.num_envs, -1, 3)

            if step > 0 and step % args.training_steps == 0:
                learn(args, agent, agent_opt, critic, critic_opt, memory)
                memory.clear_memory()
                best_validation_value = validate(args, 
                        agent.state_dict(),
                        agent_opt.state_dict(), 
                        critic.state_dict(), 
                        critic_opt.state_dict(), 
                        best_validation_value, 
                        checkpoint_path,
                        last_epoch,
                        validation_env,
                        writer,
                        parallel=False)
            step+=1
