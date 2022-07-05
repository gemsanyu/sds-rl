import csv
import sys
import pathlib
import gym

import torch as T
import numpy as np

from agent.agent import Agent
from env.sds_env import SDS_ENV
from config import define_args_parser


def select(probs, is_training=True):
    '''
    ### Select next to be executed.
    -----
    Parameter:
        probs: probabilities of each operation

    Return: index of operations, log of probabilities
    '''
    if is_training:
        dist = T.distributions.Categorical(probs)
        op = dist.sample()
        logprob = dist.log_prob(op)
    else:
        prob, op = T.max(probs, dim=1)
        logprob = T.log(prob)
    return op, logprob

if __name__ == "__main__":
    
    parser = define_args_parser()
    args = parser.parse_args(sys.argv[1:])
    agent = Agent(n_heads=8,
                 n_gae_layers=3,
                 input_dim=11,
                #  embed_dim=128,
                 embed_dim=64,
                #  gae_ff_hidden=512,
                 gae_ff_hidden=128,
                 tanh_clip=10)    
    agent_opt = T.optim.AdamW(agent.parameters(), lr=1e-4)
    # to_policy = lambda s: TimeoutPolicy(args.timeout, s)
    # dataset-8-1.json
    # dataset-128-1.json
    # dataset_root = "dataset"
    # dataset_dir = pathlib.Path(".")/dataset_root
    # dataset_dir.mkdir(parents=True, exist_ok=True)

    # dataset_name_list = ["dataset-"+num_host+"-"+idx+".json" for num_host in [8,128] for idx in range(193)+1]
    # env = SDS_ENV(workload_filename=args.workload_path, platform_path=args.platform_path)
    # features = env.reset()
    # print(env.get_mask())
    # vec_env = gym.vector.SyncVectorEnv([lambda: SDS_ENV(workload_filename=args.workload_path, platform_path=args.platform_path) for _ in range(8)])
    # features = vec_env.reset()
    # print(features.shape)
    # mask = vec_env.get_mask()
    # print(mask.shape)

    max_epoch = 10
    num_envs = 8
    max_num_steps = 64
    for epoch in range(args.max_epoch):
        mask = np.ones((num_envs, 8, 3))
        mask[:,:,2] = 0
        vec_env = gym.vector.SyncVectorEnv([lambda: SDS_ENV(workload_filename=args.workload_path, platform_path=args.platform_path) for _ in range(8)])
        features = vec_env.reset()
        sum_logprobs = T.zeros((num_envs,), dtype=T.float32) 
        sum_rewards = T.zeros((num_envs,), dtype=T.float32)
        for step in range(max_num_steps):
            features_ = T.from_numpy(features).float()
            mask_ = T.from_numpy(mask).float()
            probs, entropy = agent(features_, mask_)
            # probs = T.rand(size=mask.shape)*mask
            actions, logprobs = select(probs)
            # print(mask)
            # print(probs)
            # print(actions)
            
            # print("-------------------------")
            # print(actions, logprobs)
            features, rewards, done, mask = vec_env.step(actions)
            
            features = np.concatenate(features)
            features = features.reshape(num_envs, -1, 11)
            # print(features.shape)
            rewards = np.asanyarray(rewards)
            rewards = rewards.reshape(num_envs,)
            sum_logprobs = sum_logprobs + logprobs.sum(dim=1)
            sum_rewards = sum_rewards + rewards
            done = np.asanyarray(done)
            mask = np.asanyarray(mask)
            mask = mask.reshape(num_envs, -1, 3)
            # print(mask.shape)
            # print(features)
            # print(rewards)
            # print(done)
            # print(mask)
            # break
        print("EPOCH", epoch,":",sum_rewards.mean(), flush=True)
        critic_score = sum_rewards.mean()
        advantage = (sum_rewards-critic_score) # adv positive if rewards > mean_rewards, else negative
        loss = (sum_rewards*sum_logprobs).mean()
        loss = -loss

        agent_opt.zero_grad()
        loss.backward()
        agent_opt.step()


    # for epoch
    #     # prepare environments: select worklod vs platform path randomly

    #     for num_steps
    #         interact
    #     learn
    # features, mask = env.reset()
    # mask = T.from_numpy(mask)
    # probs = T.rand(size=mask.shape)*mask # dari 0->1 jika feasible, 0 jika enggak, ini dummy
    # actions, logprobs = select(probs)
    # state = env.step((actions, 2))
    
