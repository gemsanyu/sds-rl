import os
import pathlib

import gym
import torch as T

from agent.agent import Agent
from agent.critic import Critic
from agent.memory import PPOMemory
from env.sds_env import SDS_ENV

def setup(args):
    agent = Agent(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden,
                 tanh_clip=args.tanh_clip)    
    agent_opt = T.optim.Adam(agent.parameters(), lr=args.lr)
    critic = Critic(n_heads=args.n_heads,
                 n_gae_layers=args.n_gae_layers,
                 input_dim=11,
                 embed_dim=args.embed_dim,
                 gae_ff_hidden=args.gae_ff_hidden)
    critic_opt = T.optim.Adam(critic.parameters(), lr=args.lr)

    memory = None
    # load checkpoint if exists
    checkpoint_root = pathlib.Path(".")/"checkpoints"
    checkpoint_dir = checkpoint_root/args.title
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir/"checkpoint.pt"

    last_step = 0
    if os.path.isfile(checkpoint_path.absolute()):
        checkpoint = T.load(checkpoint_path.absolute())
        agent_state_dict = checkpoint["agent_state_dict"]
        agent_opt_state_dict = checkpoint["agent_opt_state_dict"]
        critic_state_dict = checkpoint["critic_state_dict"]
        critic_opt_state_dict = checkpoint["critic_opt_state_dict"]
        last_step = checkpoint["last_step"]
        agent.load_state_dict(agent_state_dict)
        agent_opt.load_state_dict(agent_opt_state_dict)
        critic.load_state_dict(critic_state_dict)
        critic_opt.load_state_dict(critic_opt_state_dict)

    env = gym.vector.SyncVectorEnv([lambda: SDS_ENV(args.dataset_size) for _ in range(args.num_envs)])

    memory = PPOMemory(args.mini_batch_size)
    return agent, critic, agent_opt, critic_opt, memory, env, last_step, checkpoint_path