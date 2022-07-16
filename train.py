from random import randint
from time import time
import gym
import torch as T
import numpy as np
from tqdm import tqdm

from env.sds_env import SDS_ENV
from config import get_args
from utils import select
from setup import setup
from validation import validate


def learn(args, agent, agent_opt, critic, critic_opt, memory):
    agent.train()
    for _ in range(args.n_learning_epochs):
        state_arr, mask_arr, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, batches = memory.generate_batches()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + args.gamma*values[k+1]*\
                        (1-int(dones_arr[k])) - values[k])
                discount *= args.gamma*args.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage, device=agent.device)
        values = T.tensor(values, device=agent.device)
        
        returns = advantage + values

        # standardize advantage
        advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)

        # advantage = (reward - baseline)
        # baseline = dari critic
        # baseline bisa dari model lain (arsitektur terpisah)
        # baseline bisa jadi, kita tau nilai optimal yang asli (supervised learning)
        # baseline bisa jadi juga running average dari reward yg didapatkan
        # baseline juga bisa jadi agent yang greedy -> self critic
        #     saat agent training dia milih random based on probs
        #     saat agent eval, harusnya ga boleh random, tapi pilih max probs (atau greedy)



        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float, device=agent.device)
            masks = T.tensor(mask_arr[batch], dtype=T.float, device=agent.device)
            old_probs = T.tensor(old_prob_arr[batch], device=agent.device)
            actions = T.tensor(action_arr[batch], device=agent.device)
            new_all_probs, _ = agent(states, masks)
            dist = T.distributions.Categorical(new_all_probs)
            critic_value = critic(states)
            critic_value = T.squeeze(critic_value)
            new_probs = dist.log_prob(actions).sum(dim=1)
            
            # compute loss
            prob_ratio = (new_probs - old_probs).exp()
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1-args.ppo_clip,
                    1+args.ppo_clip)*advantage[batch]
            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            agent_opt.zero_grad()
            actor_loss.backward()
            T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.grad_norm)
            agent_opt.step()

            returns = advantage[batch] + values[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            critic_opt.zero_grad()
            critic_loss.backward()
            T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.grad_norm)
            critic_opt.step()





if __name__ == "__main__":
    args = get_args()
    agent, critic, agent_opt, critic_opt, memory, env, last_step, best_validation_value, checkpoint_path, validation_env, writer = setup(args)
    validate(args, 
            agent.state_dict(),
            agent_opt.state_dict(), 
            critic.state_dict(), 
            critic_opt.state_dict(), 
            best_validation_value, 
            checkpoint_path,
            last_step,
            validation_env,
            writer,
            parallel=False)
    exit()
    # start training
    step = 0
    for epoch in range(last_epoch, args.max_epoch):
        dataset_idxs = [randint(0,num_datasets) for _ in range(args.num_envs+2)]

        # 8 train, 2 validasi
        training_idxs = dataset_idxs[:args.num_envs]
        validation_idxs = dataset_idxs[args.num_envs:]
        # batching dataset
        # pisah dataset training validation
        # init training environment
        env_fns = [lambda: SDS_ENV(dataset_idx=train_idx) for train_idx in training_idxs]
        training_env = gym.vector.AsyncVectorEnv(env_fns, shared_memory=False)
    
        val_env_fns = [lambda: SDS_ENV(dataset_idx=val_idx) for val_idx in validation_idxs]
        validation_env = gym.vector.AsyncVectorEnv(val_env_fns, shared_memory=False)
    
        # mulai generate experience dari training environments
        mask = np.ones((args.num_envs, 128, 3))
        mask[:,:,2] = 0
        features = training_env.reset()        
        for it in range(args.max_training_steps):
            with T.no_grad():
                agent.eval()
                features_ = T.from_numpy(features).to(agent.device).float()
                mask_ = T.from_numpy(mask).to(agent.device).float()
                probs, entropy = agent(features_, mask_)
                actions, logprobs = select(probs)
                legitimate = T.gather(mask_, 2, actions.unsqueeze(2))
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
            validate()
        step+=1

        # validasi,, untuk liat growth/improvement, simpen terbaik based on validation, early stopping
        # improvement dari objective function? / rewards?
        # consumed joules, mean slowdown time
        # kita belum liat seberapa berbeda kedua nilai di atas antar dataset (?)
        #     sudah weighted dan normalized per dataset
        #     tapi tiap dataset itu sama di jumlah jobs, bukan sama di 
        #         1. time range -> max_consumed_joules -> digunakan untuk normalized
        #             akan beda bangetkah scale dari objecive dari dataset dengan 
        #                 time range kecil vs time range besar
        # kalau rewards mungkin lebih 

        # training_dataset = 1000
        # per epoch ambil 100
        #     10 kali
        #         90 trainig, 10 validasi

        
        # dari 1000
        # ambil 10, 8 traning 2 validasi

        
        # yang kita rencana lakukan seperti ini:
        #     1 data doang, dipake validasi terus menerus 
