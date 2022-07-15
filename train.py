import csv
import sys
import pathlib
import gym

import torch as T
import numpy as np

from env.sds_env import SDS_ENV
from config import get_args

from utils import select
from setup import setup


def learn(args, agent, agent_opt, critic, critic_opt, memory):
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
        advantage = T.tensor(advantage)
        values = T.tensor(values)
        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float)
            masks = T.tensor(mask_arr[batch], dtype=T.float)
            old_probs = T.tensor(old_prob_arr[batch])
            actions = T.tensor(action_arr[batch])

            dist, _ = agent(states, masks)
            critic_value = critic(states)

            critic_value = T.squeeze(critic_value)
            new_probs = T.gather(input=dist, dim=2, index=actions.unsqueeze(2)).squeeze(2)
            new_action_valid = T.gather(input=masks, dim=2, index=actions.unsqueeze(2)).squeeze(2)
            print(T.count_nonzero(new_action_valid))
            # print(T.count_nonzero(new_probs))
            new_probs = new_probs.log().sum(dim=1)
            # print(new_probs)
            # print(old_probs)
            # new_probs = dist.log_prob(actions), dist[actions]
            # ubah ke gather

            prob_ratio = (new_probs - old_probs).exp()
            #prob_ratio = (new_probs - old_probs).exp()
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
    agent, critic, agent_opt, critic_opt, memory, env, last_step, checkpoint_path = setup(args)
    # start training
    mask = np.ones((args.num_envs, 128, 3))
    mask[:,:,2] = 0
    features = env.reset()
    for step in range(last_step, args.max_steps):
        print(step)
        with T.no_grad():
            features_ = T.from_numpy(features).float()
            mask_ = T.from_numpy(mask).float()
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

        # critic_score = sum_rewards.mean()
        # advantage = (sum_rewards-critic_score) # adv positive if rewards > mean_rewards, else negative
        # loss = (sum_rewards*sum_logprobs).mean()
        # loss = -loss

        # agent_opt.zero_grad()
        # loss.backward()
        # agent_opt.step()

