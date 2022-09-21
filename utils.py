from typing import NamedTuple, Optional
import json

import numpy as np
import torch as T

from env.sds_env import SDS_ENV
from batsim_py.monitors import SimulationMonitor
from batsim_py.simulator import SimulatorHandler

class ResultInfo(NamedTuple):
    total_slowdown: float
    num_jobs_finished: int
    current_time: float
    consumed_joules: float
    time_idle : float
    time_computing : float
    time_switching_off : float
    time_switching_on : float
    time_sleeping : float
    energy_waste : float

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
    logprob = logprob.sum(dim=1)
    return op, logprob

def F(mean_slowdown , consumed_joules, max_consumed_joules, alpha, beta, is_normalized):
    if is_normalized:
        consumed_joules = consumed_joules/max_consumed_joules
    return alpha * mean_slowdown + beta * consumed_joules

def compute_objective(sim_handler:SimulatorHandler, result:ResultInfo, result_prerun: Optional[ResultInfo]=None, alpha=0.5, beta=0.5, is_normalized=True):
    platform = sim_handler.platform
    hosts = platform.hosts
    total_max_watt_per_min = 0
    for host in hosts:
        max_watt_per_min = 0
        for pstate in host.pstates:
            max_watt_per_min = max(max_watt_per_min, pstate.watt_full)
        total_max_watt_per_min += max_watt_per_min

    total_time = result.current_time
    if result_prerun is not None:
        total_time -= result_prerun.current_time
    consumed_joules = result.consumed_joules
    total_slowdown = result.total_slowdown
    num_jobs_finished = result.num_jobs_finished
    time_idle = result.time_idle
    time_computing = result.time_computing
    time_switching_off = result.time_switching_off
    time_switching_on = result.time_switching_on
    time_sleeping = result.time_sleeping
    energy_waste = result.energy_waste
    
    if result_prerun is not None:
        consumed_joules -= result_prerun.consumed_joules
        total_slowdown -= result_prerun.total_slowdown
        num_jobs_finished -= result_prerun.num_jobs_finished
        time_idle -= result_prerun.time_idle
        time_computing -= result_prerun.time_computing
        time_switching_off -= result_prerun.time_switching_off
        time_switching_on -= result_prerun.time_switching_on
        time_sleeping -= result_prerun.time_sleeping
        energy_waste -= result_prerun.energy_waste
    max_consumed_joules = total_time*total_max_watt_per_min
    mean_slowdown = total_slowdown/num_jobs_finished

    score = F(mean_slowdown, consumed_joules, max_consumed_joules, alpha, beta, is_normalized)
    return consumed_joules, mean_slowdown, score, time_idle, time_computing, time_switching_off, time_switching_on, time_sleeping, energy_waste


def learn(args, agent, agent_opt, critic, critic_opt, memory):
    agent.train()
    for _ in range(args.n_learning_epochs):
        state_arr, mask_arr, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, batches = memory.generate_batches()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        # diubah agar tidak nested loop
        # perhitungan GAE generalized advantage estimation
        advantage = [0 for _ in range(len(reward_arr))]
        for t in reversed(range(len(reward_arr)-1)):
            delta = reward_arr[t] + (args.gamma * values[t+1] * (1-int(dones_arr[t]))) - values[t]
            advantage[t] = delta + (args.gamma * args.gae_lambda * advantage[t + 1] * (1-int(dones_arr[t])))

        advantage = T.tensor(advantage, device=agent.device)
        values = T.tensor(values, device=agent.device)
        
        returns = advantage + values

        # standardize advantage
        advantage = (advantage - advantage.mean())/(advantage.std()+1e-8)
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
            actor_loss = T.min(weighted_probs, weighted_clipped_probs).mean()
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

def run_partly_with_baseline(env: SDS_ENV, completed_percentage_target=0.8):
    num_jobs = 0
    with open(env.dataset_filepath) as json_file:
        data = json.load(json_file)
        num_jobs = len(data["jobs"])

    while env.simulator.is_running:
        env.scheduler.schedule()
        env.simulator.proceed_time()
        num_completed_jobs = len(env.job_monitor.info["job_id"])
        completed_percentage = (num_completed_jobs/num_jobs)
        if completed_percentage > completed_percentage_target:
            return