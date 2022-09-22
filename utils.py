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
        prob, op = T.max(probs, dim=2)
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


def learn(args, agent, agent_opt, critic, critic_opt, done, saved_experiences):
    saved_logprobs, saved_states, saved_rewards, next_state = saved_experiences
    #prepare returns
    if done:
        R = 0
    else:
        next_state = T.from_numpy(next_state).to(agent.device).float()
        R = critic(next_state).detach().item()
    returns = [0 for _ in range(len(saved_logprobs))]
    critic_vals = [0. for _ in range(len(saved_logprobs))]
    for i in range(len(returns)):
        R = saved_rewards[-i] + args.gamma*R
        returns[-i]=R
        critic_vals[-i] = critic(saved_states[-i]).squeeze(0)
    saved_logprobs = T.stack(saved_logprobs)
    critic_vals = T.stack(critic_vals)
    returns = T.tensor(returns, dtype=T.float32)
    advantage = (returns - critic_vals).detach()
    #update actor
    agent_loss = -(saved_logprobs*advantage).sum()
    agent_opt.zero_grad(set_to_none=True)
    agent_loss.backward()
    T.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=args.grad_norm)
    agent_opt.step()

    #update critic
    critic_loss = (returns-critic_vals)**2
    critic_loss = critic_loss.mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    T.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=args.grad_norm)
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