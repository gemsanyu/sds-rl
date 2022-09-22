import torch as T
import numpy as np
import pathlib
from batsim_py import simulator

from env.sds_env import SDS_ENV
from timeout_policy import TimeoutPolicy
from config import get_args
from utils import select, compute_objective, run_partly_with_baseline, ResultInfo
from setup import setup

from batsim_py.monitors import HostStateSwitchMonitor, SimulationMonitor, HostMonitor, ConsumedEnergyMonitor, JobMonitor


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
    _, _, _, _, _, _, _, checkpoint_path, writer = setup(args)
    # start training
    # 1 epoch = 1 full training data,, not the epoch commonly understood (?)
    # init training environment
    args.num_envs = 1
    # mulai generate experience dari training environments
    env = SDS_ENV(dataset_name=args.dataset_name, batsim_verbosity="information", is_test=True, alpha=args.alpha, beta=args.beta)
    env.reset()

    run_partly_with_baseline(env)
    result_prerun = ResultInfo(
        env.simulation_monitor.info["total_slowdown"],
        env.simulation_monitor.info["nb_jobs_finished"],
        env.simulator.current_time,
        env.simulation_monitor.info["consumed_joules"],
        env.simulation_monitor.info["time_idle"],
        env.simulation_monitor.info["time_computing"],
        env.simulation_monitor.info["time_switching_off"],
        env.simulation_monitor.info["time_switching_on"],
        env.simulation_monitor.info["time_sleeping"],
        env.simulation_monitor.info["energy_waste"]
    )

    timeout_policy = TimeoutPolicy(args.timeout, env.simulator)
    while env.simulator.is_running:
        env.scheduler.schedule()
        env.simulator.proceed_time()

    result_current = ResultInfo(
        env.simulation_monitor.info["total_slowdown"],
        env.simulation_monitor.info["nb_jobs_finished"],
        env.simulator.current_time,
        env.simulation_monitor.info["consumed_joules"],
        env.simulation_monitor.info["time_idle"],
        env.simulation_monitor.info["time_computing"],
        env.simulation_monitor.info["time_switching_off"],
        env.simulation_monitor.info["time_switching_on"],
        env.simulation_monitor.info["time_sleeping"],
        env.simulation_monitor.info["energy_waste"]
    )

    alpha=0.5
    beta=0.5
    consumed_joules, mean_slowdown, score, time_idle, time_computing, time_switching_off, time_switching_on, time_sleeping, energy_waste = compute_objective(env.simulator, result_current, result_prerun, alpha, beta)
    print("OBJECTIVE:", score)
    print("CONSUMED JOULES:", consumed_joules)
    print("MEAN SLOWDOWN:", mean_slowdown)
    print("TIME IDLE:", time_idle)
    print("TIME COMPUTING:", time_computing)
    print("TIME SWITCHING OFF:", time_switching_off)
    print("TIME SWITCHING ON:", time_switching_on)
    print("TIME SLEEPING:", time_sleeping)
    print("ENERGY WASTE:", env.simulation_monitor.info["energy_waste"])
