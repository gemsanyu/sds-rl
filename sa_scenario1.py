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

def run(args):
    env = SDS_ENV(dataset_name="scenario1.json", batsim_verbosity="quiet", is_test=True)
    env.reset()
    # coba sebelum dischedule kita matikan node 2 dan 3 (mulai dari 0)
    env.simulator.switch_off([2,3])
    env.host_monitor.update_info_all()
    for step in range(11):
        print("----------------------------------")
        print("TIME:", env.simulator.current_time)
        print("QUEUE", env.simulator.queue)
        for host in env.hosts:
            print("HOST ID:",host.id)
            print(host.state, host.is_allocated)
            print("current switching:", env.host_monitor.host_info[host.id]["current_time_switching_on"], env.host_monitor.host_info[host.id]["current_time_switching_off"])    
        if not env.simulator.is_running:
            break
        env.simulator.proceed_time()
if __name__ == "__main__":
    args = get_args()
    run(args)
    
