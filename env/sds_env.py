import csv
import sys
import pathlib
from this import s
from time import time
from typing import Tuple

import gym
from gym import Env, spaces
import numpy as np

from batsim_py import SimulatorHandler
from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_py.events import JobEvent

from env.easy_backfilling import EASYScheduler
from env.utils import *

SWITCH_OFF = 1
SWITCH_ON = 2

class SDS_ENV(Env):
    def __init__(self, workload_filename, platform_path, batsim_verbosity="quiet", alpha=0.5, beta=0.5) -> None:
        super(SDS_ENV, self).__init__()
        self.workload_filename = workload_filename
        self.batsim_verbosity = batsim_verbosity
        self.platform_path = platform_path
        self.alpha = alpha
        self.beta = beta
        self.simulator = SimulatorHandler()
        self.scheduler = EASYScheduler(self.simulator)
        self.num_sim_features = 5
        self.num_node_features = 6
        # 1) Instantiate monitors to collect simulation statistics
        self.simulation_monitor = SimulationMonitor(self.simulator)
        self.host_monitor = HostMonitor(self.simulator)
        self.energy_monitor = ConsumedEnergyMonitor(self.simulator)
        self.job_monitor = JobMonitor(self.simulator)
        # job infos dict, manually compile 
        self.job_infos = {}
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)
        self.previous_wasted_energy = None    

        self.simulator.start(platform=platform_path, workload=workload_filename, verbosity=self.batsim_verbosity)
        self.n_host = len(list(self.simulator.platform.hosts))
        self.hosts = list(self.simulator.platform.hosts)
        self.observation_shape = (self.n_host, self.num_sim_features+self.num_node_features)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)
        # Define an action space for each host, ranging from 0 to 3
        # 0. No Action
        # 1. Turn OFF
        # 2. Turn ON
        self.action_space = spaces.MultiDiscrete([3 for _ in range(self.n_host)])


    def add_to_job_infos(self, job):
        self.job_infos[job.id] = job  
    
    def close(self):
        self.simulator.close()

    def reset(self):
        self.simulator.close()
        # 1) Instantiate monitors to collect simulation statistics
        self.simulation_monitor = SimulationMonitor(self.simulator)
        self.host_monitor = HostMonitor(self.simulator)
        self.energy_monitor = ConsumedEnergyMonitor(self.simulator)
        self.job_monitor = JobMonitor(self.simulator)

        # job infos dict, manually compile 
        self.job_infos = {}
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)
        self.previous_wasted_energy = None

        self.simulator.start(platform=self.platform_path, workload=self.workload_filename, verbosity=self.batsim_verbosity)
        self.host_monitor.update_info_all()
        
        features = self.get_features(self.simulator.current_time)
        mask = get_feasible_mask(list(self.simulator.platform.hosts))
        return features, mask
# return features, mask

    # return next features, reward, and done flag, and empty [](this is info)
    def step(self, actions_and_dt):
        actions, dt = actions_and_dt
        self.apply(actions)
        # proceed time, and schedule
        # get next features
        if self.simulator.is_running:
            self.scheduler.schedule()
            self.simulator.proceed_time(time=dt)  # proceed directly to the next event.
            done = False
        else:
            self.simulator.close()
            self.reset()
            done=True
        current_time = self.simulator.current_time
        features = self.get_features(current_time)
        rewards = self.get_rewards(current_time, dt)
        mask = get_feasible_mask(list(self.simulator.platform.hosts))
        return features, rewards, done, mask

    def apply(self, actions):
        hosts_id_to_switch_off = np.nonzero((actions == SWITCH_OFF)).squeeze(1).tolist()
        hosts_id_to_switch_on = np.nonzero((actions == SWITCH_ON)).squeeze(1).tolist()
        if len(hosts_id_to_switch_off) > 0:
            self.simulator.switch_off(hosts_id_to_switch_off)
        if len(hosts_id_to_switch_on) > 0:
            self.simulator.switch_off(hosts_id_to_switch_on)

    def get_rewards(self, current_time, dt):
        wasted_energy = self.host_monitor.info["energy_waste"] 
        wasted_energy_ = wasted_energy
        if self.previous_wasted_energy is not None:
            wasted_energy -= self.previous_wasted_energy
        wasted_energy /= (190.*self.n_host*dt)
        self.previous_wasted_energy = wasted_energy_

        waiting_time_since_last_dt = 0.
        n_job_waitting = 0
        for job in self.job_infos.values():
            print(job)
            if job.start_time is not None and (job.start_time < current_time-dt):
                continue
            n_job_waitting += 1
            last_dt = current_time-dt
            waittime = 0
            if job.start_time is not None:
                waittime = job.start_time - last_dt
            else:
                waittime = dt # atau current_time-last_dt
            waiting_time_since_last_dt += (waittime/dt)
        waiting_time_since_last_dt /= max(n_job_waitting,1)
        reward = -self.alpha*wasted_energy -self.beta*waiting_time_since_last_dt
        return reward

    def get_features(self, current_time)-> Tuple[np.array, np.array]:
        
        simulator_features = np.zeros((self.num_sim_features,), dtype=np.float32)
        simulator_features[0] = len(self.simulator.queue)
        submission_time = self.job_monitor.info["submission_time"]
        simulator_features[1] = get_arrival_rate(submission_time, is_normalized=True)
        hosts = self.simulator.platform.hosts
        queue = self.simulator.queue
        simulator_features[2] = get_mean_waittime_queue(queue, current_time, is_normalized=True)
        simulator_features[3] = get_wasted_energy(self.energy_monitor, self.host_monitor, is_normalized=True)
        simulator_features[4] = get_mean_walltime_in_queue(queue,is_normalized=True)
        simulator_features = simulator_features[np.newaxis, ...]

        hosts = list(self.simulator.platform.hosts)
        node_features = np.zeros((len(hosts), self.num_node_features), dtype=np.float32)
        host_on_off = get_host_on_off(self.simulator.platform)
        node_features[:, 0] = host_on_off
        host_active_idle = get_host_active_idle(self.simulator.platform)
        node_features[:, 1] = host_active_idle
        current_idle_time = get_current_idle_time(self.host_monitor)
        node_features[:, 2] = current_idle_time
        remaining_runtime_percent = get_remaining_runtime_percent(list(self.simulator.platform.hosts), self.job_infos, self.simulator.current_time)
        node_features[:, 3] = remaining_runtime_percent
        normalized_wasted_energy = get_host_wasted_energy(self.host_monitor, True)
        node_features[:, 4] = normalized_wasted_energy
        normalized_switching_time = get_switching_time(self.host_monitor, True, self.simulator.current_time) 
        node_features[:, 5] = normalized_switching_time

        simulator_features = np.broadcast_to(simulator_features, (node_features.shape[0], simulator_features.shape[1]))
        features = np.concatenate((simulator_features, node_features), axis=1)
        return features
