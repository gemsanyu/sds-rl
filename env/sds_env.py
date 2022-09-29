import csv
import queue
from random import randint
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
from batsim_py.monitors import HostStateSwitchMonitor, SimulationMonitor, HostMonitor, ConsumedEnergyMonitor, JobMonitor
from batsim_py.events import JobEvent

# from env.easy_backfilling import EASYScheduler
from env.sa_backfilling import StateAwareBFScheduler
from env.easy_backfilling import EASYScheduler
from env.utils import *

# NO_OP = 0
# SWITCH_OFF = 1
# SWITCH_ON = 2
SWITCH_OFF = 0
SWITCH_ON = 1

class SDS_ENV(Env):
    def __init__(self, 
                 dataset_name, 
                 batsim_verbosity="quiet", 
                 alpha=0.5, 
                 beta=0.5, 
                 num_host=4, 
                 is_test=False) -> None:
        super(SDS_ENV, self).__init__()
        self.batsim_verbosity = batsim_verbosity
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.platform_path = pathlib.Path(".")/"platform"/("platform-"+str(num_host)+".xml")
        self.dataset_dir = pathlib.Path(".")/"dataset"
        if self.is_test:
            self.dataset_dir= self.dataset_dir/"test"
        else:
            self.dataset_dir= self.dataset_dir/"training"
        self.dataset_filepath = self.dataset_dir/self.dataset_name
        self.alpha = alpha
        self.beta = beta
        self.simulator = SimulatorHandler()
        # self.scheduler = SABFScheduler(self.simulator, self.host_monitor)
        self.num_sim_features = 5
        self.num_node_features = 6
        self.n_host = num_host
        self.observation_shape = (self.n_host, self.num_sim_features+self.num_node_features)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float32)
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
        self.last_host_info = None
        # 1) Instantiate monitors to collect simulation statistics
        self.simulation_monitor = SimulationMonitor(self.simulator)
        self.host_monitor = HostMonitor(self.simulator)
        self.host_state_switch_monitor = HostStateSwitchMonitor(self.simulator)
        self.energy_monitor = ConsumedEnergyMonitor(self.simulator)
        self.job_monitor = JobMonitor(self.simulator)

        # job infos dict, manually compile 
        self.job_infos = {}
        self.simulator.subscribe(JobEvent.SUBMITTED, self.add_to_job_infos)
        self.previous_wasted_energy = None

        self.scheduler = StateAwareBFScheduler(self.simulator, self.host_monitor)
        # self.scheduler = EASYScheduler(self.simulator)
        self.simulator.start(platform=self.platform_path, workload=self.dataset_filepath.absolute(), verbosity=self.batsim_verbosity)
        self.hosts = list(self.simulator.platform.hosts)
        self.host_monitor.update_info_all()