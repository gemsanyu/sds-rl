from batsim_py import SimulatorHandler
from batsim_py.events import HostEvent, SimulatorEvent
from batsim_py.resources import Host

class TimeoutPolicy:
    def __init__(self, t_timeout: float, simulator: SimulatorHandler) -> None:
        self.simulator = simulator
        self.t_timeout = t_timeout
        self.hosts_idle = {}
        # Subscribe to some events.
        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self.on_simulation_begins)
        self.simulator.subscribe(HostEvent.STATE_CHANGED, self.on_host_state_changed)

    def on_simulation_begins(self, s: SimulatorHandler) -> None:
        for host in s.platform.hosts:
            if host.is_idle:
                self.hosts_idle[host.id] = s.current_time
                self.setup_callback()

    def on_host_state_changed(self, h: Host) -> None:
        if h.is_idle and not h.id in self.hosts_idle:
            self.hosts_idle[h.id] = self.simulator.current_time
            self.setup_callback()
        elif not h.is_idle and h.id in self.hosts_idle:
            del self.hosts_idle[h.id]

    def setup_callback(self) -> None:
        t_next_call = self.simulator.current_time + self.t_timeout
        self.simulator.set_callback(t_next_call, self.callback)

    def callback(self, current_time: float) -> None:
        for host_id, t_idle_start in list(self.hosts_idle.items()):
            if  current_time - t_idle_start >= self.t_timeout:
                self.simulator.switch_off([host_id])

from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor

def run_simulation(shutdown_policy):
    simulator = SimulatorHandler()
    policy = shutdown_policy(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform="platform.xml",
                    workload="workloads.json",
                    verbosity="information")

    # 3) Schedule all jobs
    while simulator.is_running:
        # First Fit policy
        for job in simulator.queue:
            available = simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                allocation = [h.id for h in available[:job.res]]
                simulator.allocate(job.id, allocation)

        # proceed directly to the next event because the shutdown_policy is event-based.
        simulator.proceed_time()

    simulator.close()

    # 4) Return/Dump statistics
    return sim_mon, host_mon, e_mon

sim_none, host_none, e_none = run_simulation(lambda s: None) # Without shutdown
sim_t1, host_t1, e_t1 = run_simulation(lambda s: TimeoutPolicy(1800, s)) # Timeout (1)
sim_t5, host_t5, e_t5 = run_simulation(lambda s: TimeoutPolicy(3600, s)) # Timeout (5)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results
none, t_1, t_5 = sim_none.to_dataframe(), sim_t1.to_dataframe(), sim_t5.to_dataframe()
none['name'], t_1['name'], t_5['name'] = "None", "Timeout (1)", "Timeout (5)"
benchmark = pd.concat([none, t_1, t_5], ignore_index=True)
benchmark['mean_slowdown'] = np.log(benchmark['mean_slowdown'])
benchmark['consumed_joules'] = np.log(benchmark['consumed_joules'])
# Slowdown
plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
plt.plot('name', 'mean_slowdown', data=benchmark)
plt.grid(axis='y')
plt.ylabel("Averaged Slowdown (s)")

# Energy consumed
plt.subplot(1, 2, 2)
plt.plot('name', 'consumed_joules', data=benchmark)
plt.grid(axis='y')
plt.ylabel("Energy Consumed (J)")

# Show
plt.show()

F1 = 
F2 =  