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

def run_simulation(shutdown_policy, workload_filename):
    simulator = SimulatorHandler()
    policy = shutdown_policy(simulator)

    # 1) Instantiate monitors to collect simulation statistics
    sim_mon = SimulationMonitor(simulator)
    host_mon = HostStateSwitchMonitor(simulator)
    e_mon = ConsumedEnergyMonitor(simulator)

    # 2) Start simulation
    simulator.start(platform="platform.xml",
                    workload=workload_filename,
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
    return sim_mon, host_mon, e_mon, simulator


sim_s1, host_s1, e_s1, simulator_s1 = run_simulation(lambda s: TimeoutPolicy(1800, s), "workloads-gaia.json") # Simulation 1: Timeout (30 minute) Dataset (Gaia)
sim_s2, host_s2, e_s2, simulator_s2 = run_simulation(lambda s: TimeoutPolicy(3600, s), "workloads-gaia.json") # Simulation 2: Timeout (60 minute) Dataset (Gaia)
sim_s3, host_s3, e_s3, simulator_s3 = run_simulation(lambda s: TimeoutPolicy(1800, s), "workloads-nasa.json") # Simulation 3: Timeout (30 minute) Dataset (NASA)
sim_s4, host_s4, e_s4, simulator_s4 = run_simulation(lambda s: TimeoutPolicy(3600, s), "workloads-nasa.json") # Simulation 4: Timeout (60 minute) Dataset (NASA)
sim_s5, host_s5, e_s5, simulator_s5 = run_simulation(lambda s: TimeoutPolicy(1800, s), "workloads-lpc.json") # Simulation 5: Timeout (30 minute) Dataset (LPC)
sim_s6, host_s6, e_s6, simulator_s6 = run_simulation(lambda s: TimeoutPolicy(3600, s), "workloads-lpc.json") # Simulation 6: Timeout (60 minute) Dataset (LPC)
sim_s7, host_s7, e_s7, simulator_s7 = run_simulation(lambda s: TimeoutPolicy(1800, s), "workloads-llnl.json") # Simulation 7: Timeout (30 minute) Dataset (LLNL)
sim_s8, host_s8, e_s8, simulator_s8 = run_simulation(lambda s: TimeoutPolicy(3600, s), "workloads-llnl.json") # Simulation 8: Timeout (60 minute) Dataset (LLNL)

simulation_monitors = [sim_s1, sim_s2, sim_s3, sim_s4, sim_s5, sim_s6, sim_s7, sim_s8]
host_monitors = [host_s1, host_s2, host_s3, host_s4, host_s5, host_s6, host_s7, host_s8]
energy_monitors = [e_s1, e_s2, e_s3, e_s4, e_s5, e_s6, e_s7, e_s8]
simulators = [simulator_s1, simulator_s2, simulator_s3, simulator_s4, simulator_s5, simulator_s6, simulator_s7, simulator_s8]

def F(mean_slowdown , consumed_joules, max_consumed_joules, alpha, beta, is_normalized):
    if is_normalized:
        consumed_joules = consumed_joules/max_consumed_joules
    return alpha * mean_slowdown + beta * consumed_joules

def compute_score(sim_mon, sim_handler, alpha=0.5, beta=0.5, is_normalized=True):
  platform = sim_handler.platform
  hosts = platform.hosts
  total_max_watt_per_min = 0
  for host in hosts:
    max_watt_per_min = 0
    for pstate in host.pstates:
      max_watt_per_min = max(max_watt_per_min, pstate.watt_full)
    total_max_watt_per_min += max_watt_per_min

  total_time = sim_handler.current_time
  max_consumed_joules = total_time*total_max_watt_per_min
  consumed_joules = sim_mon.info["consumed_joules"]
  mean_slowdown = sim_mon.info["mean_slowdown"]
  score = F(mean_slowdown, consumed_joules, max_consumed_joules, alpha, beta, is_normalized)
  return score

import csv

header = ['dataset', 'timeout', 'f(1,0)=slowdown', 'f(0,1)=energy', 'f(0.5,0.5)=balance']

dataset = ["Gaia", "NASA", "LPC", "LLNL"]
timeout = ["30mins", "60mins"]

data = []
for i in range(8):
    row = []
    row.append(dataset[i//2])
    row.append(timeout[i%2])
    row.append(compute_score(simulation_monitors[i], simulators[i], 1, 0))
    row.append(compute_score(simulation_monitors[i], simulators[i], 0, 1))
    row.append(compute_score(simulation_monitors[i], simulators[i], 0.5, 0.5))
    data.append(row)

with open('output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)