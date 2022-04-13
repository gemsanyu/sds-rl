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
    return sim_mon, host_mon, e_mon, simulator

sim_none, host_none, e_none, simulator_none = run_simulation(lambda s: None) # Without shutdown
sim_t10s, host_t10s, e_t10s, simulator_t10s = run_simulation(lambda s: TimeoutPolicy(10, s)) # Timeout (1)
sim_t1, host_t1, e_t1, simulator_t1 = run_simulation(lambda s: TimeoutPolicy(1800, s)) # Timeout (1)
sim_t5, host_t5, e_t5, simulator_t5 = run_simulation(lambda s: TimeoutPolicy(3600, s)) # Timeout (5)


# exit()

def F(mean_slowdown , consumed_joules, max_consumed_joules, alpha = 0.5, beta = 0.5, is_normalized = True):
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

  total_time = simulator_none.current_time
  max_consumed_joules = total_time*total_max_watt_per_min
  consumed_joules = sim_mon.info["consumed_joules"]
  mean_slowdown = sim_mon.info["mean_slowdown"]
  score = F(mean_slowdown, consumed_joules, max_consumed_joules)
  return score

print(compute_score(sim_none, simulator_none))
print(compute_score(sim_t1, simulator_t1))
print(compute_score(sim_t5, simulator_t5))
print(compute_score(sim_t10s, simulator_t10s))