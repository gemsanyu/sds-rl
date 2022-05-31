from batsim_py import SimulatorHandler
from batsim_py.events import HostEvent, SimulatorEvent
from batsim_py.resources import Host

from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor

"""
kita tiru shutdown policy
si simulator akan subcribe ke sebuah callback function setiap dtime
callback function ini isinya:
1. extract features dari monitor
2. RL decides to switch on/off
3. RL executes on or off
4. 
"""

class RLPolicy:
    def __init__(self, 
                agent, 
                dtime, 
                simulation_monitor:SimulationMonitor, 
                energy_monitor:ConsumedEnergyMonitor,
                host_monitor:HostStateSwitchMonitor,
                job_monitor: JobMonitor,
                simulator: SimulatorHandler) -> None:
        self.simulator = simulator
        self.dtime = dtime
        self.hosts_idle = {}
        self.agent = agent
        # Subscribe to some events.
        self.simulator.subscribe(SimulatorEvent.SIMULATION_BEGINS, self.on_simulation_begins)

        # monitors for extracting features
        self.simulation_monitor = simulation_monitor
        self.energy_monitor = energy_monitor
        self.host_monitor = host_monitor
        self.job_monitor = job_monitor

    def on_simulation_begins(self, s: SimulatorHandler) -> None:
        for host in s.platform.hosts:
            if host.is_idle:
                self.hosts_idle[host.id] = s.current_time
                self.setup_callback()

    # def on_host_state_changed(self, h: Host) -> None:
    #     if h.is_idle and not h.id in self.hosts_idle:
    #         self.hosts_idle[h.id] = self.simulator.current_time
    #         self.setup_callback()
    #     elif not h.is_idle and h.id in self.hosts_idle:
    #         del self.hosts_idle[h.id]

    def setup_callback(self) -> None:
        t_next_call = self.simulator.current_time + self.dtime
        self.simulator.set_callback(t_next_call, self.callback)

    def callback(self, current_time: float) -> None:

        # print2 feature2 dulu sementara
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", current_time)
        print(self.job_monitor.info)
        print()
        print()
        print()
        # compute mean arrival rate
        # get last 50 elements if exists

        # compute rewards dari action sebelumnya
        # prepare features dan masking (node yg sedang busy ga boleh dimatikan)
        # shutdown nodes
        # 

        self.setup_callback()
        # for host_id, t_idle_start in list(self.hosts_idle.items()):
        #     if  current_time - t_idle_start >= self.t_timeout:
        #         self.simulator.switch_off([host_id])