from batsim_py import SimulatorHandler
from batsim_py.events import HostEvent, SimulatorEvent
from batsim_py.resources import Host
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor, JobMonitor

import numpy as np
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
        self.setup_callback()

    def setup_callback(self) -> None:
        t_next_call = self.simulator.current_time +5000
        self.simulator.set_callback(t_next_call, self.callback)

    def callback(self, current_time: float) -> None:

        # print2 feature2 dulu sementara
        print("CURRENT TIME", current_time)
        print("PLATFORM Features")
        print("switch on dan switch off time langsung dikirim ke batsim, ga ada di batsimpy")
        print("1. switch on: 1 second")
        print("2. switch off: 10 seconds")
        
        print("SIMULATOR FEATURES")
        print("1. jumlah jobs di queue:", len(self.simulator.queue))
        print("2. arrival rate:")
        submission_time = self.job_monitor.info["submission_time"]
        print("---overall arrival rate (not&normalized):",get_arrival_rate(submission_time, False), get_arrival_rate(submission_time, True))
        print("---last 100 jobs arrival rate (not&normalized):",get_arrival_rate(submission_time[-100:], False), get_arrival_rate(submission_time[-100:], True) )
        print("3. mean runtime jobs in queue:")
        if len(self.simulator.queue) > 0:
            exit()
        # print(self.host_monitor.info)
        # print(self.job_monitor.info)s
        # exit()
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

def get_arrival_rate(submission_times, normalized=True):
    if len(submission_times) == 0:
        return 0
    if len(submission_times) == 1:
        return submission_times[0]
    submission_times = np.asarray(submission_times)
    submission_times -= submission_times[0]
    max_time = submission_times[-1]
    submission_times_r = np.roll(submission_times, 1)  
    submission_times -= submission_times_r
    arrival_rate = np.mean(submission_times[1:])
    if normalized:
        arrival_rate /= max_time
    return arrival_rate