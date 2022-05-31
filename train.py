from agent.agent import Agent

from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor

def run_simulation(workload_filename):
    agent = Agent()
    simulator = SimulatorHandler()

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
        for job in simulator.queue:
            available = simulator.platform.get_not_allocated_hosts()
            if job.res <= len(available):
                allocation = [h.id for h in available[:job.res]]
                simulator.allocate(job.id, allocation)
        

        simulator.proceed_time()

    simulator.close()

    # 4) Return/Dump statistics
    return sim_mon, host_mon, e_mon, simulator

if __name__ = "__main__":
    sim_mon, host_mon, e_mon, simulator = run_simulation("workloads-gaia.json")