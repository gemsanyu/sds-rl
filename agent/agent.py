from batsim_py import SimulatorHandler
from batsim_py.monitors import SimulationMonitor, HostStateSwitchMonitor, ConsumedEnergyMonitor


class Agent(object):
    def __init__(self):
        super(Agent).__init__()
    
    def forward(self, 
                simulator: SimulatorHandler, 
                simulation_monitor: SimulationMonitor,
                energy_monitor: ConsumedEnergyMonitor)
        