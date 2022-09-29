from typing import NamedTuple

from batsim_py import SimulatorHandler
from batsim_py.monitors import HostMonitor
from batsim_py.resources import Host, HostState
import numpy as np

class TimeToReady(NamedTuple):
    host: Host
    state: HostState
    release_time: float
    time_to_ready: float

def get_host_time_to_ready(simulator: SimulatorHandler, host_monitor: HostMonitor):
    agenda = simulator.agenda
    host_info = host_monitor.host_info
    need_to_ready_state = [HostState.SLEEPING, HostState.SWITCHING_OFF, HostState.SWITCHING_ON]
    for ag in agenda:
        host = ag.host
        release_time = ag.release_time
        time_to_ready = release_time
        if host.state in need_to_ready_state:
            time_to_ready = host_info[host.id]["time_to_ready"]
        yield TimeToReady(host, host.state, release_time, time_to_ready)
