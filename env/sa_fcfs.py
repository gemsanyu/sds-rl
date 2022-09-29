from operator import attrgetter
from typing import List
from batsim_py.monitors import HostMonitor, TIME_TO_SWITCH_OFF, TIME_TO_SWITCH_ON
from batsim_py.resources import Host, HostState
from batsim_py.simulator import SimulatorHandler

from env.utils import get_host_time_to_ready

class StateAwareFCFSScheduler:
    def __init__(self, simulator: SimulatorHandler,
                 host_monitor: HostMonitor) -> None:
        self.simulator = simulator
        self.host_monitor = host_monitor

    def __str__(self) -> str:
        return "FCFS"

    def schedule(self) -> None:
        """  First Come First Served policy """
        assert self.simulator.is_running
        self.host_monitor.update_info_all()
        for job in self.simulator.queue:
            # add time to ready to available list, 
            time_to_ready = get_host_time_to_ready(self.simulator, self.host_monitor)
            time_to_ready = list(time_to_ready)
            time_to_ready = sorted(time_to_ready, key=lambda x: (x.time_to_ready, x.state))
            available = [ttr.host for ttr in time_to_ready if not ttr.host.is_allocated]
            if job.res <= len(available):
                # Schedule if the job can start now.
                allocation = [h.id for h in available[:job.res]]
                print("ALLOC", job, allocation)
                self.simulator.allocate(job.id, allocation)
            else:
                host_info = self.host_monitor.host_info
                # sort time_to_ready
                time_to_ready = get_host_time_to_ready(self.simulator, self.host_monitor)
                time_to_ready = list(time_to_ready)
                time_to_ready = sorted(time_to_ready, key=lambda x: (x.time_to_ready, x.state))
                
                # compute reserved time for this immediate job or p_start_t
                p_start_t = time_to_ready[job.res-1].time_to_ready
                # find nodes with time to ready <= p_start_t (reserved_nodes)
                candidates_for_reserved = [ttr for ttr in time_to_ready if ttr.time_to_ready<=p_start_t]
                # get reserved nodes, with similar logic as BF prioritized job calculation
                reserved = candidates_for_reserved[-job.res:]
                
                reserved_hosts = [rs.host for rs in reserved]            
                # make sure, at p_start_t - walltime of this job, the reserved_nodes are up
                callback_lambda_list = []
                skip_states = [HostState.COMPUTING, HostState.IDLE, HostState.SWITCHING_ON]
                for host in reserved_hosts:
                    if host.state in skip_states:
                        continue
                    make_sure_host_on_func = lambda current_time, host_=host: self.simulator.make_sure_host_on(host_, current_time)
                    time_left_to_switch_off = host_info[host.id]["time_left_to_switch_off"]
                    at = max(p_start_t-TIME_TO_SWITCH_ON+time_left_to_switch_off, 1)
                    at += self.simulator.current_time
                    callback_lambda_list += [(at, make_sure_host_on_func)]
                
                for callback_lambda in callback_lambda_list:
                    at, func_to_call = callback_lambda
                    self.simulator.set_callback(at, func_to_call)
                break
