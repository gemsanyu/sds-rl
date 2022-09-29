from batsim_py.events import JobEvent

from env.sds_env import SDS_ENV
from config import get_args

def run(args):
    env = SDS_ENV(dataset_name="scenario1.json", batsim_verbosity="quiet", is_test=True)
    env.reset()
    env.simulator.subscribe(JobEvent.SUBMITTED, env.scheduler.schedule_caller)
    env.simulator.subscribe(JobEvent.COMPLETED, env.scheduler.schedule_caller)

    # coba sebelum dischedule kita matikan node 2 dan 3 (mulai dari 0)
    env.simulator.switch_off([2,3])
    for step in range(11):
        env.host_monitor.update_info_all()
        print("----------------------------------")
        print("TIME:", env.simulator.current_time)
        print("QUEUE", env.simulator.queue)
        for host in env.hosts:
            print("HOST ID:",host.id)
            print(host.state, host.is_allocated)
            print("current switching:", env.host_monitor.host_info[host.id]["current_time_switching_on"], env.host_monitor.host_info[host.id]["current_time_switching_off"])    
        if not env.simulator.is_running:
            break
        env.simulator.proceed_time()
if __name__ == "__main__":
    args = get_args()
    run(args)
    
