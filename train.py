import csv
import sys
import pathlib

import torch as T

from env.sds_env import SDS_ENV
from config import define_args_parser




def select(probs, is_training=True):
    '''
    ### Select next to be executed.
    -----
    Parameter:
        probs: probabilities of each operation

    Return: index of operations, log of probabilities
    '''
    if is_training:
        dist = T.distributions.Categorical(probs)
        op = dist.sample()
        logprob = dist.log_prob(op)
    else:
        prob, op = T.max(probs, dim=1)
        logprob = T.log(prob)
    return op, logprob

if __name__ == "__main__":
    
    parser = define_args_parser()
    args = parser.parse_args(sys.argv[1:])
    # to_policy = lambda s: TimeoutPolicy(args.timeout, s)
    env = SDS_ENV(workload_filename=args.workload_path, platform_path=args.platform_path)
    features, mask = env.reset()
    mask = T.from_numpy(mask)
    probs = T.rand(size=mask.shape)*mask # dari 0->1 jika feasible, 0 jika enggak, ini dummy
    actions, logprobs = select(probs)
    state = env.step((actions, 2))
    
