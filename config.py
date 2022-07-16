import sys

import argparse

def define_args_parser():
    parser = argparse.ArgumentParser(
        description="MDVRPTW with RL"
    )

    # Agent Model
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        help='device where to run agent (cpu or cuda)')
    parser.add_argument('--n-heads',
                        type=int,
                        default=8,
                        help='num head in multi-head attention')
    parser.add_argument('--n-gae-layers',
                        type=int,
                        default=16,
                        help='num multi-head attention layers')
    parser.add_argument('--gae-ff-hidden',
                        type=int,
                        default=512,
                        help='num neurons in linear embedder in MHA layers')
    parser.add_argument('--embed-dim',
                        type=int,
                        default=64,
                        help='size of embeding dims/ num neurons used in every substructure of model')
    parser.add_argument('--tanh-clip',
                        type=float,
                        default=10,
                        help='tanh clipping')



    # Misc
    parser.add_argument('--max-steps',
                        type=int,
                        default=1000000,
                        help='Maximum steps of environment.')
    parser.add_argument('--mini-batch-size',
                        type=int,
                        default=16,
                        help='mini batch size for training the agent.')
    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help='Learning rate for optimizer')
    parser.add_argument('--dataset-size',
                        type=int,
                        default=1000,
                        help='num of available datasets.')
    parser.add_argument('--num-envs',
                        type=int,
                        default=4,
                        help='num of simultaneuos environments running.')
    parser.add_argument('--training-steps',
                        type=int,
                        default=64,
                        help='num of steps before one step training')
    parser.add_argument('--n-learning-epochs',
                        type=int,
                        default=10,
                        help='num of learning epochs')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='discounted return gamma')
    parser.add_argument('--gae-lambda',
                        type=float,
                        default=0.95,
                        help='gae lambda')
    parser.add_argument('--grad-norm',
                        type=float,
                        default=2,
                        help='gradient clipping norm')
    parser.add_argument('--ppo-clip',
                        type=float,
                        default=0.2,
                        help='ppo clip or epsilon')
    
    


    # Simulation
    parser.add_argument('--title',
                        type=str,
                        default="self-attention-v1",
                        help="experiment title savefiles' name")
    parser.add_argument('--validation-workload-name',
                        type=str,
                        default="workloads-nasa-cleaned-128host.json",
                        help="workload path (job requests' details")
    parser.add_argument('--platform-path',
                        type=str,
                        default="platform.xml",
                        help="platform or nodes' details")
    parser.add_argument('--timeout',
                        type=int,
                        default=3600,
                        help='timeout for timeout shutdown policy')
    parser.add_argument('--is-baseline',
                        type=bool,
                        default=False,
                        help='if baseline dont use any policy')


    return parser

def get_args():
    parser = define_args_parser()
    args = parser.parse_args(sys.argv[1:])
    return args