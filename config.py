import argparse


def define_args_parser():
    parser = argparse.ArgumentParser(
        description="MDVRPTW with RL"
    )

    # Misc
    parser.add_argument('--max-epoch',
                        type=int,
                        default=1000,
                        help='Maximum epoch.')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='Batch size for train the agent.')
    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help='Learning rate for optimizer')

    # Simulation
    parser.add_argument('--title',
                        type=str,
                        default="mpn",
                        help="experiment title savefiles' name")
    parser.add_argument('--workload-path',
                        type=str,
                        default="workloads-nasa-cleaned-8host.json",
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
