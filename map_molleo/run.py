import os
import argparse
from time import time 

from map_optimizer import Map_Optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="Model1")
    parser.add_argument("-immigration_rate", type=float, default=0.3)
    parser.add_argument("-immigrants_size", type=int, default= 10)
    parser.add_argument("-composition_file", type=str, default="composition.yaml")
    parser.add_argument('-root_output_dir', type=str, default="results")
    parser.add_argument('-patience', type=int, default=5)
    parser.add_argument('-max_generations', type=int, default=100)
    parser.add_argument('-seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.root_output_dir):
        os.mkdir(args.root_output_dir)

    start_time = time()

    optimizer = Map_Optimizer(args=args)
    optimizer.make_islands()
    optimizer.optimize()

    end_time = time()

    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

