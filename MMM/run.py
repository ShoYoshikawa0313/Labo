import os
import argparse
from time import time 
import datetime
from optimizer import Random_Optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="Model1")
    parser.add_argument("-composition_file", type=str, default="composition.yaml")
    parser.add_argument('-root_output_dir', type=str, default="results")
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-processes', type=int, default=1)
    parser.add_argument('-immigration', type=bool, default=True)
    parser.add_argument('-immigration_freq', type=int, default=5)

    args = parser.parse_args()

    if not os.path.exists(args.root_output_dir):
        os.mkdir(args.root_output_dir)

    dt = datetime.datetime.now() + datetime.timedelta(hours=9)
    args.root_output_dir = os.path.join(args.root_output_dir,dt.strftime('%m-%d_%H:%M'))
    if not os.path.exists(args.root_output_dir):
        os.mkdir(args.root_output_dir)

    start_time = time()

    optimizer = Random_Optimizer(args=args)
    optimizer.optimize()

    end_time = time()

    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

