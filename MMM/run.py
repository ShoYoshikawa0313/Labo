import os
import argparse
from time import time 
import datetime
from optimizer import Optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="Model1")
    parser.add_argument("-composition_file", type=str, default="composition.yaml")
    parser.add_argument('-root_output_dir', type=str, default="results")
    parser.add_argument('-dir_name',type=str, default="")
    parser.add_argument('-seed', type=int, default=0)
    parser.add_argument('-one_island',type=bool, default=False)

    args = parser.parse_args()

    if not os.path.exists(args.root_output_dir):
        os.mkdir(args.root_output_dir)
    
    if args.dir_name == "":
        dt = datetime.datetime.now() + datetime.timedelta(hours=9)
        args.root_output_dir = os.path.join(args.root_output_dir,dt.strftime('%m-%d_%H:%M'))
    else:
        args.root_output_dir = os.path.join(args.root_output_dir,args.dir_name)

    if not os.path.exists(args.root_output_dir):
        os.mkdir(args.root_output_dir)

    start_time = time()

    optimizer = Optimizer(args=args)
    
    optimizer.optimize()

    end_time = time()

    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

