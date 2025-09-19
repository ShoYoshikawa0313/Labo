import os
import argparse
import yaml
import os
import sys
from tdc import Oracle
from time import time 

from optimizer import GB_GA_Optimizer as Optimizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--population_size', type=int, default=120) #最小 100
    parser.add_argument('--offspring_size', type=int, default=70)
    parser.add_argument('--mutation_rate', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--mol_lm', type=str, default="GPT_OSS", choices=["BioT5", "MoleculeSTM", "GPT-4", "GPT_OSS"])
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_generations', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tasks', nargs="+", default=["qed"])
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    start_time = time()

    for task_name in args.tasks:

        evaluator = Oracle(name = task_name)
        optimizer = Optimizer(args=args)

        optimizer.optimize(evaluator=evaluator, seed=args.seed)

    end_time = time()

    hours = (end_time - start_time) / 3600.0
    print('---- The whole process takes %.2f hours ----' % (hours))


if __name__ == "__main__":
    main()

