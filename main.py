"""
The main entry.

Typical usage:
python main.py -t test_agent_training
python main.py -t test_agent_running

python main.py -a ana_some_script
python main.py -p plotting_some_script

"""

import os
import argparse
from pathlib import Path
import importlib


if __name__ ==  '__main__': # required by multiprocessing

    parser = argparse.ArgumentParser()
    # parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
    parser.add_argument('-t', '--train', help='Training', nargs='+', default='none')
    parser.add_argument('-a', '--analyze', help='Analyzing', nargs='+', default='none')
    parser.add_argument('-p', '--plot', help='Plotting', nargs='+', default='none')
    # parser.add_argument('-j', '--jobs', help='num of jobs', type=str, default='1')

    args = parser.parse_args()

    for item in args.__dict__.items():
        print(item)
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Training
    if args.train == 'none':
        args.train = []
    for name in args.train:
        script = importlib.import_module(f'training_experiments.{name}')


    # Analysis
    if args.analyze == 'none':
        args.analyze = []
    for name in args.analyze:
        script = importlib.import_module(f'analyzing_experiments.{name}')

    # Plot
    if args.plot == 'none':
        args.plot = []
    for name in args.plot:
        script = importlib.import_module(f'plotting_experiments.{name}')