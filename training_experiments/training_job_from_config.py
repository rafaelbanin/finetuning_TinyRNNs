
import sys
sys.path.extend(['.', '..'])
# when running this file, sys.path only includes ['..../training_experiments'].
# when running python training_experiments/training_job_from_config.py,
# the current working directory cognitive_dynamics is added to sys.path via '.'
# when running python training_job_from_config.py,
# the current working directory training_experiments is added to sys.path via '..'

from pathlib import Path
import argparse
import joblib
from utils import goto_root_dir
from training_experiments.training import behavior_cv_training

goto_root_dir.run()
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--train', help='Training', nargs='+', default='none')
parser.add_argument('-n', '--njobs', help='Num of jobs', nargs='+', default='1')
args = parser.parse_args()

if args.train == 'none':
    raise ValueError('No training experiment is specified.')
for name in args.train:
    c = joblib.load(name)
    n_jobs = int(args.njobs[0])
    behavior_cv_training(c, n_jobs=n_jobs, verbose_level=1)