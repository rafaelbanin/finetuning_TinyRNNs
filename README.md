# cognitive_dynamics

This repository contains the code for the paper: 
[Discovering Cognitive Strategies with Tiny Recurrent Neural Networks](https://www.biorxiv.org/content/10.1101/2023.04.12.536629v3)



## System requirements
- Tested on Windows, Linux, and MacOS
- Python <= 3.10
- Python packages: pytorch (using cuda), scikit-learn, numpy, scipy, statsmodels, pandas=1.5.3, matplotlib, joblib=1.2.0, tqdm, plotting, numba, adjustText

## Installation guide
1. Clone the repository
2. Install the required packages (~ minutes; recommended to use a virtual environment like conda)


[//]: # (Instructions to run on data)

[//]: # (Expected output)

[//]: # (Expected run time for demo)

## Instructions for use and Demo
- The repo is structured as follows:
```
cognitive_dynamics
|-- main.py (the main entry point to run the experiments)
|-- path_settings.py (set global paths to models, results, and figures)
|-- data_path.json (contains the paths to the data files)
|-- agents (contains the agent and trainer classes)
|   |-- CogAgent.py (the base class for the cognitive models)
|   |-- CogAgentTrainer.py
|   |-- RNNAgent.py (the base class for the RNN models)
|   |-- RNNAgentTrainer.py
|   |-- ...
|-- datasets
|   |-- BaseTwoStepDataset.py (the base class for most datasets)
|   |-- BartoloMonkeyDayaset.py
|   |-- ...
|-- training_experiments
|   |-- training.py (the core file for training all experiments)
|   |-- exp_monkeyV.py (the file for training the monkeyV experiment)
|   |-- exp_monkeyV_minimal.py
|   |-- ...
|-- analyzing_experiments (the files for analyzing the trained models)
|   |-- analyzing.py 
|   |-- analyzing_dynamics.py (the core file for analyzing dynamics of trained models)
|   |-- analyzing_perf.py (the core file for analyzing performance of trained models)
|   |-- ...
|   |-- ana_monkey.py (the file for analyzing the monkey experiment)
|   |-- ana_monkey_minimal.py
|   |-- ...
|-- plotting_experiments (the files for plotting the results)
|   |-- plotting.py (the core file for plotting all experiments)
|   |-- plotting_monkey.py (the file for plotting the monkey experiment)
|   |-- plotting_monkey_minimal.py
|   |-- ...
|-- simulating_experiments (the files for simulating artificial data)
|   |-- simulate_experiment.py (the core file for simulating all experiments)
|   |-- allagents_monkey_all.py (the file for simulating the models trained on the monkey experiment)py 
|   |-- ...
|-- tasks
|   |-- akam_task.py (collection of two-step tasks)
|-- utils (contains other useful functions)
|   |-- goto_root_dir.py (go to the root directory of the project; can be called at the beginning of any script)

```

- For each experiment, the models are first trained on a specific dataset ("training_experiments/exp_\*.py"), 
analyzed using various metrics ("analyzing_experiments/ana\*.py"), then plotted ("plotting_experiments/plotting\*.py").
The models can also be used to simulate artificial data on a given task ("simulating_experiments/allagents_\*.py").
- Warning: Training all the experiments will require a long time (~ weeks on hundreds of GPUs), leading to millions model instances.
- the Bartolo's Monkey dateset for the demo below, as well as other datasets, can be downloaded from the links in "data availability" section of the paper.
- Simpler demo scripts feasible on a personal computer (each file can either be executed directly from the console or from the main.py entry):
  - those scripts ending with "_minimal" in training_experiments: python main.py -t exp_monkeyV_minimal
  - those scripts ending with "_minimal" in analyzing_experiments: python main.py -a ana_monkey_minimal
  - those scripts ending with "_minimal" in plotting_experiments: python main.py -p plotting_monkey_minimal
- To train RNNs, there are two training modes:
  - "behavior_cv_training_job_combination" function will generate independent yaml files to submit jobs to a cluster (one job for each model) 
  - "behavior_cv_training_config_combination" function will train all models locally
- To train cognitive models, "behavior_cv_training_config_combination" function is used and wrapped in the block "if __name__ ==  '__main__' or '.' in __name__:" to utilize multiprocess training.



[//]: # (### Multiprocessing issues:)

[//]: # (- Advanced system settings)

[//]: # (- Advanced tab)

[//]: # (- Performance - Settings button)

[//]: # (- Advanced tab - Change button)

[//]: # (- Uncheck the "Automatically... " checkbox)

[//]: # (- Select the System managed size option box.)

