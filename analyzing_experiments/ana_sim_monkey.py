from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *

analyzing_pipeline = [
    #'analyze_model_perf_for_each_exp',
     #'run_scores_for_each_exp',
    # 'run_2d_inits_for_each_exp',
    # 'extract_1d_for_each_exp',
    'analyze_model_perf_for_each_data_proportion',
]

exp_folders = [
     'exp_sim_monkeyV',
    #'exp_sim_monkeyW',
]


## perf
if 'analyze_model_perf_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                             additional_rnn_keys={'model_identifier_keys': ['agent_name']},
                             additional_cog_keys={'model_identifier_keys': ['agent_name']},
                                 has_rnn=True,
                                 include_acc=True,
                                 dataset_loading_every_time=True,
                                 has_cog=False) # multiple datasets in one folder; must be re-loaded every time


# dynamics
if 'run_scores_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_scores_exp(exp_folder, pointwise_loss=True, has_cog=False)#,model_filter={'agent_name': 'LS0_seed0', 'cog_type': 'MB0s'})

if 'run_2d_inits_for_each_exp' in analyzing_pipeline:
    for exp_folder in exp_folders:
        run_2d_inits_exp(exp_folder, grid_num=50)

if 'analyze_model_perf_for_each_data_proportion' in analyzing_pipeline:
    for exp_folder in exp_folders:
        find_best_models_for_exp(exp_folder, 'PRLCog',
                             additional_rnn_keys={'model_identifier_keys':['trainval_percent'],'model_based':{"base","100_trainval"}}, has_cog=False
                             #additional_cog_keys={'model_identifier_keys':['trainval_percent']}
                             )
