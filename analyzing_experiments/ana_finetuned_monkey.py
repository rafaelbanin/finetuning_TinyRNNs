from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing import *


analyzing_pipeline = [
    'run_initial_loss_for_two_agents', # Novo passo de análise
]

exp_folders = ['exp_finetuned_monkeyV']

if 'run_initial_loss_for_two_agents' in analyzing_pipeline:
    for exp_folder in exp_folders:
        filters = {
            'model_based': ['100_pre_trained', '70_pre_trained', '50_pre_trained', '20_pre_trained','base'],
            'trainval_percent': [0,10,20,50,70,100],
            'agent_name': ['MB1_seed0', 'Q(0)_seed0','No_seed']
        }
        sort_order = ['agent_name', 'model_based', 'rnn_type', 'hidden_dim']

        aggregated_perf_df, detailed_df = find_initial_test_loss_for_exp(
            exp_folder,
            additional_rnn_keys={'model_identifier_keys': ['agent_name']},
            model_filters=filters,
            rnn_sort_keys=sort_order,
            save_suffix='_initial_test_two_agents'
        )
        print("Análise de test_loss inicial para dois agentes concluída.")