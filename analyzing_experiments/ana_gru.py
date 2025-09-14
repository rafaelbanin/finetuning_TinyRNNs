"""
Analyze the 1-d GRU.
"""
import sys
from plotting_experiments.plotting import *
import numpy as np

sys.path.append('..')
from utils import goto_root_dir
from datasets import Dataset
from agents import RNNAgent
from pathlib import Path
from training_experiments import training
from path_settings import *
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from path_settings import *
goto_root_dir.run()

def torch_uniform(a, b, size):
    return (b - a) * torch.rand(size,dtype=torch.float64) + a

def reinit_gru(gru):
    input_size = 3
    hidden_size = 1

    rg = 10
    new_weight_ih_l0 = torch_uniform(-rg, rg, (hidden_size * 3, input_size))
    new_weight_hh_l0 = torch_uniform(-rg, rg, (hidden_size * 3, hidden_size))
    new_bias_ih_l0 = torch_uniform(-rg, rg, (hidden_size * 3))
    new_bias_hh_l0 = torch_uniform(-rg, rg, (hidden_size * 3))

    gru.weight_ih_l0 = nn.Parameter(new_weight_ih_l0)
    gru.weight_hh_l0 = nn.Parameter(new_weight_hh_l0)
    gru.bias_ih_l0 = nn.Parameter(new_bias_ih_l0)
    gru.bias_hh_l0 = nn.Parameter(new_bias_hh_l0)

## If testing an initialized rnn agent
config = {
    ### dataset info; other dataset can be loaded by changing the dataset name
    'dataset': 'BartoloMonkey',
    'behav_data_spec': {'animal_name': 'V', 'filter_block_type': 'both', 'block_truncation': (10, 70)}, # 'both' for all blocks
    'behav_format': 'tensor',
    ### model info
    'rnn_type': 'GRU',
    'input_dim': 3,
    'hidden_dim': 1,
    'output_dim': 2,
    'device': 'cpu', #'cpu',
    'trainable_h0': False,
    'readout_FC': True,
    'one_hot': False,
    'seed': 0,
    'model_path': 'test_agent_running/test_rnn_agent',
    }
# behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
# behav_dt = behav_dt.behav_to(config)  # transform format following specifications
# print('Data block num', behav_dt.batch_size)
# behav_data = behav_dt.get_behav_data([0], {'behav_format': 'tensor'})
for subfig in range(9):
    plot_start()
    for seed in range(subfig*20, (subfig+1)*20):
        config['seed'] = seed
        ag = RNNAgent(config)
        reinit_gru(ag.model.rnn)
        input = torch.zeros([1, 1, 3],dtype=torch.float64)
        h0_list = np.linspace(-1,1,100,endpoint=True)
        h1_list = []
        for h0 in h0_list:
            output = ag(input, h0=torch.tensor([h0],dtype=torch.float64))
            internal = output['internal']
            #h0 = internal[0,0,0]
            h1 = internal[1,0,0]
            h1_list.append(h1.detach().numpy())

        h1_list = np.array(h1_list)
        plt.plot(h0_list,h1_list-h0_list,alpha=0.5)
    plt.xlabel('h')
    plt.ylabel('h change')
    plt.savefig(FIG_PATH / 'GRU'/ 'gru1_h_change_{}.pdf'.format(subfig), bbox_inches='tight')
    plt.show()

