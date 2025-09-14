import sys
# sys.path.append("/volume/cognitive_dynamics")
from utils import goto_root_dir
goto_root_dir.run()
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing import *
from collections import OrderedDict
from datasets import Dataset
import torch
goto_root_dir.run()

def get_sgru_from_gru(gru, onehot_input_num):
    """gru: dict of gru model.state_dict()"""
    if onehot_input_num == 4:
        normal_inputs = np.array(
            [[0, 0, 0], [0, 0, 1], [1, 1, 0], [1, 1, 1]])  # shape (4,3=('action','stage2','reward'))
    elif onehot_input_num == 8:
        normal_inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0],
                                  [1, 1, 1]])  # shape (4,3=('action','stage2','reward'))
    else:
        raise NotImplementedError
    gru2sgru = OrderedDict()
    if 'dummy_param' in gru.keys():
        gru2sgru['dummy_param'] = gru['dummy_param'] # shape (0,)
    else:
        gru2sgru['dummy_param'] = torch.empty(0)
    gru2sgru['rnn.rnncell.weight_hh'] = gru['rnn.weight_hh_l0'][:, :, None].repeat(1,1, onehot_input_num) # shape (3 * hidden_size,1) -> (3* hidden_size, 1, 4)
    gru2sgru['rnn.rnncell.bias_ih'] = gru['rnn.bias_ih_l0'][:, None].repeat(1,onehot_input_num) # shape (3 * hidden_size,) -> (3* hidden_size, 4)
    for input_idx in range(onehot_input_num):
        normal_input = normal_inputs[input_idx]
        for i in range(len(normal_input)):
            if normal_input[i] == 1:
                gru2sgru['rnn.rnncell.bias_ih'][:, input_idx] += gru['rnn.weight_ih_l0'][:,i] # rnn.weight_ih_l0 shape: (3 * hidden_size,input_size=3)
    gru2sgru['rnn.rnncell.bias_hh'] = gru['rnn.bias_hh_l0'][:, None].repeat(1,onehot_input_num) # shape (3 * hidden_size,) -> (3* hidden_size, 4)
    if 'lin.weight' in gru.keys():
        gru2sgru['lin.weight'] = gru['lin.weight'] # shape (2, hidden_size=1)
        gru2sgru['lin.bias'] = gru['lin.bias'] # shape (2,)
    else:
        gru2sgru['lin_coef'] = gru['lin_coef']
    return gru2sgru

def update_model_path(model_path):
    model_path_id = str(model_path.parent).replace('GRU', 'SGRU')+'.finetune-True'
    model_path_name = model_path.name
    return Path(model_path_id) / model_path_name

model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC']
cv_keys = ['outer_fold', 'inner_fold']
compete_from_keys = ['l1_weight', 'seed']

for dataset in [#'BartoloMonkey', 'MillerRat',
                'AkamRat','AkamRatPRL', 'AkamRatRTS']:
    if dataset == 'BartoloMonkey':
        onehot_input_num = 4
        exp_folders = ['exp_monkeyV','exp_monkeyW']
        additional_rnn_keys=['complex_readout','symm']
    elif dataset == 'MillerRat':
        onehot_input_num = 8
        exp_folders = [
            'exp_seg_millerrat55',
            'exp_seg_millerrat64',
            'exp_seg_millerrat70',
            'exp_seg_millerrat71',
        ]
        additional_rnn_keys = ['symm']
    elif dataset == 'AkamRat':
        onehot_input_num = 8
        exp_folders = [
            'exp_seg_akamrat49',
            'exp_seg_akamrat50',
            'exp_seg_akamrat51',
            'exp_seg_akamrat52',
            'exp_seg_akamrat53',
            'exp_seg_akamrat54',
            'exp_seg_akamrat95',
            'exp_seg_akamrat96',
            'exp_seg_akamrat97',
            'exp_seg_akamrat98',
            'exp_seg_akamrat99',
            'exp_seg_akamrat100',
            'exp_seg_akamrat264',
            'exp_seg_akamrat268',
            'exp_seg_akamrat263',
            'exp_seg_akamrat266',
            'exp_seg_akamrat267',
        ]
        additional_rnn_keys = ['symm']
    elif dataset == 'AkamRatPRL':
        onehot_input_num = 4
        exp_folders = [
            'exp_seg_akamratprl388',
            'exp_seg_akamratprl383',
            'exp_seg_akamratprl382',
            'exp_seg_akamratprl380',
            'exp_seg_akamratprl368',
            'exp_seg_akamratprl367',
            'exp_seg_akamratprl361',
            'exp_seg_akamratprl360',
            'exp_seg_akamratprl359',
            'exp_seg_akamratprl358',
        ]
        additional_rnn_keys = ['symm']
    elif dataset == 'AkamRatRTS':
        onehot_input_num = 8
        exp_folders = [
            'exp_seg_akamratrts279',  # missing some GRU d=3
            'exp_seg_akamratrts278',
            'exp_seg_akamratrts277',
            'exp_seg_akamratrts275',
            'exp_seg_akamratrts274',
            'exp_seg_akamratrts273',
            'exp_seg_akamratrts272',
            'exp_seg_akamratrts271',
            'exp_seg_akamratrts270',
            'exp_seg_akamratrts269',
            ]
        additional_rnn_keys=['symm']
    else:
        raise NotImplementedError
    id_keys = model_identifier_keys + additional_rnn_keys + cv_keys + compete_from_keys

    for exp_folder in exp_folders:
        perf = combine_exp_summary(exp_folder, id_keys=id_keys, filter_dict={'agent_type': 'RNN'})
        # perf_sgru = perf[(perf['rnn_type'] == 'SGRU') & (perf['hidden_dim'] == 1)]

        perf_gru = perf[(perf['rnn_type'] == 'GRU')]
        gru_row = perf_gru.iloc[-1]
        for i, gru_row in perf_gru.iterrows():
            config = transform_model_format(gru_row, source='row', target='config')
            gru = transform_model_format(gru_row, source='row', target='agent')

            # for key, value in gru.model.state_dict().items():
            #     print(key, value.shape)
            # dt = Dataset(config['dataset'], behav_data_spec={'all_trial_type': True}).behav_to({'behav_format':'tensor'})
            # inp = dt.get_behav_data(np.arange(dt.batch_size), {'behav_format': 'tensor'})['input'].cuda()
            # gru_output = gru.model(inp)
            # inp_onehot = dt.get_behav_data(np.arange(dt.batch_size), {'behav_format': 'tensor', 'one_hot': True})['input'].cuda()

            config.update({
                    'input_dim': onehot_input_num,
                    'one_hot': True,
                    'device': 'cpu',
                    'rnn_type': 'SGRU',
                    'finetune': True,
                    'model_path': update_model_path(config['model_path']),
            })
            summary_path = MODEL_SAVE_PATH / config['model_path'] / 'temp_summary.pkl'
            if os.path.exists(summary_path):
                print(f'Agent {config["model_path"]} already exists. Skip transforming.')
                continue
            sgru = Agent(config['agent_type'], config=config)
            sgru.model.load_state_dict(get_sgru_from_gru(gru.model.state_dict(), onehot_input_num))
            sgru.save()
            print('Upgraded model saved to', config['model_path'])