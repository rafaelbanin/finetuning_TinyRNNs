import os

from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utils import goto_root_dir
import joblib
from path_settings import *

goto_root_dir.run()
if False:
    exp_folder = 'RTS_agents_millerrat55'
    dts = {
        ag: Dataset('SimAgent', behav_data_spec={'agent_path': exp_folder,'agent_name': ag})
        for ag in [
            'MB0s_seed0',
            'LS0_seed0',
            'LS1_seed0',
            'MB0_seed0',
            'MB1_seed0',
            # 'MB0md_seed0',
            'RC_seed0',
            'Q(0)_seed0',
            'Q(1)_seed0'
        ]
    }
    dts |= {'m55': Dataset('MillerRat', behav_data_spec={'animal_name': 'm55'})}
    for dt_name, dt in dts.items():
        lag = 10
        X_list = []
        y_list = []
        predictor_included = ['reward-common', 'reward-rare', 'nonreward-common', 'nonreward-rare']
        predictor_num = len(predictor_included)
        for episode in range(dt.batch_size):
            action = dt.behav['action'][episode]
            stage2 = dt.behav['stage2'][episode]
            reward = dt.behav['reward'][episode]
            trial_num = len(action)
            common = (action == stage2) * 1
            pred_outcome = reward - 0.5 # +0.5 for rewarded trials, -0.5 for not rewarded trials
            pred_transition = common - 0.5 # +0.5 for common transition, -0.5 for rare transition
            pred_interact = pred_outcome * pred_transition * 2
            # +0.5 for common transition rewarded and rare transitions non-rewarded trials
            # -0.5 for rare transition rewarded and common transition non-rewarded trials
            predictors = {
                'outcome': pred_outcome,
                'transition': pred_transition,
                'interaction': pred_interact,
                'reward-common': reward * common * 0.5,
                'reward-rare': reward * (1 - common) * 0.5,
                'nonreward-common': (1 - reward) * common * 0.5,
                'nonreward-rare': (1 - reward) * (1 - common) * 0.5,
            }
            X = np.zeros([trial_num - lag, lag, predictor_num])
            y = np.zeros([trial_num - lag])
            for i in range(trial_num - lag):
                y[i] = (action[i + lag] == action[i + lag - 1]) * 1
                for l in range(lag):
                    for j, pred in enumerate(predictor_included):
                        X[i, l, j] = predictors[pred][i + l]
            X_list.append(X)
            y_list.append(y)
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        total_trial_num = X.shape[0]
        print(X.shape, y.shape)
        clf = LogisticRegression(random_state=0).fit(X.reshape([total_trial_num, lag * predictor_num]), y)
        coef = clf.coef_.reshape([lag, predictor_num])
        ana_exp_path = ANA_SAVE_PATH / exp_folder
        filename = '.'.join(predictor_included)
        filename = f'{dt_name}.lag{lag}.{filename}.pkl'
        os.makedirs(ana_exp_path, exist_ok=True)
        joblib.dump({
            'predictor_included': predictor_included,
            'lag': lag,
            'coef': coef,
        }, ana_exp_path / filename)


from plotting_experiments.plotting import plot_start

fix = plot_start()
for animal_name in ['m55', 'm64','m70','m71']:
    # load behav
    config = {
        ### dataset info
      'dataset': 'MillerRat',
      'behav_format': 'tensor',
      'behav_data_spec': {'animal_name': animal_name, 'max_segment_length': 150},
    }
    behav_data_spec = config['behav_data_spec']
    dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec)
    behav = dt.behav
    lag = 3
    model_path_dict = {
        'm55':{
            'GRU':r'exp_seg_millerrat55\rnn_type-GRU.hidden_dim-2.readout_FC-False.l1_weight-1e-05\outerfold3_innerfold4_seed1',
            'MB':r'exp_seg_millerrat55\cog_type-MB1\outerfold3_innerfold4_seed0',
        },
        'm64':{
            'GRU':r'exp_seg_millerrat64\rnn_type-GRU.hidden_dim-2.readout_FC-False.l1_weight-1e-05\outerfold5_innerfold4_seed0',
            'MB':r'exp_seg_millerrat64\cog_type-MB1\outerfold2_innerfold6_seed0',
        },
        'm70':{
            'GRU':r'exp_seg_millerrat70\rnn_type-GRU.hidden_dim-2.readout_FC-False.l1_weight-1e-05\outerfold0_innerfold3_seed2',
            'MB':r'exp_seg_millerrat70\cog_type-MB1\outerfold0_innerfold8_seed0',
        },
        'm71':{
            'GRU':r'exp_seg_millerrat71\rnn_type-GRU.hidden_dim-2.readout_FC-True.l1_weight-1e-05\outerfold0_innerfold1_seed2',
            'MB':r'exp_seg_millerrat71\cog_type-MB1\outerfold0_innerfold3_seed0',
        }
    }
    GRU_scores = joblib.load(
        ANA_SAVE_PATH / model_path_dict[animal_name]['GRU'] / 'total_scores.pkl')['scores']
    MB_scores = joblib.load(
        ANA_SAVE_PATH / model_path_dict[animal_name]['MB'] / 'total_scores.pkl')['scores']

    predictor_included = ['reward-common', 'reward-rare', 'nonreward-common', 'nonreward-rare']
    predictor_num = len(predictor_included)

    X_list = []
    y_list = []
    GRU_y_list = []
    MB_y_list = []
    for episode in range(dt.batch_size):
        action = dt.behav['action'][episode]
        stage2 = dt.behav['stage2'][episode]
        reward = dt.behav['reward'][episode]
        GRU_prob = np.exp(GRU_scores[episode]) / np.exp(GRU_scores[episode]).sum(axis=1, keepdims=True)
        MB_prob = np.exp(MB_scores[episode]) / np.exp(MB_scores[episode]).sum(axis=1, keepdims=True)
        # sample action from prob
        GRU_action = np.zeros_like(action)
        MB_action = np.zeros_like(action)
        for i in range(len(action)):
            GRU_action[i] = np.random.choice(np.arange(2), p=GRU_prob[i])
            MB_action[i] = np.random.choice(np.arange(2), p=MB_prob[i])
        trial_num = len(action)
        common = (action == stage2) * 1
        predictors = {
            'reward-common': reward * common * 1,
            'reward-rare': reward * (1 - common) * 1,
            'nonreward-common': (1 - reward) * common * 1,
            'nonreward-rare': (1 - reward) * (1 - common) * 1,
        }
        X = np.zeros([trial_num - lag, lag, predictor_num])
        y = np.zeros([trial_num - lag])
        GRU_y = np.zeros([trial_num - lag])
        MB_y = np.zeros([trial_num - lag])
        for i in range(trial_num - lag):
            y[i] = (action[i + lag] == action[i + lag - 1]) * 1
            GRU_y[i] = (GRU_action[i + lag] == action[i + lag - 1]) * 1
            MB_y[i] = (MB_action[i + lag] == action[i + lag - 1]) * 1
            for l in range(lag):
                for j, pred in enumerate(predictor_included):
                    X[i, l, j] = predictors[pred][i + l]
        X_list.append(X)
        y_list.append(y)
        GRU_y_list.append(GRU_y)
        MB_y_list.append(MB_y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    GRU_y = np.concatenate(GRU_y_list, axis=0)
    MB_y = np.concatenate(MB_y_list, axis=0)

    total_trial_num = X.shape[0]
    print(X.shape, y.shape)
    def wrap_Xy(X, y,label,marker):
        clf = LogisticRegression(random_state=0).fit(X.reshape([total_trial_num, lag * predictor_num]), y)
        coef = clf.coef_.reshape([lag, predictor_num]).sum(axis=0)
        CR, UR, CO, UO = coef
        # CR=common reward, UR=uncommon reward, CO=common omission, UO=uncommon omission
        MF_idx = (CR+UR) - (CO+UO)
        planning_idx = (CR-UR) + (UO-CO)
        # plt.scatter(planning_idx, MF_idx, s=100, c='orange', marker=marker, label=label)
        print(label,animal_name, planning_idx, MF_idx)
        return planning_idx, MF_idx
    planning_idx, MF_idx = wrap_Xy(X, y, 'data', 'o')
    planning_idx_GRU, MF_idx_GRU = wrap_Xy(X, GRU_y, 'GRU', 's')
    planning_idx_MB, MF_idx_MB = wrap_Xy(X, MB_y, 'MB', 'd')
    plt.scatter(planning_idx, planning_idx_GRU, s=30, marker='s', label='Planning index', facecolors='none', edgecolors='k',alpha=0.8)
    plt.scatter(MF_idx, MF_idx_GRU, s=30, marker='o', label='Model-free index', facecolors='none', edgecolors='k',alpha=0.8)
    # plt.scatter(planning_idx, planning_idx_MB, s=30, marker='d', label='Planning index', facecolors='none', edgecolors='b',alpha=0.8)
    # plt.scatter(MF_idx, MF_idx_MB, s=30, marker='o', label='Model-free index', facecolors='none', edgecolors='b',alpha=0.8)

plt.xlim([-2,8])
plt.ylim([-2,8])
plt.xticks([0,8])
plt.yticks([0,8])
plt.plot([-2,8],[-2,8],'k--',alpha=0.3)
plt.vlines(0, -2, 8, colors='k', linestyles='--',alpha=0.3)
plt.hlines(0, -2, 8, colors='k', linestyles='--',alpha=0.3)
# only keep the first two legend
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2])
plt.xlabel('Rat index')
plt.ylabel('GRU index')
plt.savefig(FIG_PATH / 'exp_seg_millerrat' / f'planning_MF_idx.pdf', bbox_inches='tight')
plt.close()
# plt.show()