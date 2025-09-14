import matplotlib.pyplot as plt

from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from plotting_experiments.plotting import *
from utils import goto_root_dir
goto_root_dir.run()

def get_logit_from_path(model_path):
    model_logits = []
    with set_os_path_auto():
        model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
        model_scores = np.concatenate(model_pass['scores'], axis=0) # concat over blocks
        assert len(model_scores.shape) == 2
        if 2<= model_scores.shape[1] <= 4:
            # one-step tasks
            for i in range(1, model_scores.shape[1]):
                model_logits.append(model_scores[:,0] - model_scores[:,i])
            model_logits = np.concatenate(model_logits, axis=0)
        elif model_scores.shape[1] == 6:
            # two-step tasks
            model_logits.append(model_scores[0::2,0] - model_scores[0::2,1])
            model_logits.append(model_scores[1::2,2] - model_scores[1::2,3])
            model_logits.append(model_scores[1::2,4] - model_scores[1::2,5])
            model_logits = np.concatenate(model_logits, axis=0)
        else:
            raise ValueError(f'Unexpected model_scores shape {model_scores.shape}')
    return model_logits

def combine_logits_for_one_model(model_summary, model_name, model_selector, num_outerfold, additional_info=[]):
    """
    From the model_summary, select the model with model_selector (only one model should be selected for each model_name),
    and combine the logits from the model_paths that contain num_outerfold instances.
    """
    model_summary_selected = model_summary[model_summary.apply(model_selector, axis=1)]
    if len(model_summary_selected) == 0:
        print(f'Warning: no model selected for {model_name}')
        return []
    elif len(model_summary_selected) > 2:
        print(f'Warning: {len(model_summary_selected)} models selected for {model_name}')
        return []

    model_paths = model_summary_selected.iloc[0]['model_path']
    model_info = {}
    if 'total_trial_num' in additional_info:
        row = model_summary_selected.iloc[0]
        train_trial_num = row['mean_train_trial_num']
        val_trial_num = row['mean_val_trial_num']
        total_trial_num = train_trial_num + val_trial_num
        model_info['total_trial_num'] = total_trial_num
    assert len(model_paths) == num_outerfold
    model_logits_list = []
    for model_path in model_paths:
        model_logits = get_logit_from_path(model_path)
        model_logits_list.append(model_logits)
    return model_logits_list, model_info

def compile_summary_for_exps(exp_folders, additional_rnn_keys={}, rnn_filter=None,
                          additional_cog_keys={}, has_rnn=True, has_cog=True,
                          lambda_filter=None, lambda_filter_name='', model_selectors=None, num_outerfold = 10, additional_info=[],
                          ):
    assert model_selectors is not None
    rnn_summary, cog_summary = combine_exps(exp_folders, additional_rnn_keys, rnn_filter,
                                       additional_cog_keys, None, None, has_rnn, has_cog,
                                       lambda_filter, lambda_filter_name, combine_df_type='best_summary')

    if has_rnn:
        model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC'] + additional_rnn_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        model_identifier_keys = [k for k in model_identifier_keys if k in rnn_summary.columns]
        [rnn_summary[k].fillna('unspecified', inplace=True) for k in model_identifier_keys ]
        rnn_summary = rnn_summary.groupby(model_identifier_keys, as_index=False).agg(
            model_path=('model_path', list),
            mean_train_trial_num=('train_trial_num','mean'),
            mean_val_trial_num=('val_trial_num','mean'),
        )

    if has_cog:
        model_identifier_keys = ['cog_type'] + additional_cog_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        model_identifier_keys = [k for k in model_identifier_keys if k in cog_summary.columns]
        [cog_summary[k].fillna('unspecified', inplace=True) for k in model_identifier_keys]
        cog_summary = cog_summary.groupby(model_identifier_keys, as_index=False).agg(
            model_path=('model_path', list),
            mean_train_trial_num=('train_trial_num','mean'),
            mean_val_trial_num=('val_trial_num','mean'),
        )

    model_summary = pd.concat([rnn_summary, cog_summary], axis=0, ignore_index=True)
    # assert model_path list has certain length
    model_summary['model_path_len'] = model_summary['model_path'].apply(lambda x: len(x))
    print('model_path_len: value count\n', model_summary['model_path_len'].value_counts())
    print(f'model_path_len is expected to be {num_outerfold} for each model row')
    # with pd_full_print_context():
    #     print(model_summary)

    model_logits_dict = {}
    model_info_dict = {}
    for model_name, model_selector in model_selectors.items():
        model_logits, model_info = combine_logits_for_one_model(model_summary, model_name, model_selector, num_outerfold, additional_info=additional_info)
        if len(model_logits):
            model_logits_dict[model_name] = model_logits
            model_info_dict[model_name] = model_info

    logit_corr_list = np.zeros((len(model_selectors), len(model_selectors), num_outerfold, num_outerfold))
    model_names = list(model_selectors.keys())
    for model_i in range(len(model_names)):
        for model_j in range(len(model_names)):
            for fold_i in range(num_outerfold):
                for fold_j in range(num_outerfold):
                    corr = np.corrcoef(model_logits_dict[model_names[model_i]][fold_i],
                                             model_logits_dict[model_names[model_j]][fold_j])[0,1]
                    logit_corr_list[model_i, model_j, fold_i, fold_j] = corr
    logit_corr_mat = logit_corr_list.mean(axis=(2,3))
    return logit_corr_list, logit_corr_mat, model_info_dict

def plot_logit_corr_matrix(logit_corr_mat, model_names, compile_exp_folder):
    plt.figure(figsize=(8,8))
    plt.imshow(logit_corr_mat, vmin=0, vmax=1, cmap='hot')
    plt.xticks(range(len(model_names)), model_names, rotation=90)
    plt.yticks(range(len(model_names)), model_names)
    plt.colorbar()
    plt.savefig(FIG_PATH / compile_exp_folder / f'logit_corr_matrix.pdf', bbox_inches="tight")
    plt.show()

def plot_logit_corr_barplot(logit_corr_list, model_names, compile_exp_folder):
    def wrap_corr_barplot(x1, x2, xticklabels=['self', 'other'], other_model_names=[]):
        plot_start(figsize=(2, 2))
        half_width = 0.3
        plt.scatter(np.random.rand(len(x1))*half_width*2 - half_width, x1,
                    facecolors='none', edgecolors='grey', alpha=0.2, s=2)
        plt.hlines(x1.mean(), - half_width, half_width, color='k', linestyle='--', linewidth=1)

        for idx, model_name in enumerate(other_model_names):
            plt.scatter(np.random.rand(len(x2[idx]))*half_width*2 + 1 - half_width, x2[idx],
                        facecolors='none', edgecolors='grey', alpha=0.2, s=2)
            # add model name as text
            plt.text(1, x2[idx].mean(), model_name, ha='right', va='center', fontsize=8)
        plt.hlines(x2.mean(), 1 - half_width, 1 + half_width, color='k', linestyle='--', linewidth=1)

        plt.xticks([0,1], xticklabels, rotation=45)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.ylabel('Model logit correlation')
        plt.savefig(FIG_PATH / compile_exp_folder / f'logit_corr_barplot_{xticklabels[0]}.pdf', bbox_inches="tight")
        plt.show()

    assert model_names[0] == 'GRU1' and model_names[1] == 'GRU2', model_names
    wrap_corr_barplot(logit_corr_list[0, 0, :, :].flatten(), logit_corr_list[0, 2:, :, :],#.flatten(),
                      ['Within GRU (d=1)', 'GRU (d=1) & other'], other_model_names=model_names[2:])
    wrap_corr_barplot(logit_corr_list[1, 1, :, :].flatten(),
                      # np.concatenate(
                      #     [logit_corr_list[1, :1, :, :],
                           logit_corr_list[1, 2:, :, :],#], axis=0).flatten(),
                    ['Within GRU (d=2)', 'GRU (d=2) & other'], other_model_names=model_names[2:]
                      )
for exp_folder in [
    # 'exp_monkeyV_dataprop',
    # 'exp_seg_millerrat55_dataprop',
    ]:
    trainval_percent_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    model_selectors = {
        f'GRU2-{p}': lambda row, p=p: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 2 and row['trainval_percent'] == p
        for p in trainval_percent_list
    }
    logit_corr_list, logit_corr_mat, info = compile_summary_for_exps([exp_folder],
                              additional_rnn_keys={'model_identifier_keys':['trainval_percent']},
                               additional_cog_keys={'model_identifier_keys':['trainval_percent']},
                             rnn_filter={'readout_FC': True},
                              model_selectors=model_selectors,
                             additional_info=['total_trial_num']
                             )
    trial_counts = [info[model_name]['total_trial_num'] for model_name in model_selectors.keys()]
    plot_start(figsize=(2,2))
    plt.plot(trial_counts, np.diag(logit_corr_mat), marker='x', markersize=5)
    plt.ylabel('Logit correlation')
    plt.xlabel('# Trials available')
    plt.yticks([0.8, 0.9, 1])
    plt.savefig(FIG_PATH / exp_folder / 'logit_corr_vs_trial_num.pdf', bbox_inches="tight")
    plt.show()

    # model_names = list(model_selectors.keys())
    # plot_logit_corr_matrix(logit_corr_mat, model_names, exp_folder)
    # plot_logit_corr_barplot(logit_corr_list, model_names, exp_folder)


for exp_folder in [
    'exp_monkeyV',
    'exp_monkeyW',
    ]:
    model_selectors = {
        'GRU1': lambda row: row['rnn_type'] == 'SGRU' and row['hidden_dim'] == 1,
        'GRU2': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 2,
        'MB0s': lambda row: row['cog_type'] == 'MB0s',
        'LS0': lambda row: row['cog_type'] == 'LS0',
        'MB0': lambda row: row['cog_type'] == 'MB0',
        'MB1': lambda row: row['cog_type'] == 'MB1',
        'MB0off': lambda row: row['cog_type'] == 'MB0off',
        'MF0sp': lambda row: row['cog_type'] == 'MF0sp',
        'MB0p': lambda row: row['cog_type'] == 'MB0p',
        'RC': lambda row: row['cog_type'] == 'RC',
    }
    logit_corr_list, logit_corr_mat = compile_summary_for_exps([exp_folder],
                              additional_rnn_keys={'model_identifier_keys': ['expand_size']},
                             rnn_filter={'readout_FC': True,
                                          'symm': 'none',
                                          'finetune': 'none',
                                          'complex_readout': 'none'},
                              model_selectors=model_selectors,
                             )
    model_names = list(model_selectors.keys())
    plot_logit_corr_matrix(logit_corr_mat, model_names, exp_folder)
    plot_logit_corr_barplot(logit_corr_list, model_names, exp_folder)

for exp_folder in [
    'exp_seg_millerrat55',
    'exp_seg_millerrat64',
    'exp_seg_millerrat70',
    'exp_seg_millerrat71',
    ]:
    model_selectors = {
        'GRU1': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 1,
        'GRU2': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 2,
        'MFs': lambda row: row['cog_type'] == 'MFs',
        'MB0s': lambda row: row['cog_type'] == 'MB0s',
        'LS0': lambda row: row['cog_type'] == 'LS0',
        'MB0': lambda row: row['cog_type'] == 'MB0',
        'MB1': lambda row: row['cog_type'] == 'MB1',
        'MXs': lambda row: row['cog_type'] == 'MXs',
        'RC': lambda row: row['cog_type'] == 'RC',
        'Q(0)': lambda row: row['cog_type'] == 'Q(0)',
        'Q(1)': lambda row: row['cog_type'] == 'Q(1)',
    }
    logit_corr_list, logit_corr_mat = compile_summary_for_exps([exp_folder],
                              additional_rnn_keys={},#'model_identifier_keys': ['expand_size']},
                             rnn_filter={'readout_FC': True,
                                          'symm': 'none',
                                          'finetune': 'none',},
                                model_selectors=model_selectors,
                             )
    model_names = list(model_selectors.keys())
    plot_logit_corr_matrix(logit_corr_mat, model_names, exp_folder)
    plot_logit_corr_barplot(logit_corr_list, model_names, exp_folder)


# not finished yet
# for dt, exp_folder, model_selectors in [
#     (Dataset('SuthaharanHuman', behav_data_spec={}, verbose=False), 'exp_Suthaharan_splittrial',{'GRU': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 3}),
#     (Dataset('BahramiHuman', behav_data_spec={}, verbose=False), 'exp_Bahrami_splittrial', {'GRU': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 4}),
#     (Dataset('GillanHuman', behav_data_spec={}, verbose=False), 'exp_Gillan_splittrial', {'GRU': lambda row: row['rnn_type'] == 'GRU' and row['hidden_dim'] == 3}),
#     ]:
#         for block in range(dt.batch_size):
#             logit_corr_list, _ = compile_summary_for_exps([exp_folder],
#                                       additional_rnn_keys={},
#                                      rnn_filter={'readout_FC': False, 'block': block},
#
#                                         model_selectors=model_selectors,
#                                  )
#     model_names = list(model_selectors.keys())
#     plot_logit_corr_barplot(logit_corr_list, model_names, exp_folder)