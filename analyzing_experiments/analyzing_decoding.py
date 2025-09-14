import pandas as pd

from .analyzing import *
from .analyzing_dynamics import *
import os
from datasets import Dataset
import numpy as np
from utils import goto_root_dir
import joblib
from path_settings import *
from datasets import Dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time


def extract_pcs(X, cross_validation=0, pca_num=-1):
    """ Extract PCs from X.

    Args:
        X: (n_samples, n_features)

    Returns:
        PC_result: dict
    """
    assert len(X.shape) == 2, X.shape
    n_samples, n_features = X.shape
    if pca_num == -1:
        pca_num = min(n_samples, n_features)
    pca = PCA(pca_num)
    if cross_validation == 0: # no cross-validation
        X = pca.fit_transform(X - X.mean(0))
        var_ratio = pca.explained_variance_ratio_
        var_cum = np.cumsum(var_ratio)
        part_ratio = var_ratio.sum()**2/(var_ratio**2).sum()
        return {
            'PC': X,
            'variance_ratio': var_ratio,
            'variance_cumulative_ratio': var_cum,
            'participation_ratio': part_ratio,
        }
    else:
        # Kfold
        kf = KFold(n_splits=cross_validation, shuffle=True, random_state=0)
        X_PCs = np.zeros((n_samples, pca_num))
        for idx, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            pca = PCA(pca_num)
            pca_mean = X_train.mean(0)
            pca.fit(X_train - pca_mean)
            X_PCs[test_index] = pca.transform(X_test - pca_mean)
        return {
            'PC': X_PCs,
        }


def quick_decoding(X, y, alphas=None):
    """ Quick decoding using RidgeCV.
    This model used the validation set to estimate test loss. Not recommended.
    """
    raise NotImplementedError
    assert len(X.shape) == 2, X.shape
    assert len(y.shape) == 1, y.shape
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    clf = RidgeCV(alphas=alphas,store_cv_values=True)
    clf.fit(X, y)
    loc = np.where(alphas==clf.alpha_)[0][0] # find the index of the best alpha
    mse = clf.cv_values_[:, loc].mean() # mean cross-validation error for the best alpha
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean())
    r2 = 1 - mse/null_mse # cross-validated R^2
    return r2


def quick_decoding_multidim(X, y, alphas=None):
    """ Quick decoding using RidgeCV.
    This model used the validation set to estimate test loss. Not recommended.
    """
    raise NotImplementedError
    n_targets = y.shape[1]
    assert len(X.shape) == 2, X.shape
    assert len(y.shape) == 2, y.shape
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)
    clf = RidgeCV(alphas=alphas, store_cv_values=True, alpha_per_target=True)
    clf.fit(X, y)
    # clf.alpha_ shape: (n_targets,)
    # alphas shape: (n_alphas,)
    locs = clf.alpha_.reshape([-1, 1]) == alphas.reshape([1, -1]) # shape: (n_targets, n_alphas)
    target_locs, alpha_locs = np.where(locs)
    assert (target_locs == np.arange(n_targets)).all() # each target has a best alpha
    cv_errors = clf.cv_values_[:, target_locs, alpha_locs] # CV error when each target with the best alpha shape: (n_samples, n_targets)
    mse = cv_errors.mean(0) # mean cross-validation error over samples
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean(0).reshape([1, -1]), multioutput='raw_values')
    r2 = 1 - mse/null_mse # cross-validated R^2
    return r2


def decoding_CV(X, y, alphas=None):
    if len(y.shape) == 1:
        y = y.reshape([-1, 1])
    n_targets = y.shape[1]
    assert len(X.shape) == 2, X.shape
    assert X.shape[0] == y.shape[0], (X.shape, y.shape)
    if alphas is None:
        alphas = np.logspace(-6, 6, 13)

    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    y_pred = np.zeros(y.shape)
    for idx, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RidgeCV(alphas=alphas, store_cv_values=True, alpha_per_target=True)
        clf.fit(X_train, y_train)
        y_pred[test_index] = clf.predict(X_test)
    mse = mean_squared_error(y, y_pred, multioutput='raw_values')
    null_mse = mean_squared_error(y, np.ones(y.shape) * y.mean(0).reshape([1, -1]), multioutput='raw_values')
    r2 = 1 - mse/null_mse # cross-validated R^2
    return {'mse': mse, 'null_mse': null_mse, 'r2': r2}


def pca_decoding(X_PCs, y, verbose=True, max_pca_num=-1):
    """ Decoding y from X_PCs.

    Args:
        X_PCs: (n_samples, n_PCs), extracted PCs of X
        y: (n_samples,)
        verbose: bool
        max_pca_num: int, maximum number of PCs to use

    Returns:
        R2: (n_PCs,)
    """
    sample_num, feature_num = X_PCs.shape
    assert len(y.shape) == 1, y.shape
    if max_pca_num == -1:
        max_pca_num = min(feature_num, 250)
    if verbose:
        print('max_pca_num', max_pca_num)
    R2_list = []
    start = time.time()
    for pca_num in range(1, max_pca_num+1):
        r2 = decoding_CV(X_PCs[:, :pca_num], y)['r2'][0]
        R2_list.append(r2)
        if verbose:
            print(pca_num, 'R2', r2, 'time cost', time.time() - start)
    return R2_list

def construct_behav_regressor(behav_dt):
    predictors = []
    for episode in range(len(behav_dt['action'])):
        action = behav_dt['action'][episode] - 0.5
        stage2 = behav_dt['stage2'][episode] - 0.5
        reward = behav_dt['reward'][episode] - 0.5
        action_x_stage2 = action * stage2 * 2
        action_x_reward = action * reward * 2
        stage2_x_reward = stage2 * reward * 2

        pred_temp = np.array([action, stage2, reward, action_x_stage2, action_x_reward, stage2_x_reward]).T # (n_trials, 6)
        pred_temp_prev = np.concatenate([np.zeros([1, 6]), pred_temp[:-1]], axis=0) # (n_trials, 6)
        predictors.append(np.concatenate([pred_temp, pred_temp_prev], axis=1)) # (n_trials, 12)
    predictors = np.concatenate(predictors, axis=0)
    return predictors


def run_two_model_compare_decoding(exp_folder, neuro_data_spec):
    """
    Compre two models' predicted logits and logit changes (not successful yet)
    Maybe decode these logits from neural data (not implemented yet).
    """
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    rnn_summary = rnn_summary[(rnn_summary['rnn_type'] == 'SGRU') & (rnn_summary['outer_fold'] == 8)
                              & (rnn_summary['hidden_dim'] == 2) & (rnn_summary['readout_FC'] == True)]
    rnn_summary['model_name'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    cog_summary = cog_summary[(cog_summary['cog_type'] == 'MB0s') & (cog_summary['outer_fold'] == 8)]
    cog_summary['model_name'] = cog_summary['cog_type']
    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config),
                 neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    os.makedirs(ana_path, exist_ok=True)

    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace(
        'bin_size_', 'bs')
    for session_name in dt.behav_data_spec['session_name']:
        neuro_dt, epispode_idx, _, _ = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                         remove_nan=True, shape=2, **neuro_data_spec)
        behav_dt = dt.get_behav_data(epispode_idx, {})

        pca_results = extract_pcs(neuro_dt)
        joblib.dump(pca_results, ana_path / f'{session_name}_{fname}_pca.pkl')
        X_PCs = pca_results['PC']

        two_model_logits = {}
        for idx, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
            model_path = transform_model_format(row, source='row', target='path')
            print(model_path)
            model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
            trial_types = [model_pass['trial_type'][i] for i in epispode_idx]
            trial_types = np.concatenate(trial_types)
            model_scores = [model_pass['scores'][i] for i in epispode_idx]
            logits, logits_change = extract_value_changes(model_scores, value_type='logit')
            two_model_logits[row['model_name']] = logits_change
        from plotting_experiments.plotting import plot_2d_values
        plot_2d_values(two_model_logits['SGRU'], two_model_logits['MB0s'], trial_types,
                       x_range=(-5,5), y_range=(-5,5), x_label='SGRU', y_label='MB0s', title='', ref_line=True,
                       ref_x=0.0, ref_y=0.0, ref_diag=True, hist=False
                       )
        plt.show()
        syss

def run_decoding_exp(exp_folder, neuro_data_spec, analyses=None, ignore_analyzed=True,
                        predictor_neuro_data_spec=None,filters=lambda row: True,
                     ):
    """ Run decoding experiment.

    Assume that all models in the exp sharing the same dataset.
    """
    if analyses is None:
        raise ValueError('analyses is None')
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    use_based_on_test = False
    suffix = '_based_on_test' if use_based_on_test else ''
    rnn_summary = joblib.load(ana_exp_path / f'rnn_final_best_summary{suffix}.pkl')
    cog_summary = joblib.load(ana_exp_path / f'cog_final_best_summary{suffix}.pkl')

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config), neuro_data_spec=neuro_data_spec, verbose=False)
    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    if predictor_neuro_data_spec is not None:
        fname += '_predictor_' + '_'.join([f'{k}_{v}' for k, v in predictor_neuro_data_spec.items()])
    if predictor_neuro_data_spec is None:
        predictor_dt = dt
        predictor_neuro_data_spec = neuro_data_spec
    else:
        predictor_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config),
                               neuro_data_spec=predictor_neuro_data_spec)
    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    os.makedirs(ana_path, exist_ok=True)
    # ana_path.mkdir(exist_ok=True)
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace('bin_size_', 'bs')
    if 'prev_trial' in analyses:
        fname += '_prev_trial'
    if 'balanced_dichotomy' in analyses:
        fname += '_balanced'
    for session_name in dt.behav_data_spec['session_name']:
        print(fname)
        neuro_dt, epispode_idx, _, _ = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=2, **neuro_data_spec)
        # neuro_dt shape: # (total_trial_num, feat_num)
        predictor_neuro_dt, predictor_epispode_idx, _, _ = predictor_dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=2, **predictor_neuro_data_spec)
        assert (epispode_idx == predictor_epispode_idx).all()
        behav_dt = dt.get_behav_data(epispode_idx, {})
        behav_var = construct_behav_regressor(behav_dt)

        neuro_latent = extract_pcs(predictor_neuro_dt, cross_validation=12, pca_num=10)['PC']

        def wrap_decoding_save(ana_path, save_name, X, Y):
            analysis_filename = ana_path / f'{session_name}_{fname}_{save_name}.pkl'
            if analysis_filename.exists() and ignore_analyzed:
                print(analysis_filename, 'exists')
            else:
                neuron_decoding = decoding_CV(X, Y)
                neuron_R2 = neuron_decoding['r2']
                print(session_name, save_name, 'R2>0.1 proportion', np.mean(neuron_R2>0.1))
                joblib.dump(neuron_decoding, analysis_filename)

        if 'task_var_decode_neuron' in analyses:
            wrap_decoding_save(ana_path, 'task_var_decode_neuron', behav_var, neuro_dt, )

        if 'latent_decode_neuron' in analyses:
            wrap_decoding_save(ana_path, 'latent_decode_neuron', neuro_latent, neuro_dt)

        pca_results = extract_pcs(neuro_dt)
        joblib.dump(pca_results, ana_path / f'{session_name}_{fname}_pca.pkl')
        X_PCs = pca_results['PC']

        for idx, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
            if row['hidden_dim'] < 1:
                continue
            if not filters(row):
                continue
            model_path = transform_model_format(row, source='row', target='path')
            print(fname, model_path)
            if not os.path.exists(ANA_SAVE_PATH / model_path / f'total_scores.pkl'):
                print(ANA_SAVE_PATH / model_path / f'total_scores.pkl', 'not exists')
                continue
            model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
            model_scores = [model_pass['scores'][i] for i in epispode_idx]
            model_internal = [model_pass['internal'][i] for i in epispode_idx]

            ana_model_path = ANA_SAVE_PATH / model_path / 'decoding'
            os.makedirs(ana_model_path, exist_ok=True)

            _, _, full_state_vars = extract_value_changes(model_internal, value_type=0, return_full_dim=True)
            if 'value_decode_neuron' in analyses:
                wrap_decoding_save(ana_model_path, 'value', full_state_vars, neuro_dt)

            if 'latent_value_decode_neuron' in analyses:
                wrap_decoding_save(ana_model_path,'latent_value', np.concatenate([neuro_latent, full_state_vars], axis=1), neuro_dt)

            if 'latent_value_interact_decode_neuron' in analyses:
                latent_value_interact = neuro_latent[:, :, np.newaxis] * full_state_vars[:, np.newaxis, :]
                latent_value_interact = latent_value_interact.reshape([latent_value_interact.shape[0], -1])
                wrap_decoding_save(ana_model_path,'latent_value_interact', np.concatenate([neuro_latent, full_state_vars, latent_value_interact], axis=1), neuro_dt)

            if 'task_var_value_decode_neuron' in analyses:
                wrap_decoding_save(ana_model_path,'varvalue', np.concatenate([behav_var, full_state_vars], axis=1), neuro_dt)

            if 'ccgp' in analyses or 'condition_decoding' in analyses or 'mds' in analyses:
                if 'prev_trial' not in analyses:
                    logits, _ = extract_value_changes(model_scores, value_type='logit') #shape: (n_trials, )
                    cur_context = (logits < 0) * 1 # 0 or 1; logit > 0 == prefer A1 == action 0; logit < 0 == prefer A2 == action 1
                    cur_reward = np.concatenate(behav_dt['reward']) # 0 or 1
                    cur_action = np.concatenate(behav_dt['action']) # 0 or 1
                    condition = np.array([cur_context, cur_reward, cur_action]).T # (n_trials, 3)
                    neuro_dt # (n_trials, n_neurons)
                else:
                    assert 'mds' not in analyses, 'mds not supported for prev_trial'
                    logits, _ = extract_value_changes([s[1:] for s in model_scores], value_type='logit') #start from the second trial
                    cur_context = (logits < 0) * 1 # 0 or 1
                    prev_reward = np.concatenate([r[:-1] for r in behav_dt['reward']])
                    cur_action = np.concatenate([a[1:] for a in behav_dt['action']])
                    condition = np.array([cur_context, prev_reward, cur_action]).T # (n_total_trials - 1, 3)
                    n_neurons = neuro_dt.shape[-1]
                    neuro_dt = neuro_dt.reshape((len(epispode_idx), -1, n_neurons)) # n_episodes, n_trials, n_neurons
                    neuro_dt = neuro_dt[:, 1:, :].reshape((-1, n_neurons))
                assert len(condition) == len(neuro_dt)

                print('condition count', np.unique(condition, axis=0, return_counts=True))

                condition_type = condition[:, 0] * 4 + condition[:, 1] * 2 + condition[:, 2] # (n_trials, )

                all_labels = np.unique(condition_type)
                num_points = len(all_labels)
                if num_points < 8:
                    print('==================WARNING: num_points', num_points, 'too small')
                    break

                from itertools import combinations
                label_sets_1 = [x for x in combinations(all_labels, num_points // 2) if 0 in x]  # to avoid repetetion
                # [(0,1,2,3), (0,1,2,4), ...]
                label_sets_2 = [tuple([x for x in range(num_points) if not x in label_set1]) for label_set1 in
                                label_sets_1]
                # [(4,5,6,7), (3,5,6,7), ...]
                context0 = (
                    0,  # context=0, reward=0, action=0
                    1,  # context=0, reward=0, action=1
                    2,  # context=0, reward=1, action=0
                    3,  # context=0, reward=1, action=1
                )
                context0_idx = label_sets_1.index(context0)
                reward0 = (
                    0,  # context=0, reward=0, action=0
                    1,  # context=0, reward=0, action=1
                    4,  # context=1, reward=0, action=0
                    5,  # context=1, reward=0, action=1
                )
                reward0_idx = label_sets_1.index(reward0)
                action0 = (
                    0,  # context=0, reward=0, action=0
                    2,  # context=0, reward=1, action=0
                    4,  # context=1, reward=0, action=0
                    6,  # context=1, reward=1, action=0
                )
                action0_idx = label_sets_1.index(action0)

                if 'mds' in analyses:
                    assert 'prev_trial' not in analyses, 'prev_trial not supported for mds'
                    from sklearn.manifold import MDS
                    mds = MDS(n_components=3, dissimilarity='euclidean', random_state=0)
                    neuro_dt_cond_mean = np.zeros((num_points, neuro_dt.shape[1]))
                    for i in range(num_points):
                        neuro_dt_cond_mean[i] = neuro_dt[condition_type == i].mean(0)
                    mds_result = mds.fit(neuro_dt_cond_mean) # shape: (num_points, 3)
                    mds_embedding = mds_result.embedding_

                    neuro_dt_finer_cond_mean = np.zeros((3,2,2, neuro_dt.shape[1])) # context, reward, action, neuron
                    for context_value in range(3):
                        context0_thre = np.quantile(logits, 2/3) # ~1
                        context1_thre = np.quantile(logits, 1/3) # ~-1
                        print('context0_thre', context0_thre, 'context1_thre', context1_thre)
                        if context_value == 0:
                            cond_idx = logits >= context0_thre
                        elif context_value == 1:
                            cond_idx = (logits >= context1_thre) & (logits < context0_thre)
                        elif context_value == 2:
                            cond_idx = logits < context1_thre
                        print('context_value', context_value, 'cond_idx', cond_idx.sum())
                        for reward_value in range(2):
                            cond_idx1 = cond_idx & (cur_reward == reward_value)
                            print('reward_value', reward_value, 'cond_idx', cond_idx1.sum())
                            for action_value in range(2):
                                cond_idx2 = cond_idx1 & (cur_action == action_value)
                                print('action_value', action_value, 'cond_idx', cond_idx2.sum())
                                neuro_dt_finer_cond_mean[context_value, reward_value, action_value] = neuro_dt[cond_idx2].mean(0)
                    mds_finer_result = mds.fit(neuro_dt_finer_cond_mean.reshape([12, -1]))
                    mds_finer_embedding = mds_finer_result.embedding_.reshape([3,2,2,3]) # context, reward, action, 3

                    mds_trial_result = mds.fit(neuro_dt) # shape: (n_trials, 3)
                    mds_trial_embedding = mds_trial_result.embedding_

                    # pca
                    pca = PCA(n_components=3)
                    pca_trial_embedding = pca.fit_transform(neuro_dt)

                    os.makedirs(ANA_SAVE_PATH / exp_folder / 'mds', exist_ok=True)
                    joblib.dump({
                        'mds_embedding': mds_embedding,
                        'mds_finer_embedding': mds_finer_embedding,
                        'mds_trial_embedding': mds_trial_embedding,
                        'pca_trial_embedding': pca_trial_embedding,
                        'logits': logits,
                        'condition': condition,
                        'condition_type': condition_type,
                        'context0': context0,
                        'reward0': reward0,
                        'action0': action0,

                    }, ANA_SAVE_PATH / exp_folder / 'mds' / f'{session_name}_{fname}_mds.pkl')

                training_ratio = 0.75
                from sklearn.linear_model import RidgeClassifierCV
                for analysis in set(analyses).intersection({'ccgp', 'condition_decoding'}):
                    all_score = []
                    for label_set1, label_set2 in zip(label_sets_1, label_sets_2):
                        label_sets_training_1 = list(combinations(label_set1, int(num_points // 2 * training_ratio))) # [(0,1,2), (0,1,3), ...]
                        label_sets_training_2 = list(combinations(label_set2, int(num_points // 2 * training_ratio))) # [(4,5,6), (4,5,7), ...]

                        dichotomy_score = []
                        for train_p1 in label_sets_training_1:
                            for train_p2 in label_sets_training_2:
                                train_p1 = list(train_p1) # [0,1,2]
                                train_p2 = list(train_p2) # [4,5,6]
                                test_p1 = [x for x in label_set1 if x not in train_p1] # [3]
                                test_p2 = [x for x in label_set2 if x not in train_p2] # [7]
                                train_p1_idx = np.where(np.isin(condition_type, train_p1))[0]
                                train_p2_idx = np.where(np.isin(condition_type, train_p2))[0]
                                test_p1_idx = np.where(np.isin(condition_type, test_p1))[0]
                                test_p2_idx = np.where(np.isin(condition_type, test_p2))[0]

                                if analysis == 'ccgp':
                                    # index already found
                                    pass
                                elif analysis == 'condition_decoding':
                                    # resample the index but with the same proportion
                                    p1_idx = np.concatenate([train_p1_idx, test_p1_idx], axis=0)
                                    p2_idx = np.concatenate([train_p2_idx, test_p2_idx], axis=0)
                                    train_p1_idx = np.random.choice(p1_idx, size=len(train_p1_idx), replace=False)
                                    train_p2_idx = np.random.choice(p2_idx, size=len(train_p2_idx), replace=False)
                                    test_p1_idx = np.array([x for x in p1_idx if x not in train_p1_idx])
                                    test_p2_idx = np.array([x for x in p2_idx if x not in train_p2_idx])

                                if 'balanced_dichotomy' in analyses:
                                    # make sure the training set is balanced
                                    def balance_idx(idx1, idx2):
                                        if len(idx1) < len(idx2):
                                            idx2 = np.random.choice(idx2, size=len(idx1), replace=False)
                                        elif len(idx1) > len(idx2):
                                            idx1 = np.random.choice(idx1, size=len(idx2), replace=False)
                                        return idx1, idx2
                                    train_p1_idx, train_p2_idx = balance_idx(train_p1_idx, train_p2_idx)
                                    test_p1_idx, test_p2_idx = balance_idx(test_p1_idx, test_p2_idx)

                                X_train = np.concatenate([neuro_dt[train_p1_idx], neuro_dt[train_p2_idx]], axis=0)
                                y_train = np.array([0] * len(train_p1_idx) + [1] * len(train_p2_idx))
                                X_test = np.concatenate([neuro_dt[test_p1_idx], neuro_dt[test_p2_idx]], axis=0)
                                y_test = np.array([0] * len(test_p1_idx) + [1] * len(test_p2_idx))

                                clf = RidgeClassifierCV(alphas=np.logspace(-6, 6, 13))
                                clf.fit(X_train, y_train)
                                score = clf.score(X_test, y_test)
                                dichotomy_score.append(score)
                        print('=======','label_set1', label_set1, 'label_set2', label_set2, 'dichotomy_score', dichotomy_score, 'mean', np.mean(dichotomy_score))
                        all_score.append(dichotomy_score)

                    all_score = np.array(all_score)
                    all_score_each_dichotomy = np.mean(all_score, axis=1)
                    top3_index = np.argsort(all_score_each_dichotomy)[-3:][::-1]
                    print('===all_score', all_score_each_dichotomy, 'top 3 index', top3_index, 'top 3 score', all_score_each_dichotomy[top3_index], 'top 3 dichotomy', [label_sets_1[i] for i in top3_index])
                    print('context score', all_score_each_dichotomy[context0_idx], 'reward score', all_score_each_dichotomy[reward0_idx], 'action score', all_score_each_dichotomy[action0_idx])
                    result_dict = {
                        'label_sets_1': label_sets_1,
                        'label_sets_2': label_sets_2,
                        'context0': context0,
                        'reward0': reward0,
                        'action0': action0,
                        'context0_idx': context0_idx,
                        'reward0_idx': reward0_idx,
                        'action0_idx': action0_idx,
                        'all_score': all_score,
                        'all_score_each_dichotomy': all_score_each_dichotomy,
                        'top3_index': top3_index,
                        'top3_score': all_score_each_dichotomy[top3_index],
                        'top3_dichotomy': [label_sets_1[i] for i in top3_index],
                        'context_score': all_score_each_dichotomy[context0_idx],
                        'reward_score': all_score_each_dichotomy[reward0_idx],
                        'action_score': all_score_each_dichotomy[action0_idx],
                    }
                    os.makedirs(ANA_SAVE_PATH / exp_folder / analysis, exist_ok=True)
                    joblib.dump(result_dict, ANA_SAVE_PATH / exp_folder / analysis / f'{session_name}_{fname}_{analysis}.pkl')
                break # only run one model



            # neural activity predict state variables
            y_list = []
            if 'decode_logit' in analyses:
                logits, _ = extract_value_changes(model_scores, value_type='logit')
                y_list += [('logits', logits)]
            if 'decode_logit_abs' in analyses and row['hidden_dim'] == 2:
                logits, _ = extract_value_changes(model_scores, value_type='logit')
                logits_abs = np.abs(logits)
                y_list += [('logits_abs', logits_abs)]
            if 'decode_logit_ortho' in analyses and row['hidden_dim'] == 2:
                logits, _ = extract_value_changes(model_scores, value_type='logit')
                value1, _ = extract_value_changes(model_internal, value_type=0)
                value2, _ = extract_value_changes(model_internal, value_type=1)
                # LR: logits ~ value1 + value2
                lr = LinearRegression()
                lr.fit(np.stack([value1, value2], axis=1), logits)
                a, b = lr.coef_
                cov_v1 = np.var(value1)
                cov_v2 = np.var(value2)
                cov_v1v2 = np.cov(value1, value2)[0, 1]
                d = 1
                c = -(b*cov_v2 + a*cov_v1v2) / (a*cov_v1 + b*cov_v1v2)
                logits_ortho = c * value1 + d * value2
                assert np.abs(np.corrcoef(logits_ortho, logits)[0, 1]) < 1e-3
                y_list += [('logits_ortho', logits_ortho)]
            if 'decode_logit_change' in analyses:
                _, logits_change = extract_value_changes(model_scores, value_type='logit')
                y_list += [('logits_change', logits_change)]
            if 'decode_chosen_value' in analyses and row['hidden_dim'] == 2:
                # only for cognitive models and RNNs with 2 hidden units and readout_FC=False
                chosen_values, _ = extract_value_changes(model_internal, value_type='chosen_value', action=behav_dt['action'])
                y_list += [('chosen_values', chosen_values)]
            if 'decode_value' in analyses:
                for i in range(model_internal[0].shape[1]):
                    values, _ = extract_value_changes(model_internal, value_type=i)
                    y_list += [(i, values)]

            for y_name, y in y_list:
                analysis_filename = ana_model_path / f'{session_name}_{fname}_{y_name}_R2s.pkl'
                if analysis_filename.exists() and ignore_analyzed:
                    print(analysis_filename, 'exists')
                else:
                    R2s = pca_decoding(X_PCs, y, verbose=False, max_pca_num=100)
                    joblib.dump(R2s, analysis_filename)
                    print(session_name, model_path, 'predicting', y_name, 'final R2', R2s[-1])

def run_neural_dynamics_modeling_exp(exp_folder, neuro_data_spec):

    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    # cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config), neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'neural_dynamics_modeling'
    os.makedirs(ana_path, exist_ok=True)

    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace('bin_size_', 'bs')
    for session_name in dt.behav_data_spec['session_name']:
        neuro_dt, epispode_idx, _, _ = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=3, **neuro_data_spec)
        behav_dt = dt.get_behav_data(epispode_idx, {})
        behav_var = np.transpose(
            np.array([
                np.array(behav_dt['action']), # (n_episodes, n_trials)
                np.array(behav_dt['reward']) # (n_episodes, n_trials)
            ]),  # (2, n_episodes, n_trials)
        [1, 2, 0]
        ) # (n_episodes, n_trials, 2)
        episode_num, trial_num, neuron_num = neuro_dt.shape
        results = extract_pcs(neuro_dt.reshape([episode_num * trial_num, neuron_num]))
        results['PC'] = results['PC'].reshape([episode_num, trial_num, -1])
        results['X'] = neuro_dt
        results['behav_var'] = behav_var
        joblib.dump(results, ana_path / f'{session_name}_{fname}_pca.pkl')


def compile_decoding_results(exp_folder, neuro_data_spec, extract_feature_func=None,other_fname='', filters=lambda row: True):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    use_based_on_test = False
    suffix = '_based_on_test' if use_based_on_test else ''
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / f'rnn_final_best_summary{suffix}.pkl')
    rnn_summary['model_type'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / f'cog_final_best_summary{suffix}.pkl')
    cog_summary['model_type'] = cog_summary['cog_type']

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config), neuro_data_spec=neuro_data_spec)

    ana_path = ANA_SAVE_PATH / config['dataset'] / 'decoding'
    ana_path.mkdir(exist_ok=True)
    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()]) + other_fname
    fname = fname.replace('start_time_before_event_', 'stbe').replace('end_time_after_event_', 'etae').replace(
        'bin_size_', 'bs')
    session_names = dt.behav_data_spec['session_name']
    subset_from_kept_feat_dict = {}
    feat_scale_dict = {}

    config = transform_model_format(rnn_summary.iloc[0], source='row', target='config')
    for session_name in session_names:
        _, _, kept_feat_idx, feat_scale = dt.get_neuro_data(session_name=session_name, zcore=True,
                                                   remove_nan=True, shape=2, **neuro_data_spec)
        # kept_feat_idx is the shape of orginal neural data X: True means the feature is kept
        # feat_scale is the scale of all kept features; number of kept features is the same as the number of True in kept_feat_idx
        # we have customized functions to extract subset of features, in the shape of feat_scale
        if extract_feature_func is None:
            subset_from_kept_feat = np.ones_like(feat_scale, dtype=bool) # all features are kept
        else:
            subset_from_kept_feat = extract_feature_func(kept_feat_idx, **neuro_data_spec)
        subset_from_kept_feat_dict[session_name] = subset_from_kept_feat
        feat_scale_dict[session_name] = feat_scale

        print(session_name, 'final kept feature num', np.sum(subset_from_kept_feat))

    # simpler_decoding_name = 'task_var_decode_neuron'
    # complex_decoding_name = 'varvalue' #

    # simpler_decoding_name = 'latent_decode_neuron'
    # complex_decoding_name = 'latent_value'
    
    simpler_decoding_name = 'latent_decode_neuron'
    complex_decoding_name = 'latent_value_interact'
    def compile_summary(summary, model_name_key, model_identifier_keys):
        new_rows = []
        for i, row in summary.iterrows():
            hidden_dim = row['hidden_dim']
            if hidden_dim < 1:
                continue
            if not filters(row):
                continue
            model_name = row[model_name_key]
            model_path = transform_model_format(row, source='row', target='path')
            ana_model_path = ANA_SAVE_PATH / model_path / 'decoding'
            new_row = row.copy()
            new_row['trainvaltest_loss'] = (new_row['test_loss'] * new_row['test_trial_num'] +
                                            new_row['trainval_loss'] * new_row['trainval_trial_num']
                                            ) / (new_row['test_trial_num'] + new_row['trainval_trial_num'])
            total_feat_num = 0
            for session_name in session_names:
                feat_scale = feat_scale_dict[session_name]
                simpler_decoding = joblib.load(ana_path / f'{session_name}_{fname}_{simpler_decoding_name}.pkl')
                task_var_mse = simpler_decoding['mse'] * feat_scale ** 2
                task_null_mse = simpler_decoding['null_mse'] * feat_scale ** 2
                complex_decoding = joblib.load(
                    ana_model_path / f'{session_name}_{fname}_{complex_decoding_name}.pkl')
                mse = complex_decoding['mse'] * feat_scale ** 2
                null_mse = complex_decoding['null_mse'] * feat_scale ** 2
                assert np.isclose(null_mse, task_null_mse).all()
                subset_filter = subset_from_kept_feat_dict[session_name]
                neuron_R2 = complex_decoding['r2']
                neuron_R2 = neuron_R2[subset_filter]
                # population_cpd = 1 - mse[subset_filter].sum() / task_var_mse[subset_filter].sum()
                new_row[session_name + '_sum_task_model_mse'] = mse[subset_filter].sum()
                new_row[session_name + '_sum_task_mse'] = task_var_mse[subset_filter].sum()
                new_row[session_name + '_sum_null_mse'] = null_mse[subset_filter].sum()
                new_row[session_name+'_population_cpd'] = 1 - new_row[session_name + '_sum_task_model_mse'] / new_row[session_name + '_sum_task_mse']
                new_row[session_name+'_population_R2'] = 1 - new_row[session_name + '_sum_task_model_mse'] / new_row[session_name + '_sum_null_mse']
                new_row[session_name+'_population_task_R2'] = 1 - new_row[session_name + '_sum_task_mse'] / new_row[session_name + '_sum_null_mse']
                new_row[session_name+'_mean_R2'] = np.mean(neuron_R2)
                new_row[session_name+'_R2_greater_0p1'] = np.mean(neuron_R2>0.1)
                new_row[session_name+'_pseudocell_num'] = len(neuron_R2)
                total_feat_num += len(neuron_R2)
            new_row['sum_task_mse'] = np.sum([new_row[session_name + '_sum_task_mse'] for session_name in session_names])
            new_row['sum_task_model_mse'] = np.sum([new_row[session_name + '_sum_task_model_mse'] for session_name in session_names])
            new_row['sum_null_mse'] = np.sum([new_row[session_name + '_sum_null_mse'] for session_name in session_names])
            new_row['population_cpd'] = 1 - new_row['sum_task_model_mse'] / new_row['sum_task_mse']
            new_row['population_R2'] = 1 - new_row['sum_task_model_mse'] / new_row['sum_null_mse']
            new_row['population_task_R2'] = 1 - new_row['sum_task_mse'] / new_row['sum_null_mse']
            new_row['pseudocell_num'] = total_feat_num
            new_row['mean_R2'] = np.sum([new_row[session_name+'_mean_R2'] * new_row[session_name+'_pseudocell_num']
                                         for session_name in session_names]) / total_feat_num
            new_row['R2_greater_0p1'] = np.sum([new_row[session_name+'_R2_greater_0p1']*new_row[session_name+'_pseudocell_num']
                                        for session_name in session_names]) / total_feat_num
            new_rows.append(new_row)
        new_summary = pd.DataFrame(new_rows)
        agg_dict = {
            'population_cpd': ('population_cpd', 'mean'),
            'population_R2': ('population_R2', 'mean'),
            'population_task_R2': ('population_task_R2', 'mean'),
            'pseudocell_num': ('pseudocell_num', 'mean'),
            'mean_R2': ('mean_R2', 'mean'),
            'mean_R2_max': ('mean_R2', 'max'),
            'R2_greater_0p1': ('R2_greater_0p1', 'mean'),
        }
        for session_name in session_names:
            agg_dict[session_name+'_population_cpd'] = (session_name+'_population_cpd', 'mean')
            agg_dict[session_name+'_population_R2'] = (session_name+'_population_R2', 'mean')
            agg_dict[session_name+'_pseudocell_num'] = (session_name+'_pseudocell_num', 'mean')
            agg_dict[session_name+'_mean_R2'] = (session_name+'_mean_R2', 'mean')
            agg_dict[session_name+'_R2_greater_0p1'] = (session_name+'_R2_greater_0p1', 'mean')
        perf = new_summary.groupby(model_identifier_keys, as_index=False).agg(**agg_dict)
        return new_summary, perf

    rnn_summary, rnn_perf = compile_summary(rnn_summary, 'rnn_type', ['rnn_type', 'hidden_dim', 'readout_FC'])
    cog_summary, cog_perf = compile_summary(cog_summary, 'cog_type', ['cog_type', 'hidden_dim'])
    with pd_full_print_context():
        print(rnn_perf)
        print(cog_perf)
    joblib.dump(rnn_perf, ana_exp_path / f'rnn_neuron_decoding_perf{suffix}_{simpler_decoding_name}_{complex_decoding_name}_{fname}.pkl')
    joblib.dump(rnn_summary, ana_exp_path / f'rnn_neuron_decoding_best_summary{suffix}_{simpler_decoding_name}_{complex_decoding_name}_{fname}.pkl')
    joblib.dump(cog_perf, ana_exp_path / f'cog_neuron_decoding_perf{suffix}_{simpler_decoding_name}_{complex_decoding_name}_{fname}.pkl')
    joblib.dump(cog_summary, ana_exp_path / f'cog_neuron_decoding_best_summary{suffix}_{simpler_decoding_name}_{complex_decoding_name}_{fname}.pkl')

    rnn_summary.to_csv(ana_exp_path / f'rnn_neuron_decoding_best_summary{suffix}_{simpler_decoding_name}_{complex_decoding_name}_{fname}.csv', index=False)