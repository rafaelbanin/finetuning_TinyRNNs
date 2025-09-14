import pandas as pd
from utils import set_os_path_auto, highlighted_print
from .analyzing import *
from .analyzing_check import check_missing_models
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

def select_final_rnn_perf(exp_folder, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss', return_dim_est=False, combine_model_then_select=None,sort_keys=None,
                          include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False):
    if additional_keys is None:
        additional_keys = {}
    filter_dict = {'agent_type': 'RNN'}#################################################################
    model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC','finetune'] + additional_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', []) # best model from inner fold and average over outer fold
    compete_from_keys = ['l1_weight', 'seed'] + additional_keys.setdefault('compete_from_keys', []) # the keys to pick the best model instance
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict,
                                            inner_fold_perf_key=inner_fold_perf_key, filter_dict_for_summary=combine_model_then_select,include_acc=include_acc, include_acc_filter=include_acc_filter,
                                            dataset_loading_every_time=dataset_loading_every_time)
    if combine_model_then_select is not None:
        new_rnn_type = '+'.join(combine_model_then_select['rnn_type'])
        perf['rnn_type'] = new_rnn_type
        summary['rnn_type'] = new_rnn_type
   
 
    if sort_keys is not None:
        #assert sort_keys[0] == 'block' # only support block for now
        idx = 0
        for k in sort_keys:
            perf.insert(idx, k, perf.pop(k))
            summary.insert(idx, k, summary.pop(k))
            idx += 1
        # change none to -1
        if 'block' in perf.columns:
            perf['block'] = perf['block'].apply(lambda x: -1 if x == 'none' else x)
            summary['block'] = summary['block'].apply(lambda x: -1 if x == 'none' else x)
        perf = perf.sort_values(by=sort_keys)
        summary = summary.sort_values(by=sort_keys)


    perf.insert(1, 'test_loss', perf.pop('test_loss'))
    perf.insert(2, 'trainval_loss', perf.pop('trainval_loss'))
    perf.insert(3, 'train_loss', perf.pop('train_loss'))
    perf.insert(4, 'val_loss', perf.pop('val_loss'))
    if include_acc:
        perf.insert(5, 'test_acc', perf.pop('test_acc'))
        perf.insert(6, 'trainval_acc', perf.pop('trainval_acc'))
        perf.insert(7, 'train_acc', perf.pop('train_acc'))
        perf.insert(8, 'val_acc', perf.pop('val_acc'))
    if verbose:
        with pd_full_print_context():
            print(perf)
    from scipy.stats import ttest_rel
    def estimate_dimensionality(perf, verbose=True, ttest=True):
        perf_gru = perf[perf['readout_FC'] == True].reset_index(drop=True)
        # given a hidden dimension, select the best model with the lowest test loss (from GRU/SGRU/PNR1)
        perf_gru = select_best_models_by_keys(perf_gru, group_by_keys=['hidden_dim'], perf_key='test_loss', select_func='min')

        perf_gru['less_pvalue'] = [np.zeros(len(perf_gru)) for _ in range(len(perf_gru))]
        perf_gru['less_than_former'] = [1 for _ in range(len(perf_gru))]
        for i, i_row in perf_gru.iterrows():
            L = i_row['agg_outer_fold']
            assert all(L[i] < L[i+1] for i in range(len(L) - 1)), L # make sure the outer fold number is in order
            for j, j_row in perf_gru.iterrows():
                if ttest:
                    pvalue = ttest_rel(i_row['agg_test_loss'], j_row['agg_test_loss'], alternative='less').pvalue
                else:
                    pvalue = 1 - (np.mean(i_row['agg_test_loss']) < np.mean(j_row['agg_test_loss'])) # 0 if i is less than j
                perf_gru.loc[i, 'less_pvalue'][j] = pvalue
                if j<i and pvalue > 0.05:
                    perf_gru.loc[i, 'less_than_former'] = 0
        if verbose:
            with pd_full_print_context():
                print('Estimated dimensionality:')
                if 'rnn_type' in perf_gru.columns:
                    col_list = ['block'] if 'block' in perf_gru.columns else []
                    col_list += ['rnn_type', 'hidden_dim', 'less_than_former', 'test_loss', 'less_pvalue']
                    print(perf_gru[col_list])
        return perf_gru


    if 'block' not in perf.columns: # when the same folder has one subject
        perf_gru = estimate_dimensionality(perf, verbose=verbose)
    else:
        perf_gru = []
        for block in perf['block'].unique():
            perf_gru.append(estimate_dimensionality(perf[perf['block'] == block], verbose=verbose, ttest=False))
        perf_gru = pd.concat(perf_gru, axis=0, join='outer')
    if return_dim_est:
        return perf, summary, perf_gru
    return perf, summary


def select_final_cog_perf(exp_folder, agent_type, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss',cog_hidden_dim=None, sort_keys=None,
                          include_acc=False, include_acc_filter=lambda row: True,dataset_loading_every_time=False):
    """
    Select the final performance of the cognitive model.

    Args:
        exp_folder (str): The folder of the experiment.
        agent_type (str): The type of the cognitive model. PRLCog, RTSCog, NTSCog...
    """
    if cog_hidden_dim is None:
        cog_hidden_dim = {
            'BAS': 0, 'MB0s': 1, 'MFs': 1, 'LS0': 1, 'LS1': 1, 'MB0se': 1, 'MBsvlr': 1,'MBsflr': 1,
            'MB0': 2,'MBf': 2, 'MB1': 2, 'MXs': 2, 'MB0md': 2, 'MB0mdnb': 2, 'MB0m': 2, 'Q(1)': 2, 'MBsah': 2,
            'MB0off': 2, 'MF0sp': 2,'MFDp':4,
            'MB0p': 3, 'MB1p': 3,
            'Q(0)': 4, 'RC': 8,
            'MF_MB_bs_rb_ck': 6,#
            'MF_bs_rb_ck': 4,#
            'MB_bs_rb_ck': 4,#
            'MF_MB_dec_bs_rb_ck': 6,#
            'MF_dec_bs_rb_ck': 4,#
            'MB_dec_bs_rb_ck': 4,#
            'MF_MB_vdec_bs_rb_ck': 6,#
            'MF_MB_bs_rb_ec': 6 + 1, #
            'MF_MB_vdec_bs_rb_ec': 6 + 1, #
            'MF_MB_dec_bs_rb_ec': 6 + 1,#
            'MF_MB_dec_bs_rb_mc': 6 + 2,#
            'MF_MB_dec_bs_rb_ec_mc': 6 + 3,#
            'MFmoMF_MB_dec_bs_rb_ec_mc': 6 + 4 + 3, #
            'MFmoMF_dec_bs_rb_ec_mc': 4 + 4 + 3,#
            'MB_dec_bs_rb_ec_mc': 4 + 3,
            'AC3': 27,
            'AC4': 27,
        }
    if additional_keys is None:
        additional_keys = {}
    model_identifier_keys = ['cog_type'] + additional_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', []) # best model from inner fold and average over outer fold
    compete_from_keys = ['seed'] + additional_keys.setdefault('compete_from_keys', []) # the keys to pick the best model instance
    filter_dict = {'agent_type': agent_type}
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict, inner_fold_perf_key=inner_fold_perf_key,
                                            include_acc=include_acc,  include_acc_filter=include_acc_filter,
                                            dataset_loading_every_time=dataset_loading_every_time)
    perf['hidden_dim'] = perf.apply(lambda row: cog_hidden_dim[row['cog_type']], axis=1)
    summary['hidden_dim'] = summary.apply(lambda row: cog_hidden_dim[row['cog_type']], axis=1)
    if sort_keys is not None:
        key0 = sort_keys[0]
        assert key0 in ['block','subject_number']# only support block for now
        idx = 0
        for k in sort_keys:
            perf.insert(idx, k, perf.pop(k))
            summary.insert(idx, k, summary.pop(k))
            idx += 1
        # change none to -1
        perf[key0] = perf[key0].apply(lambda x: -1 if x == 'none' else x)
        summary[key0] = summary[key0].apply(lambda x: -1 if x == 'none' else x)
        perf = perf.sort_values(by=sort_keys)
        summary = summary.sort_values(by=sort_keys)
    if verbose:
        with pd_full_print_context():
            print(perf)
    return perf, summary


def find_best_models_for_exp(exp_folder, cog_agent_type, additional_rnn_keys=None, additional_cog_keys=None,
                             rnn_sort_keys=None, cog_sort_keys=None, has_rnn=True, has_cog=True, cog_hidden_dim=None, return_dim_est=True, check_missing=True,
                             include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False):
    goto_root_dir.run()
    if check_missing:
        check_missing_models(exp_folder)

    # for inner_fold_perf_key in ['trainval_loss', #'test_loss']:
    inner_fold_perf_key = 'trainval_loss'
    if inner_fold_perf_key == 'trainval_loss':
        fname = ''
    else:
        fname = '_based_on_test'
    print('========Select best models based on inner_fold_perf_key:', inner_fold_perf_key,'exp_folder:', exp_folder)
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)
    if has_rnn:
        #for combine_model_then_select in [{'rnn_type': ['GRU', 'SGRU'], 'finetune':[True, False, 'none']}]:###
        for combine_model_then_select in [None]:#, {'rnn_type': ['GRU', 'SGRU'], 'finetune':[True, False, 'none']}]:###
            temp = select_final_rnn_perf(exp_folder, additional_keys=additional_rnn_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                                        return_dim_est=return_dim_est,
                                         combine_model_then_select=combine_model_then_select,sort_keys=rnn_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                         dataset_loading_every_time=dataset_loading_every_time)
            if return_dim_est:
                rnn_perf, rnn_summary, perf_est_dim = temp
            else:
                rnn_perf, rnn_summary = temp
                perf_est_dim = None
            fname_temp = fname
            if combine_model_then_select is not None:
                fname_temp += '_combine_then_select'
            joblib.dump(rnn_perf, ana_exp_path / f'rnn_final_perf{fname_temp}.pkl')
            joblib.dump(rnn_summary, ana_exp_path / f'rnn_final_best_summary{fname_temp}.pkl')
            joblib.dump(perf_est_dim, ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.pkl')
            # also save csv
            rnn_perf.to_csv(ana_exp_path / f'rnn_final_perf{fname_temp}.csv')
            rnn_summary.to_csv(ana_exp_path / f'rnn_final_best_summary{fname_temp}.csv')
            if perf_est_dim is not None:
                perf_est_dim.to_csv(ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.csv')
    else:
        pass
        # below will overwrite the previous results
        # empty_pd = pd.DataFrame()
        # joblib.dump(empty_pd, ana_exp_path / f'rnn_final_perf{fname}.pkl')
        # joblib.dump(empty_pd, ana_exp_path / f'rnn_final_best_summary{fname}.pkl')

    if has_cog:
        cog_perf, cog_summary = select_final_cog_perf(exp_folder, cog_agent_type, additional_keys=additional_cog_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                      cog_hidden_dim=cog_hidden_dim, sort_keys=cog_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                                      dataset_loading_every_time=dataset_loading_every_time)
        joblib.dump(cog_perf, ana_exp_path / f'cog_final_perf{fname}.pkl')
        joblib.dump(cog_summary, ana_exp_path / f'cog_final_best_summary{fname}.pkl')
    else:
        pass
        # below will overwrite the previous results
        # empty_pd = pd.DataFrame()
        # joblib.dump(empty_pd, ana_exp_path / f'cog_final_perf{fname}.pkl')
        # joblib.dump(empty_pd, ana_exp_path / f'cog_final_best_summary{fname}.pkl')

def select_best_models_spec_loss_for_exp(exp_folder, additional_rnn_keys=None, model_filter=None,
                                         ):
    '''
    Assume that the final_perf.pkl has been saved in the exp_folder.
    Based on the loss of the specified trials, select the best models.
    Save to the exp_folder.

    '''
    goto_root_dir.run()
    loss_name = 'this_animal_val_loss'
    loss_trial_num_name = 'this_animal_val_trial_num'
    # assert len(loss_name) and loss_based_block is not None
    print('========Select best models based on :', loss_name,'exp_folder:', exp_folder)
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)

    rnn_summary = joblib.load(ana_exp_path / f'rnn_final_best_summary.pkl')
    if model_filter is not None:
        for k,v in model_filter.items():
            if k in rnn_summary.columns:
                rnn_summary = rnn_summary[rnn_summary[k]==v]
                print(f'filter rnn_summary by {k}={v}')
            else:
                print(f'Warning: {k} not in rnn_summary.columns')

    # with pd_full_print_context():
    #     print(rnn_final_best_summary)
    summary = []
    for i_row, row in rnn_summary.iterrows():
        model_path = row['model_path']
        print(model_path)
        behav_loss = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')['behav_loss']
        ## below is specific to the rat 49, to be changed later
        val_index = row['val_index']
        print('val_index:', val_index)
        behav_loss = np.concatenate([behav_loss[i] for i in val_index if i < 143], axis=0)
        print('behav_loss shape',behav_loss.shape)
        row[loss_trial_num_name] = behav_loss.shape[0]
        row[loss_name] = behav_loss.mean()
        summary.append(row)
    summary = pd.DataFrame(summary)
    additional_keys = additional_rnn_keys if additional_rnn_keys is not None else {}
    model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC'] + additional_keys.setdefault('model_identifier_keys', [])  # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys',[])  # best model from inner fold and average over outer fold
    compete_from_keys = ['l1_weight', 'seed'] + additional_keys.setdefault('compete_from_keys',[])  # the keys to pick the best model instance

    summary = select_best_models_by_keys(summary, group_by_keys=model_identifier_keys+cv_keys, perf_key=loss_name, select_func='min')
    summary.insert(0, loss_trial_num_name, summary.pop(loss_trial_num_name))
    summary.insert(1, loss_name, summary.pop(loss_name))
    with pd_full_print_context():
        print(summary)
    joblib.dump(summary, ana_exp_path / f'rnn_final_best_summary_{loss_name}.pkl')

def combine_exps(exp_folders, additional_rnn_keys={}, rnn_filter=None, cog_filter=None,
                          additional_cog_keys={}, additional_rnn_agg={}, additional_cog_agg={}, has_rnn=True, has_cog=True,
                          rnn_lambda_filter=None,cog_lambda_filter=None, lambda_filter_name='', combine_df_type='perf'):
    """
    Collect models from exp_folders, either perf or best_summary.
    """
    assert combine_df_type in ['perf', 'best_summary']
    rnn_perf_list = []
    cog_perf_list = []
    rnn_filter = rnn_filter or {}
    cog_filter = cog_filter or {}
    for exp_folder in exp_folders:
        ana_exp_path = ANA_SAVE_PATH / exp_folder
        if has_cog:
            path = ana_exp_path / f'cog_final_{combine_df_type}.pkl'
            if os.path.exists(path):
                cog_perf = joblib.load(path)
                if len(cog_perf) == 0:
                    print(f'Warning: empty cog_final_{combine_df_type}.pkl found in', exp_folder)
                    continue
                print(exp_folder, 'len(cog_perf)', len(cog_perf))
                for k, v in cog_filter.items():
                    cog_perf = cog_perf[cog_perf[k] == v]
                    print(f'filter cog_perf by {k}={v}',exp_folder,'len(cog_perf)', len(cog_perf))
                cog_perf_list.append(cog_perf)
            else:
                print(f'Warning: no cog_final_{combine_df_type}.pkl found in', exp_folder)
        if has_rnn:
            path = ana_exp_path / f'rnn_final_{combine_df_type}.pkl'
            if os.path.exists(path):
                rnn_perf = joblib.load(path)
                rnn_perf['exp_folder'] = exp_folder
                print(exp_folder, 'len(rnn_perf)', len(rnn_perf))
                for k, v in rnn_filter.items():
                    rnn_perf = rnn_perf[rnn_perf[k] == v]
                    print(exp_folder, f'filter rnn_perf by {k}={v}','len(rnn_perf)', len(rnn_perf))
                rnn_perf_list.append(rnn_perf)
            else:
                print(f'Warning: no rnn_final_{combine_df_type}.pkl found in', exp_folder)
    rnn_perf = pd.concat(rnn_perf_list) if len(rnn_perf_list) else pd.DataFrame()
    cog_perf = pd.concat(cog_perf_list) if len(cog_perf_list) else pd.DataFrame()
    combined_perf = pd.concat([rnn_perf, cog_perf], axis=0, join='outer')
    if rnn_lambda_filter is not None:
        rnn_perf = rnn_perf[rnn_lambda_filter(rnn_perf, combined_perf)] # lambda_filter: operated_dt, ref_dt
    if cog_lambda_filter is not None:
        cog_perf = cog_perf[cog_lambda_filter(cog_perf, combined_perf)]
        assert len(lambda_filter_name) > 0
    return rnn_perf, cog_perf


def compile_perf_for_exps(exp_folders, compile_exp_folder, additional_rnn_keys={}, rnn_filter=None, cog_filter=None,
                          additional_cog_keys={}, additional_rnn_agg={}, additional_cog_agg={}, has_rnn=True, has_cog=True,
                          lambda_filter=None, rnn_lambda_filter=None, cog_lambda_filter=None, lambda_filter_name='',
                          ):
    """
    Collect models from exp_folders, and compute the subject-average performance.
    """
    if lambda_filter is not None:
        assert rnn_lambda_filter is None and cog_lambda_filter is None
        rnn_lambda_filter = lambda_filter
        cog_lambda_filter = lambda_filter

    rnn_perf, cog_perf = combine_exps(exp_folders, additional_rnn_keys, rnn_filter, cog_filter,
                          additional_cog_keys, additional_rnn_agg, additional_cog_agg, has_rnn, has_cog,
                          rnn_lambda_filter, cog_lambda_filter, lambda_filter_name, combine_df_type='perf')
    ana_exp_path = ANA_SAVE_PATH / compile_exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)

    perf_agg_dict = dict(
            test_loss=('test_loss','mean'),
            test_loss_outer_inner_sem=('test_loss_outer_inner_sem','mean'),
            test_loss_sub_sem=('test_loss','sem'),
            sub_count=('test_loss','count'),
            test_trial_num_total=('test_trial_num','sum'),
    )
    if has_rnn and 'test_acc' in rnn_perf.columns or has_cog and 'test_acc' in cog_perf.columns:
        perf_agg_dict.update(dict(
            test_acc=('test_acc','mean'),
            test_acc_sub_sem=('test_acc','sem'),
            test_acc_outer_inner_sem=('test_acc_outer_inner_sem','mean'),
        ))
    if has_rnn:
        print('len(rnn_perf)', len(rnn_perf))
        model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC'] + additional_rnn_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        [rnn_perf[k].fillna('unspecified', inplace=True) for k in model_identifier_keys if k in rnn_perf.columns]
        rnn_perf = rnn_perf.groupby(model_identifier_keys, as_index=False).agg(
            **perf_agg_dict,
            **additional_rnn_agg,
        )
        highlighted_print(rnn_perf)
        joblib.dump(rnn_perf, ana_exp_path / f'rnn_final_perf{lambda_filter_name}.pkl')

    if has_cog:
        print('len(cog_perf)', len(cog_perf))
        model_identifier_keys = ['cog_type'] + additional_cog_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
        [cog_perf[k].fillna('unspecified', inplace=True) for k in model_identifier_keys if k in cog_perf.columns]
        cog_perf = cog_perf.groupby(model_identifier_keys, as_index=False).agg(
            hidden_dim=('hidden_dim','max'),
            **perf_agg_dict,
            **additional_cog_agg,
        )
        highlighted_print(cog_perf)
        joblib.dump(cog_perf, ana_exp_path / f'cog_final_perf{lambda_filter_name}.pkl')


    L = []
    if has_rnn:
        def get_dim_from_perf(perf_est_dim):
            perf_est_dim = perf_est_dim[perf_est_dim['less_than_former'] == 1]
            dim = perf_est_dim['hidden_dim'].max()
            return dim


        for exp_folder in exp_folders:
            if not os.path.exists(ANA_SAVE_PATH / exp_folder / f'rnn_final_perf_est_dim.pkl'):
                print('Warning: no rnn_final_perf_est_dim.pkl found in', exp_folder)
                continue
            perf_est_dim = joblib.load(ANA_SAVE_PATH / exp_folder / 'rnn_final_perf_est_dim.pkl')
            if perf_est_dim is None or len(perf_est_dim) == 0:
                print('Warning: no rnn_final_perf_est_dim.pkl found in', exp_folder)
                continue
            # for k, v in rnn_filter.items():
            #     perf_est_dim = perf_est_dim[perf_est_dim[k] == v]
            # if len(perf_est_dim) == 0:
            #     print('Warning: after filtering, no models in rnn_final_perf_est_dim.pkl found in', exp_folder)
            #     continue
            if 'block' not in perf_est_dim.columns:
                dim = get_dim_from_perf(perf_est_dim)
                L.append({'exp_folder': exp_folder, 'dimension': dim})
            else:
                for block in perf_est_dim['block'].unique():
                    dim = get_dim_from_perf(perf_est_dim[perf_est_dim['block'] == block])
                    L.append({'exp_folder': exp_folder, 'block': block, 'dimension': dim})
        df = pd.DataFrame(L)
        with pd_full_print_context():
            print(df)

        joblib.dump(df, ana_exp_path / f'rnn_final_perf_est_dim{lambda_filter_name}.pkl')





def select_final_rnn_perf_f(exp_folder, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss', return_dim_est=False, combine_model_then_select=None,sort_keys=None,
                          include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False,model_based = None):
    if additional_keys is None:
        additional_keys = {}
    filter_dict = {'agent_type': 'RNN'}#################################################################
    model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC','model_based'] + additional_keys.setdefault('model_identifier_keys', []) # the keys to uniquely identify the model
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', []) # best model from inner fold and average over outer fold
    compete_from_keys = ['l1_weight', 'seed'] + additional_keys.setdefault('compete_from_keys', []) # the keys to pick the best model instance
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict,
                                            inner_fold_perf_key=inner_fold_perf_key, filter_dict_for_summary=combine_model_then_select,include_acc=include_acc, include_acc_filter=include_acc_filter,
                                            dataset_loading_every_time=dataset_loading_every_time)
    if combine_model_then_select is not None:
        new_rnn_type = '+'.join(combine_model_then_select['model_based'])
        perf['model_based'] = new_rnn_type
        summary['rnn_type'] = new_rnn_type
   
 
    if sort_keys is not None:
        #assert sort_keys[0] == 'block' # only support block for now
        idx = 0
        for k in sort_keys:
            perf.insert(idx, k, perf.pop(k))
            summary.insert(idx, k, summary.pop(k))
            idx += 1
        # change none to -1
        if 'block' in perf.columns:
            perf['block'] = perf['block'].apply(lambda x: -1 if x == 'none' else x)
            summary['block'] = summary['block'].apply(lambda x: -1 if x == 'none' else x)
        perf = perf.sort_values(by=sort_keys)
        summary = summary.sort_values(by=sort_keys)


    perf.insert(1, 'test_loss', perf.pop('test_loss'))
    perf.insert(2, 'trainval_loss', perf.pop('trainval_loss'))
    perf.insert(3, 'train_loss', perf.pop('train_loss'))
    perf.insert(4, 'val_loss', perf.pop('val_loss'))
    if include_acc:
        perf.insert(5, 'test_acc', perf.pop('test_acc'))
        perf.insert(6, 'trainval_acc', perf.pop('trainval_acc'))
        perf.insert(7, 'train_acc', perf.pop('train_acc'))
        perf.insert(8, 'val_acc', perf.pop('val_acc'))
    if verbose:
        with pd_full_print_context():
            print(perf)
    from scipy.stats import ttest_rel
    def estimate_dimensionality(perf, verbose=True, ttest=True):
        perf_gru = perf[perf['readout_FC'] == True].reset_index(drop=True)
        # given a hidden dimension, select the best model with the lowest test loss (from GRU/SGRU/PNR1)
        perf_gru = select_best_models_by_keys(perf_gru, group_by_keys=['hidden_dim'], perf_key='test_loss', select_func='min')

        perf_gru['less_pvalue'] = [np.zeros(len(perf_gru)) for _ in range(len(perf_gru))]
        perf_gru['less_than_former'] = [1 for _ in range(len(perf_gru))]
        for i, i_row in perf_gru.iterrows():
            L = i_row['agg_outer_fold']
            assert all(L[i] < L[i+1] for i in range(len(L) - 1)), L # make sure the outer fold number is in order
            for j, j_row in perf_gru.iterrows():
                if ttest:
                    pvalue = ttest_rel(i_row['agg_test_loss'], j_row['agg_test_loss'], alternative='less').pvalue
                else:
                    pvalue = 1 - (np.mean(i_row['agg_test_loss']) < np.mean(j_row['agg_test_loss'])) # 0 if i is less than j
                perf_gru.loc[i, 'less_pvalue'][j] = pvalue
                if j<i and pvalue > 0.05:
                    perf_gru.loc[i, 'less_than_former'] = 0
        if verbose:
            with pd_full_print_context():
                print('Estimated dimensionality:')
                if 'rnn_type' in perf_gru.columns:
                    col_list = ['block'] if 'block' in perf_gru.columns else []
                    col_list += ['rnn_type', 'hidden_dim', 'less_than_former', 'test_loss', 'less_pvalue']
                    print(perf_gru[col_list])
        return perf_gru


    if 'block' not in perf.columns: # when the same folder has one subject
        perf_gru = estimate_dimensionality(perf, verbose=verbose)
    else:
        perf_gru = []
        for block in perf['block'].unique():
            perf_gru.append(estimate_dimensionality(perf[perf['block'] == block], verbose=verbose, ttest=False))
        perf_gru = pd.concat(perf_gru, axis=0, join='outer')
    if return_dim_est:
        return perf, summary, perf_gru
    return perf, summary



def find_best_models_for_exp_finetuned(exp_folder, cog_agent_type, additional_rnn_keys=None, additional_cog_keys=None,
                             rnn_sort_keys=None, cog_sort_keys=None, has_rnn=True, has_cog=True, cog_hidden_dim=None, return_dim_est=True, check_missing=True,
                             include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False, model_based = None):
    goto_root_dir.run()
    if check_missing:
        check_missing_models(exp_folder)

    # for inner_fold_perf_key in ['trainval_loss', #'test_loss']:
    inner_fold_perf_key = 'trainval_loss'
    if inner_fold_perf_key == 'trainval_loss':
        fname = ''
    else:
        fname = '_based_on_test'
    print('========Select best models based on inner_fold_perf_key:', inner_fold_perf_key,'exp_folder:', exp_folder)
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)
    if has_rnn:
        #for combine_model_then_select in [{'rnn_type': ['GRU', 'SGRU'], 'finetune':[True, False, 'none']}]:###
        for combine_model_then_select in [None]:###
            temp = select_final_rnn_perf_f(exp_folder, additional_keys=additional_rnn_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                                        return_dim_est=return_dim_est,
                                         combine_model_then_select=combine_model_then_select,sort_keys=rnn_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                         dataset_loading_every_time=dataset_loading_every_time, model_based = model_based)
            if return_dim_est:
                rnn_perf, rnn_summary, perf_est_dim = temp
            else:
                rnn_perf, rnn_summary = temp
                perf_est_dim = None
            fname_temp = fname
            if combine_model_then_select is not None:
                fname_temp += '_combine_then_select'
            joblib.dump(rnn_perf, ana_exp_path / f'rnn_final_perf{fname_temp}.pkl')
            joblib.dump(rnn_summary, ana_exp_path / f'rnn_final_best_summary{fname_temp}.pkl')
            joblib.dump(perf_est_dim, ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.pkl')
            # also save csv
            rnn_perf.to_csv(ana_exp_path / f'rnn_final_perf{fname_temp}.csv')
            rnn_summary.to_csv(ana_exp_path / f'rnn_final_best_summary{fname_temp}.csv')
            if perf_est_dim is not None:
                perf_est_dim.to_csv(ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.csv')
    else:
        pass
        # below will overwrite the previous results
        # empty_pd = pd.DataFrame()
        # joblib.dump(empty_pd, ana_exp_path / f'rnn_final_perf{fname}.pkl')
        # joblib.dump(empty_pd, ana_exp_path / f'rnn_final_best_summary{fname}.pkl')

    if has_cog:
        cog_perf, cog_summary = select_final_cog_perf(exp_folder, cog_agent_type, additional_keys=additional_cog_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                      cog_hidden_dim=cog_hidden_dim, sort_keys=cog_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                                      dataset_loading_every_time=dataset_loading_every_time)
        joblib.dump(cog_perf, ana_exp_path / f'cog_final_perf{fname}.pkl')
        joblib.dump(cog_summary, ana_exp_path / f'cog_final_best_summary{fname}.pkl')
    else:
        pass
        # below will overwrite the previous results
        # empty_pd = pd.DataFrame()
        # joblib.dump(empty_pd, ana_exp_path / f'cog_final_perf{fname}.pkl')
        # joblib.dump(empty_pd, ana_exp_path / f'cog_final_best_summary{fname}.pkl')





def find_final_performance(exp_folder, model_based_filter_list):
    # Esta função atua como um wrapper para o pipeline de análise
    # para obter a performance final agregada.
    model_identifier_keys = ['model_based', 'trainval_percent']
    cv_keys = ['outer_fold', 'inner_fold']
    compete_from_keys = ['seed']
    filter_dict = {'agent_type': 'RNN'}
    
    # O filtro `model_based` é aplicado aqui
    filter_dict_for_summary = {'model_based': model_based_filter_list}
    
    perf, _ = select_final_agent_perf(
        exp_folder, 
        model_identifier_keys=model_identifier_keys, 
        cv_keys=cv_keys, 
        compete_from_keys=compete_from_keys, 
        filter_dict=filter_dict,
        filter_dict_for_summary=filter_dict_for_summary
    )
    return perf



def analyze_finetuned_models_full(exp_folder, models):
    """
    Analyzes the results for finetuned models based on the
    `find_best_models_for_exp_finetuned_spec` function.

    Args:
        exp_folder (str): The name of the experiment folder.
    """

    print(f"Starting analysis for experiment folder: {exp_folder}")
    find_best_models_for_exp_finetuned_spec(
        exp_folder=exp_folder,
        cog_agent_type='PRLCog',  # This seems to be a placeholder based on the original code.
        additional_rnn_keys={'model_identifier_keys': ['trainval_percent']},
        has_rnn=True,
        has_cog=False,
        model_based=models
    )
    print("Analysis complete.")



def combine_exp_summary(exp_folder, id_keys=None, filter_dict=None):
    if id_keys is None:
        print("Aviso: id_keys não especificado para combine_exp_summary. Colunas da 'config' não serão promovidas automaticamente.")
        id_keys = []
    if filter_dict is None:
        filter_dict = {}
    
    path = MODEL_SAVE_PATH / Path(exp_folder)
    summary_paths = []
    print('Searching for summary files...')
    for p in path.rglob("*"):
        if p.name == 'allfold_summary.pkl':
            summary_paths.append(p)
    print('Found', len(summary_paths), 'summary files.')
    
    combined_summary_list = []
    print('Filtering by', filter_dict)
    
    for summary_path in tqdm(summary_paths):
        try:
            summary = joblib.load(summary_path)
        except Exception as e:
            print(f"Erro ao carregar {summary_path}: {e}")
            continue

        filter_flag = True
        for k_filter, v_filter in filter_dict.items():
            current_summary_passes_filter = False
            if k_filter in summary.columns and not summary[summary[k_filter] == v_filter].empty:
                current_summary_passes_filter = True
            elif 'config' in summary.columns and isinstance(summary['config'].iloc[0], dict) and \
                 k_filter in summary['config'].iloc[0] and summary['config'].iloc[0][k_filter] == v_filter:
                current_summary_passes_filter = True
            
            if not current_summary_passes_filter:
                filter_flag = False
                break
        
        if not filter_flag:
            continue
        
        combined_summary_list.append(summary)

    if not combined_summary_list:
        print('Warning: no summary found or passed filters!')
        return pd.DataFrame()

    combined_summary_df = pd.concat(combined_summary_list, axis=0, join='outer')
    combined_summary_df = combined_summary_df.reset_index(drop=True)

    if combined_summary_df.empty:
        print('Warning: combined_summary_df is empty after concat!')
        return pd.DataFrame()

    if 'config' in combined_summary_df.columns:
        for key in id_keys:
            if key not in combined_summary_df.columns:
                combined_summary_df[key] = combined_summary_df.apply(
                    lambda row: 'none' if not isinstance(row['config'], dict) or key not in row['config'] else row['config'][key],
                    axis=1
                )
    else:
        print("Aviso: Coluna 'config' não encontrada no DataFrame combinado. Não é possível promover id_keys.")

    if 'trainval_percent' in combined_summary_df.columns:
        combined_summary_df['trainval_percent'] = combined_summary_df.apply(
            lambda row: row['config'].get('trainval_percent', row['trainval_percent']) 
                         if isinstance(row['config'], dict) else row['trainval_percent'],
            axis=1
        )
    else:
        if 'config' in combined_summary_df.columns:
             combined_summary_df['trainval_percent'] = combined_summary_df.apply(
                lambda row: row['config'].get('trainval_percent', 100) 
                             if isinstance(row['config'], dict) else 100,
                axis=1
             )

    print(f"combine_exp_summary retornando DataFrame com {len(combined_summary_df)} linhas.")
    return combined_summary_df


def select_best_models_by_keys(df, group_by_keys=None, perf_key='', select_func='min'):
    if group_by_keys is None:
        raise ValueError('group_by_keys should not be None')
    if perf_key == '':
        raise ValueError('perf_key should not be empty')
    if select_func == 'min':
        select_func_ = np.argmin
    elif select_func == 'max':
        select_func_ = np.argmax
    else:
        raise ValueError('select_func not recognized')

    best_df = pd.DataFrame()
    for name, group in df.groupby(group_by_keys):
        best_idx = select_func_(group[perf_key])
        best_df = pd.concat([best_df, group.iloc[best_idx:best_idx+1]], axis=0, join='outer')
    best_df = best_df.reset_index(drop=True)
    return best_df


def select_final_agent_perf(exp_folder, model_identifier_keys=None, cv_keys=None, compete_from_keys=None, filter_dict=None, inner_fold_perf_key='trainval_loss', filter_dict_for_summary=None,
    include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False, spec = False):
    summary = combine_exp_summary(exp_folder, model_identifier_keys+cv_keys+compete_from_keys, filter_dict=filter_dict)
    if 'trainval_percent' in summary.columns:
        summary = summary[~summary['trainval_percent'].isin([9, 18])]
    model_identifier_keys = model_identifier_keys.copy()
    if filter_dict_for_summary is not None:
        for k, v in filter_dict_for_summary.items():
            if not isinstance(v, list):
                v = [v]
            summary = summary[summary[k].isin(v)]
            model_identifier_keys.remove(k)
        filter_dict_for_summary_keys = list(filter_dict_for_summary.keys())
    else:
        filter_dict_for_summary_keys = []
    summary = select_best_models_by_keys(summary, group_by_keys=model_identifier_keys+filter_dict_for_summary_keys+cv_keys, perf_key='val_loss', select_func='min')
    if include_acc:
        summary = insert_model_acc_in_df(summary, dataset_loading_every_time=dataset_loading_every_time, include_acc_filter=include_acc_filter)
    perf_inner_sem = summary.groupby(model_identifier_keys+['outer_fold'], as_index=False).agg(test_loss_inner_sem=('test_loss', 'sem'))
    perf_outer_inner_sem = summary.groupby(model_identifier_keys, as_index=False).agg(test_loss_outer_inner_sem=('test_loss', 'sem'))
    perf_mean_inner_sem = perf_inner_sem.groupby(model_identifier_keys, as_index=False).agg(test_loss_mean_inner_sem=('test_loss_inner_sem', 'mean'))
    if include_acc:
        acc_inner_sem = summary.groupby(model_identifier_keys+['outer_fold'], as_index=False).agg(test_acc_inner_sem=('test_acc', 'sem'))
        acc_outer_inner_sem = summary.groupby(model_identifier_keys, as_index=False).agg(test_acc_outer_inner_sem=('test_acc', 'sem'))
        acc_mean_inner_sem = acc_inner_sem.groupby(model_identifier_keys, as_index=False).agg(test_acc_mean_inner_sem=('test_acc_inner_sem', 'mean'))
    assert inner_fold_perf_key in ['trainval_loss', 'test_loss']
    summary = select_best_models_by_keys(summary, group_by_keys=model_identifier_keys+['outer_fold'], perf_key=inner_fold_perf_key, select_func='min')
    summary['total_test_loss'] = summary.apply(lambda row: row['test_loss']*row['test_trial_num'], axis=1)
    summary['total_trainval_loss'] = summary.apply(lambda row: row['trainval_loss']*row['trainval_trial_num'], axis=1)
    summary['total_train_loss'] = summary.apply(lambda row: row['train_loss']*row['train_trial_num'], axis=1)
    summary['total_val_loss'] = summary.apply(lambda row: row['val_loss']*row['val_trial_num'], axis=1)
    if include_acc:
        summary['total_test_acc'] = summary.apply(lambda row: row['test_acc']*row['test_trial_num'], axis=1)
        summary['total_trainval_acc'] = summary.apply(lambda row: row['trainval_acc']*row['trainval_trial_num'], axis=1)
        summary['total_train_acc'] = summary.apply(lambda row: row['train_acc']*row['train_trial_num'], axis=1)
        summary['total_val_acc'] = summary.apply(lambda row: row['val_acc']*row['val_trial_num'], axis=1)
    perf_agg_dict = dict(
        agg_outer_fold=('outer_fold', list),
        agg_test_loss=('test_loss', list),
        total_test_loss=('total_test_loss', 'sum'),
        test_trial_num=('test_trial_num', 'sum'),
        total_trainval_loss=('total_trainval_loss', 'sum'),
        trainval_trial_num=('trainval_trial_num', 'sum'),
        total_train_loss=('total_train_loss', 'sum'),
        train_trial_num=('train_trial_num', 'sum'),
        total_val_loss=('total_val_loss', 'sum'),
        val_trial_num=('val_trial_num', 'sum'),
        test_loss_outer_std=('test_loss', 'std'),
        test_loss_outer_sem=('test_loss', 'sem'),
        mean_train_trial_num=('train_trial_num', 'mean'),
        mean_val_trial_num=('val_trial_num', 'mean'),
    )
    if include_acc:
        perf_agg_dict.update({
            'agg_test_acc': ('test_acc', list),
            'total_test_acc': ('total_test_acc', 'sum'),
            'total_trainval_acc': ('total_trainval_acc', 'sum'),
            'total_train_acc': ('total_train_acc', 'sum'),
            'total_val_acc': ('total_val_acc', 'sum'),
            'test_acc_outer_std': ('test_acc', 'std'),
            'test_acc_outer_sem': ('test_acc', 'sem'),
        })
    perf = summary.groupby(model_identifier_keys, as_index=False).agg(**perf_agg_dict)
    perf['test_loss_mean_inner_sem'] = perf_mean_inner_sem['test_loss_mean_inner_sem']
    perf['test_loss_outer_inner_sem'] = perf_outer_inner_sem['test_loss_outer_inner_sem']
    perf['test_loss'] = perf['total_test_loss']/perf['test_trial_num']
    perf['trainval_loss'] = perf['total_trainval_loss']/perf['trainval_trial_num']
    perf['train_loss'] = perf['total_train_loss']/perf['train_trial_num']
    perf['val_loss'] = perf['total_val_loss']/perf['val_trial_num']
    if include_acc:
        perf['test_acc_mean_inner_sem'] = acc_mean_inner_sem['test_acc_mean_inner_sem']
        perf['test_acc_outer_inner_sem'] = acc_outer_inner_sem['test_acc_outer_inner_sem']
        perf['test_acc'] = perf['total_test_acc']/perf['test_trial_num']
        perf['trainval_acc'] = perf['total_trainval_acc']/perf['trainval_trial_num']
        perf['train_acc'] = perf['total_train_acc']/perf['train_trial_num']
        perf['val_acc'] = perf['total_val_acc']/perf['val_trial_num']
    return perf, summary


def find_best_models_for_exp_finetuned_spec(exp_folder, cog_agent_type, additional_rnn_keys=None, additional_cog_keys=None,
                             rnn_sort_keys=None, cog_sort_keys=None, has_rnn=True, has_cog=True, cog_hidden_dim=None, return_dim_est=True, check_missing=True,
                             include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False, model_based = None):
    goto_root_dir.run()
    if check_missing:
        check_missing_models(exp_folder)
    inner_fold_perf_key = 'trainval_loss'
    if inner_fold_perf_key == 'trainval_loss':
        fname = ''
    else:
        fname = '_based_on_test'
    print('========Select best models based on inner_fold_perf_key:', inner_fold_perf_key,'exp_folder:', exp_folder)
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    os.makedirs(ana_exp_path, exist_ok=True)
    if has_rnn:
        for combine_model_then_select in [None]:
            temp = select_final_rnn_perf_f_spec(exp_folder, additional_keys=additional_rnn_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                                        return_dim_est=return_dim_est,
                                         combine_model_then_select=combine_model_then_select,sort_keys=rnn_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                         dataset_loading_every_time=dataset_loading_every_time, model_based = model_based)
            if return_dim_est:
                rnn_perf, rnn_summary, perf_est_dim = temp
            else:
                rnn_perf, rnn_summary = temp
                perf_est_dim = None
            fname_temp = fname
            if combine_model_then_select is not None:
                fname_temp += '_combine_then_select'
            joblib.dump(rnn_perf, ana_exp_path / f'rnn_final_perf{fname_temp}.pkl')
            joblib.dump(rnn_summary, ana_exp_path / f'rnn_final_best_summary{fname_temp}.pkl')
            joblib.dump(perf_est_dim, ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.pkl')
            rnn_perf.to_csv(ana_exp_path / f'rnn_final_perf{fname_temp}.csv')
            rnn_summary.to_csv(ana_exp_path / f'rnn_final_best_summary{fname_temp}.csv')
            if perf_est_dim is not None:
                perf_est_dim.to_csv(ana_exp_path / f'rnn_final_perf_est_dim{fname_temp}.csv')
    else:
        pass
    if has_cog:
        cog_perf, cog_summary = select_final_cog_perf(exp_folder, cog_agent_type, additional_keys=additional_cog_keys, inner_fold_perf_key=inner_fold_perf_key,
                                                      cog_hidden_dim=cog_hidden_dim, sort_keys=cog_sort_keys, include_acc=include_acc, include_acc_filter=include_acc_filter,
                                                      dataset_loading_every_time=dataset_loading_every_time)
        joblib.dump(cog_perf, ana_exp_path / f'cog_final_perf{fname}.pkl')
        joblib.dump(cog_summary, ana_exp_path / f'cog_final_best_summary{fname}.pkl')
    else:
        pass


def select_final_rnn_perf_f_spec(exp_folder, additional_keys=None, verbose=True, inner_fold_perf_key='trainval_loss', return_dim_est=False, combine_model_then_select=None,sort_keys=None,
                          include_acc=False, include_acc_filter=lambda row: True, dataset_loading_every_time=False,model_based = None):
    if additional_keys is None:
        additional_keys = {}
    filter_dict = {'agent_type': 'RNN'}
    model_identifier_keys = ['rnn_type', 'hidden_dim', 'readout_FC','model_based'] + additional_keys.setdefault('model_identifier_keys', [])
    cv_keys = ['outer_fold', 'inner_fold'] + additional_keys.setdefault('cv_keys', [])
    compete_from_keys = ['l1_weight', 'seed'] + additional_keys.setdefault('compete_from_keys', [])
    perf, summary = select_final_agent_perf(exp_folder, model_identifier_keys, cv_keys, compete_from_keys, filter_dict,
                                            inner_fold_perf_key=inner_fold_perf_key, filter_dict_for_summary=combine_model_then_select,include_acc=include_acc, include_acc_filter=include_acc_filter,
                                            dataset_loading_every_time=dataset_loading_every_time, spec = True)
    if combine_model_then_select is not None:
        new_rnn_type = '+'.join(combine_model_then_select['model_based'])
        perf['model_based'] = new_rnn_type
        summary['rnn_type'] = new_rnn_type
    if sort_keys is not None:
        idx = 0
        for k in sort_keys:
            perf.insert(idx, k, perf.pop(k))
            summary.insert(idx, k, summary.pop(k))
            idx += 1
        if 'block' in perf.columns:
            perf['block'] = perf['block'].apply(lambda x: -1 if x == 'none' else x)
            summary['block'] = summary['block'].apply(lambda x: -1 if x == 'none' else x)
        perf = perf.sort_values(by=sort_keys)
        summary = summary.sort_values(by=sort_keys)
    perf.insert(1, 'test_loss', perf.pop('test_loss'))
    perf.insert(2, 'trainval_loss', perf.pop('trainval_loss'))
    perf.insert(3, 'train_loss', perf.pop('train_loss'))
    perf.insert(4, 'val_loss', perf.pop('val_loss'))
    if include_acc:
        perf.insert(5, 'test_acc', perf.pop('test_acc'))
        perf.insert(6, 'trainval_acc', perf.pop('trainval_acc'))
        perf.insert(7, 'train_acc', perf.pop('train_acc'))
        perf.insert(8, 'val_acc', perf.pop('val_acc'))
    if verbose:
        with pd_full_print_context():
            print(perf)
    from scipy.stats import ttest_rel
    def estimate_dimensionality(perf, verbose=True, ttest=True):
        perf_gru = perf[perf['readout_FC'] == True].reset_index(drop=True)
        perf_gru = select_best_models_by_keys(perf_gru, group_by_keys=['hidden_dim'], perf_key='test_loss', select_func='min')
        perf_gru['less_pvalue'] = [np.zeros(len(perf_gru)) for _ in range(len(perf_gru))]
        perf_gru['less_than_former'] = [1 for _ in range(len(perf_gru))]
        for i, i_row in perf_gru.iterrows():
            L = i_row['agg_outer_fold']
            assert all(L[i] < L[i+1] for i in range(len(L) - 1)), L
            for j, j_row in perf_gru.iterrows():
                if ttest:
                    pvalue = ttest_rel(i_row['agg_test_loss'], j_row['agg_test_loss'], alternative='less').pvalue
                else:
                    pvalue = 1 - (np.mean(i_row['agg_test_loss']) < np.mean(j_row['agg_test_loss']))
                perf_gru.loc[i, 'less_pvalue'][j] = pvalue
                if j<i and pvalue > 0.05:
                    perf_gru.loc[i, 'less_than_former'] = 0
        if verbose:
            with pd_full_print_context():
                print('Estimated dimensionality:')
                if 'rnn_type' in perf_gru.columns:
                    col_list = ['block'] if 'block' in perf_gru.columns else []
                    col_list += ['rnn_type', 'hidden_dim', 'less_than_former', 'test_loss', 'less_pvalue']
                    print(perf_gru[col_list])
        return perf_gru
    if 'block' not in perf.columns:
        perf_gru = estimate_dimensionality(perf, verbose=verbose)
    else:
        perf_gru = []
        for block in perf['block'].unique():
            perf_gru.append(estimate_dimensionality(perf[perf['block'] == block], verbose=verbose, ttest=False))
        perf_gru = pd.concat(perf_gru, axis=0, join='outer')
    if return_dim_est:
        return perf, summary, perf_gru
    return perf, summary

def analyze_initial_rnn_performance(
    exp_folder,
    additional_id_keys=None,
    filter_dict_for_summary=None,
    sort_keys_list=None
):
    if additional_id_keys is None: additional_id_keys = {}
    model_identifier_keys_for_grouping = ['rnn_type', 'hidden_dim', 'readout_FC',
                                          'model_based', 'trainval_percent', 'agent_name']
    model_identifier_keys_for_grouping += additional_id_keys.get('model_identifier_keys', [])
    model_identifier_keys_for_grouping = sorted(list(set(model_identifier_keys_for_grouping)))
    id_keys_for_combine_call = model_identifier_keys_for_grouping + ['outer_fold', 'inner_fold', 'seed']
    id_keys_for_combine_call = sorted(list(set(id_keys_for_combine_call)))
    perf, detailed_summary = analyze_initial_agent_performance(
        exp_folder,
        model_identifier_keys_for_grouping=model_identifier_keys_for_grouping,
        id_keys_for_combine_call=id_keys_for_combine_call,
        filter_dict_for_summary=filter_dict_for_summary
    )
    if perf.empty:
        return pd.DataFrame(), pd.DataFrame()
    if sort_keys_list is not None:
        for df_to_sort in [perf, detailed_summary]:
            existing_sort_keys = [k for k in sort_keys_list if k in df_to_sort.columns]
            if not existing_sort_keys: continue
            for k_idx, k_val in enumerate(existing_sort_keys):
                df_to_sort.insert(k_idx, k_val, df_to_sort.pop(k_val))
            df_to_sort.sort_values(by=existing_sort_keys, inplace=True)
            df_to_sort.reset_index(drop=True, inplace=True)
    elif 'trainval_percent' in perf.columns:
        default_sort = [k for k in ['agent_name','model_based', 'rnn_type', 'hidden_dim', 'trainval_percent'] if k in perf.columns]
        if default_sort: perf = perf.sort_values(by=default_sort).reset_index(drop=True)
        default_sort_detail = [k for k in ['agent_name','model_based', 'rnn_type', 'hidden_dim', 'trainval_percent', 'outer_fold', 'seed'] if k in detailed_summary.columns]
        if default_sort_detail: detailed_summary = detailed_summary.sort_values(by=default_sort_detail).reset_index(drop=True)
    if 'test_loss' in perf.columns:
        col_to_move = perf.pop('test_loss')
        insert_idx = 0
        ref_col_order = ['agent_name','trainval_percent', 'model_based', 'rnn_type', 'hidden_dim']
        for r_col in ref_col_order:
            if r_col in perf.columns:
                insert_idx = perf.columns.get_loc(r_col) + 1
                break
        perf.insert(insert_idx if insert_idx < len(perf.columns) else len(perf.columns), 'test_loss', col_to_move)
    return perf, detailed_summary

def analyze_initial_agent_performance(
    exp_folder,
    model_identifier_keys_for_grouping=None,
    id_keys_for_combine_call=None,
    filter_dict_for_summary=None
):
    if model_identifier_keys_for_grouping is None:
        raise ValueError('model_identifier_keys_for_grouping devem ser especificados.')
    if id_keys_for_combine_call is None:
        print("Aviso: id_keys_for_combine_call não foi fornecido para analyze_initial_agent_performance. Tentando adivinhar.")
        id_keys_for_combine_call = model_identifier_keys_for_grouping + ['outer_fold', 'inner_fold', 'seed']
        id_keys_for_combine_call = sorted(list(set(id_keys_for_combine_call)))
    summary_df = combine_exp_summary(exp_folder, id_keys=id_keys_for_combine_call, filter_dict=None)
    if summary_df.empty:
        print(f"Aviso: Nenhum sumário encontrado ou combinado para {exp_folder}.")
        return pd.DataFrame(), pd.DataFrame()
    if filter_dict_for_summary is not None:
        for k, v_list in filter_dict_for_summary.items():
            if not isinstance(v_list, list): v_list = [v_list]
            if k in summary_df.columns:
                summary_df = summary_df[summary_df[k].isin(v_list)]
            else:
                print(f"Aviso: Chave de filtro '{k}' não encontrada nas colunas: {summary_df.columns.tolist()}")
    if summary_df.empty:
        print(f"Aviso: Nenhum sumário restante após a filtragem para {exp_folder}.")
        return pd.DataFrame(), pd.DataFrame()
    grouping_keys = [k for k in model_identifier_keys_for_grouping if k in summary_df.columns]
    if not grouping_keys:
        print("Aviso: Nenhuma chave de agrupamento válida. Verifique model_identifier_keys_for_grouping.")
        return pd.DataFrame(), summary_df
    missing_grouping_keys = [k for k in grouping_keys if k not in summary_df.columns]
    if missing_grouping_keys:
        print(f"ERRO: Chaves de agrupamento ausentes: {missing_grouping_keys}. Colunas: {summary_df.columns.tolist()}")
        return pd.DataFrame(), summary_df
    print(f"Agrupando por: {grouping_keys}")
    perf_agg_dict = {
        'test_loss': ('test_loss', 'mean'),
        'test_loss_std': ('test_loss', 'std'),
        'test_loss_sem': ('test_loss', lambda x: x.sem(ddof=0) if len(x) > 1 else np.nan),
        'test_trial_num': ('test_trial_num', 'mean'),
        'run_count': ('test_loss', 'count')
    }
    aggregated_perf = summary_df.groupby(grouping_keys, as_index=False).agg(**perf_agg_dict)
    return aggregated_perf, summary_df


def find_initial_test_loss_for_exp(
    exp_folder,
    model_filters=None,
    rnn_sort_keys=None,
    save_suffix='_initial_perf',
    additional_rnn_keys=None,
):
    print(f"======== Analisando Test Loss Inicial para exp_folder: {exp_folder} ========")
    ana_exp_output_path = Path(ANA_SAVE_PATH) / exp_folder 
    os.makedirs(ana_exp_output_path, exist_ok=True)
    rnn_perf_agg, rnn_summary_detailed = analyze_initial_rnn_performance(
        exp_folder,
        filter_dict_for_summary=model_filters,
        sort_keys_list=rnn_sort_keys
    )
    if not rnn_perf_agg.empty:
        print("\n--- Performance Inicial Agregada (RNN) ---")
        with pd_full_print_context(): print(rnn_perf_agg)
        file_basename_agg = f'rnn_final_perf{save_suffix}'
        joblib.dump(rnn_perf_agg, ana_exp_output_path / f'{file_basename_agg}.pkl')
        rnn_perf_agg.to_csv(ana_exp_output_path / f'{file_basename_agg}.csv', index=False)
        print(f"Performance agregada salva em {ana_exp_output_path / file_basename_agg}.[pkl/csv]")
    if not rnn_summary_detailed.empty:
        file_basename_detail = f'rnn_detailed_summary{save_suffix}'
        joblib.dump(rnn_summary_detailed, ana_exp_output_path / f'{file_basename_detail}.pkl')
        rnn_summary_detailed.to_csv(ana_exp_output_path / f'{file_basename_detail}.csv', index=False)
        print(f"Sumário detalhado salvo em {ana_exp_output_path / file_basename_detail}.[pkl/csv]")
    if rnn_perf_agg.empty and rnn_summary_detailed.empty:
        print("Nenhum dado de performance inicial foi processado ou salvo.")
    return rnn_perf_agg, rnn_summary_detailed
