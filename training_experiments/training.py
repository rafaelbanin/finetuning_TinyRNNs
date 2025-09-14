import torch
import multiprocessing as mp
import numpy as np
from sklearn.model_selection import KFold
import os
import sys
from pathlib import Path
import joblib
# import multiprocessing as mp
import torch.multiprocessing as mps
import gc
from agents import Agent
import pandas as pd
from datasets import Dataset
from path_settings import *
import pprint
from utils.logger import PrinterLogger
from utils import goto_root_dir, set_os_path_auto
from . import config_control
import time

pp = pprint.PrettyPrinter(depth=4, sort_dicts=False, compact=True)

def split_over_trials(n_trial, n_trial_splits=[], seed=0):
    np.random.seed(seed)
    assert sum(n_trial_splits) == n_trial
    all_trial_index = np.random.permutation(np.arange(0, n_trial))
    indexes = []
    trial_start = 0
    for n_trial_split in n_trial_splits:
        indexes.append(np.sort(all_trial_index[trial_start:trial_start + n_trial_split]))
        trial_start += n_trial_split
    return indexes

def _get_cv_agent_summary(ag):
    trainval_trial_num = ag.best_model_pass['train']['total_trial_num'] + ag.best_model_pass['val']['total_trial_num']
    trainval_loss = ag.best_model_pass['train']['behav_loss'] * ag.best_model_pass['train']['total_trial_num'] + \
                    ag.best_model_pass['val']['behav_loss'] * ag.best_model_pass['val']['total_trial_num']
    trainval_loss /= trainval_trial_num
    summary = {
        'outer_fold': ag.config['outer_fold'],
        'inner_fold': ag.config['inner_fold'],
        'seed': ag.config['seed'],
        'test_loss': ag.best_model_pass['test']['behav_loss'],
        'trainval_loss': trainval_loss,
        'train_loss': ag.best_model_pass['train']['behav_loss'],
        'val_loss': ag.best_model_pass['val']['behav_loss'],
        'test_trial_num': ag.best_model_pass['test']['total_trial_num'],
        'trainval_trial_num': trainval_trial_num,
        'train_trial_num': ag.best_model_pass['train']['total_trial_num'],
        'val_trial_num': ag.best_model_pass['val']['total_trial_num'],
        'test_index': ag.config['test_index'],
        'train_index': ag.config['train_index'],
        'val_index': ag.config['val_index'],
        'model_path': ag.config['model_path'],
        'config': ag.config,
        'trainval_percent': ag.config.get('trainval_percent', 100)
    }
    if 'all_target_names' in ag.best_model_pass['train'] and len(ag.best_model_pass['train']['all_target_names']) > 1:
        for target_name in ag.best_model_pass['train']['all_target_names']:
            for data_pass_name in ['train', 'val', 'test']:
                if 'behav_loss_' + target_name in ag.best_model_pass[data_pass_name]:
                    summary[data_pass_name + '_loss_' + target_name] = ag.best_model_pass[data_pass_name]['behav_loss_' + target_name]
    return summary

def _ensure_trainval_percent_column(df):
    if 'trainval_percent' not in df.columns and 'config' in df.columns:
        df['trainval_percent'] = df['config'].apply(lambda c: c.get('trainval_percent', 100))
    return df

def _train_one_agent(ag, verbose_level=2):
    """Train one agent with bound data.

    After training, the training summary of the well-trained model will be saved in the agent's folder.
    After completing the inner fold, all the training summary will be combined into a single file.
    The existence of the summary file indicates that the training of this model is completed.
    """
    summary_path = MODEL_SAVE_PATH / ag.config['model_path'] / 'temp_summary.pkl'
    if os.path.exists(summary_path):
        print(f'Agent {ag.config["model_path"]} already trained. Skip training.')
        try: # EOFError
            with set_os_path_auto():
                summary = joblib.load(summary_path)
            ag.load(ag.config['model_path'])
            ag.summary = summary
            return ag
        except EOFError: # unknown error caused by nautilus
            # delete the corrupted file into trash bin
            os.remove(summary_path)
            print(f'Corrupted summary file {summary_path} deleted.')

    # either not exist or corrupted 
    ag = ag.train(verbose_level=verbose_level)
    summary = _get_cv_agent_summary(ag)
    joblib.dump(summary, summary_path)
    ag.summary = summary
    return ag

def agent_pool_auto_train(agents_list, n_jobs=1, verbose_level=2):
    """Automatically train a pool of agents with bound data.

    Args:
        agents_list: a list of trainable agents, with bound trainer data.
        n_jobs: multiprocessing: -1 for using all cores; 1 for one core.

    Returns:
         the updated agents_list, otherwise the updated agents will be lost after subprocesses ends
    """

    if n_jobs == -1:
        n_jobs = min(mp.cpu_count() // 2, len(agents_list)) # use half of the logical cores
    config = agents_list[0].config
    if n_jobs > 1 and config['agent_type'] == 'RNN':
        print('Torch automatically use all cpu cores. Multiprocessing can only be 2x speedup, but introducing many unexpected bugs. Using 1 core instead.')
        n_jobs = 1
    print(f'Start {len(agents_list)} agents training with {n_jobs} processes.')
    time_start = time.time()
    if n_jobs > 1:
        assert 'device' not in config or config['device'] == 'cpu'
        pool = mp.Pool(processes=n_jobs)
        temp_results = [pool.apply_async(_train_one_agent, args=(ag, verbose_level)) for ag in agents_list]
        cv_results = [p.get() for p in temp_results]
        pool.terminate()
        pool.close()
        del pool
        gc.collect()
    else:
        cv_results = []
        for ag in agents_list:
            cv_results.append(_train_one_agent(ag, verbose_level=verbose_level))
    print(f'Finished {len(agents_list)} agents training with {n_jobs} processes. Time used: {time.time() - time_start:.3f}s')
    return cv_results # a list of agents



def behavior_cv_training_test_only(config, n_jobs=1, verbose_level=1):
    """
    Executa a estrutura CV, mas APENAS calcula e salva a test_loss
    inicial (sem treinar).
    """
    exp_model_path = Path(config['model_path'])
    training_summary_path = MODEL_SAVE_PATH / exp_model_path / f'allfold_summary.pkl'

    # Se já existe, não faz nada (ou você pode optar por deletar e refazer)
    if os.path.exists(training_summary_path):
        if verbose_level >= 1: print('Already existing (Test Only) ', training_summary_path)
        with set_os_path_auto(): return joblib.load(training_summary_path)

    pp.pprint(config)
    behav_data_spec = config['behav_data_spec']
    if isinstance(behav_data_spec, list):
        behav_data_spec = {k: config[k] for k in behav_data_spec}
    behav_dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec, verbose=False).behav_to(config)
    print('Data block num', behav_dt.batch_size)

    outer_fc_rs = 1
    outer_kf = KFold(n_splits=config['outer_splits'], shuffle=True, random_state=outer_fc_rs)
    seed_num = config['seed_num']
    training_summary_list = []

    print("\n--- Running Initial Test Loss Calculation Only ---")

    # Determine the model path for the synthetic pre-trained model if applicable
    sintetic_model_path = None
    if config['model_based'] == 'base':
        # Se base
        sintetic_model_path = 'exp_finetuned_monkeyV/agent_name-{agent_name}.rnn_type-{rnn_type}.hidden_dim-{hidden_dim}.model_based-sintetic.trainval_percent-100'.format(
            agent_name=config['agent_name'],
            rnn_type=config['rnn_type'],
            hidden_dim=config['hidden_dim']
        )
        print(f"Loading weights from pre-trained synthetic model at: {sintetic_model_path}")
    else:
        if config['model_based'] == '100_pre_trained':
            model_number = '100'
        elif config['model_based'] == '70_pre_trained':
            model_number = '70'
        elif config['model_based'] == '50_pre_trained':
            model_number = '50'
        elif config['model_based'] == '20_pre_trained':
            model_number = '20'

        # Outros modelos sintéticos
        sintetic_model_path = 'exp_finetuned_monkeyV/agent_name-{agent_name}.rnn_type-{rnn_type}.hidden_dim-{hidden_dim}.model_based-sintetic.trainval_percent-{num}'.format(
            agent_name=config['agent_name'],
            rnn_type=config['rnn_type'],
            hidden_dim=config['hidden_dim'],
            num = model_number
        )
        print(f"Loading weights from pre-trained synthetic model at: {sintetic_model_path}")




    for outer_idx, (trainval_index_full, test_index) in enumerate(outer_kf.split(range(behav_dt.batch_size))):
        print(f"  Outer Fold {outer_idx}...")

        inner_kf_rs = outer_idx
        trainval_index = trainval_index_full.copy()
        current_trainval_percent = config.get('trainval_percent', 100)

        if current_trainval_percent < 100 and current_trainval_percent > 0:
            blocks_total_num = len(trainval_index)
            blocks_subsample_num = np.round((blocks_total_num * current_trainval_percent / 100)).astype(int)

            if blocks_subsample_num < config['inner_splits']:
                 print(f"    WARNING: Subsample ({blocks_subsample_num}) < inner_splits. Reducing splits to {blocks_subsample_num}")
                 n_splits_to_use = blocks_subsample_num if blocks_subsample_num > 1 else 2
            else:
                 n_splits_to_use = config['inner_splits']

            trainval_index = np.random.choice(trainval_index, size=blocks_subsample_num, replace=False)
            trainval_index.sort()
        elif current_trainval_percent == 0:
            n_splits_to_use = config['inner_splits'] # Não importa, vamos tratar
        else: # 100%
            n_splits_to_use = config['inner_splits']

        # KFold precisa de pelo menos n_splits amostras.
        # Se trainval_index for muito pequeno (ou 0), KFold falhará.
        # Precisamos de uma estratégia para o split interno, mesmo sem treinar.
        # Vamos criar splits "falsos" se necessário.
        if len(trainval_index) < n_splits_to_use or current_trainval_percent == 0:
             print(f"    Using dummy inner splits for trainval_percent={current_trainval_percent}%.")
             inner_splits_data = [(np.array([]), np.array([]))] # Um único split "falso"
             num_inner_splits = 1
        else:
             inner_kf = KFold(n_splits=n_splits_to_use, shuffle=True, random_state=inner_kf_rs)
             inner_splits_data = inner_kf.split(trainval_index)
             num_inner_splits = n_splits_to_use

        # Carrega os dados de teste UMA VEZ por outer_fold
        test_data = behav_dt.get_behav_data(test_index, config)

        for inner_idx_raw, (train_idx_inner, val_idx_inner) in enumerate(inner_splits_data):

            # Se estamos usando splits falsos, inner_idx = 0. Senão, usa o real.
            inner_idx = 0 if current_trainval_percent == 0 else inner_idx_raw

            if current_trainval_percent == 0:
                 train_index_cv = np.array([])
                 val_index_cv = np.array([])
            else:
                 train_index_cv = trainval_index[train_idx_inner]
                 val_index_cv = trainval_index[val_idx_inner]

            for seed in range(seed_num):
                model_name = f'outerfold{outer_idx}_innerfold{inner_idx}_seed{seed}'
                current_config = config | {
                    'outer_fold': outer_idx,
                    'inner_fold': inner_idx,
                    'seed': seed,
                    'train_index': train_index_cv,
                    'val_index': val_index_cv,
                    'test_index': test_index,
                    'model_path': exp_model_path / (model_name + '_INITIAL_ONLY') # Nome diferente
                }

                # Cria o agente (sem ['train'])
                a = Agent(config['agent_type'], config=current_config)

                # Carrega os pesos se for um modelo "Sintetic_base"
                if sintetic_model_path:
                    # Encontra o arquivo de pesos correspondente a este fold
                    # Exemplo: exp_finetuned_monkeyV/.../outerfold0_innerfold0_seed0
                    load_model_path = Path(sintetic_model_path) / f'outerfold{outer_idx}_innerfold{inner_idx}_seed{seed}'
                    try:
                        print(f"  Outer Fold {outer_idx}, Inner Fold {inner_idx}, Seed {seed}: Loading weights from {load_model_path}")
                        a.load(model_path=str(load_model_path), mode='eval')
                    except FileNotFoundError:
                        print(f"  WARNING: Pre-trained synthetic model not found at {load_model_path}. Using random weights.")

                # Mover para GPU se necessário
                if 'device' in config and config['device'] == 'cuda' and torch.cuda.is_available():
                    a.model.to('cuda')
                    test_data_gpu = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k,v in test_data.items()}
                    # Cuidado com mask tuple
                    if isinstance(test_data_gpu['mask'], tuple):
                         test_data_gpu['mask'] = (test_data_gpu['mask'][0].to('cuda'), test_data_gpu['mask'][1].to('cuda'))
                    test_pass = a._eval_1step(test_data_gpu['input'], test_data_gpu['target'], test_data_gpu['mask'])
                else:
                    test_pass = a._eval_1step(test_data['input'], test_data['target'], test_data['mask'])

                # Cria o sumário
                summary = {
                    'outer_fold': outer_idx, 'inner_fold': inner_idx, 'seed': seed,
                    'test_loss': test_pass['behav_loss'].item(),
                    'trainval_loss': np.nan, 'train_loss': np.nan, 'val_loss': np.nan,
                    'test_trial_num': test_pass['total_trial_num'].item(),
                    'trainval_trial_num': 0, 'train_trial_num': 0, 'val_trial_num': 0,
                    'test_index': test_index,
                    'train_index': train_index_cv, # Guarda os índices (mesmo que vazios)
                    'val_index': val_index_cv,     # Guarda os índices (mesmo que vazios)
                    'model_path': current_config['model_path'],
                    'config': current_config,
                    'trainval_percent': current_config.get('trainval_percent', 100),
                }
                training_summary_list.append(summary)
                del a, test_pass # Libera memória

    final_summary_df = pd.DataFrame(training_summary_list)
    final_summary_df = final_summary_df.drop_duplicates(subset=['outer_fold', 'inner_fold', 'seed', 'trainval_percent'], keep='last').reset_index(drop=True)

    os.makedirs(training_summary_path.parent, exist_ok=True)
    joblib.dump(final_summary_df, training_summary_path)
    print(f"\nSumário (Apenas Teste Inicial) salvo em {training_summary_path}")
    return final_summary_df

def behavior_cv_training(config, n_jobs=1, verbose_level=1):
    """Cross-validated training of one model on one behavioral dataset.

    This function is supposed to train a series of models (different only due to kfolds and seeds) on one dataset.
    In the outer loop, perform kfold for train+val dataset and test dataset separation.
    In the inner loop, perform kfold for train dataset and val dataset separation.
    For each separation, many seeds are required.
    The model is trained on the training set and validated on the validation set.
    The model is trained for a number of epochs, and the best model is selected based on the validation loss.
    The best model is saved in the folder of the current experiment.
    The results of all models are saved in the folder of the current experiment.

    This will create nested folders, save models, configs, final results in:
        files/trained_models/experiment_name_folder/agent_name_folder/outerfold?_innerfold?_seed?

    Args:
        config: everything
        n_jobs: multiprocessing: -1 for using all cores.
        verbose_level: 0 for no print (maybe seldom used);
            1 for print only the new results;
            2 for print existing+new results.

    """
    if 'split_training' in config and config['split_training']:
        print("A modificação para 'trainval_percent=0' foi implementada apenas para behavior_cv_training.")
        return behavior_split_training(config, n_jobs=n_jobs, verbose_level=verbose_level)

    exp_model_path = Path(config['model_path'])
    training_summary_path = MODEL_SAVE_PATH / exp_model_path / f'allfold_summary.pkl'
    training_summary = pd.DataFrame() # Começa vazio, carrega depois se existir

    if os.path.exists(training_summary_path):
        with set_os_path_auto(): temp_df = joblib.load(training_summary_path)
        temp_df = _ensure_trainval_percent_column(temp_df) 

        if 0 in temp_df['trainval_percent'].unique():
             if verbose_level >= 1: print('Already existing (with 0%) ', training_summary_path)
             return temp_df
        else:
             if verbose_level >= 1: print('Already existing (WITHOUT 0%) ', training_summary_path, ' - Will add 0% points.')
             training_summary = temp_df
    else:
        training_summary = pd.DataFrame()

    pp.pprint(config)
    behav_data_spec = config['behav_data_spec']
    if isinstance(behav_data_spec, list):
        behav_data_spec = {k: config[k] for k in behav_data_spec}
    behav_dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec, verbose=False).behav_to(config)
    print('Data block num', behav_dt.batch_size)

    outer_fc_rs = 1
    outer_kf = KFold(n_splits=config['outer_splits'], shuffle=True, random_state=outer_fc_rs)
    seed_num = config['seed_num']

    initial_summaries_all = []

    for outer_idx, (trainval_index_full, test_index) in enumerate(outer_kf.split(range(behav_dt.batch_size))):
        pool_training_summary_path = MODEL_SAVE_PATH / exp_model_path / f'outerfold{outer_idx}_summary.pkl'

        if os.path.exists(pool_training_summary_path):
             with set_os_path_auto(): pool_training_summary = joblib.load(pool_training_summary_path)
             pool_training_summary = _ensure_trainval_percent_column(pool_training_summary) 
             if verbose_level >= 1: print('Already existing ', pool_training_summary_path)
             training_summary = pd.concat([training_summary, pool_training_summary], axis=0, join='outer').drop_duplicates(subset=['outer_fold', 'inner_fold', 'seed', 'trainval_percent'], keep='last').reset_index(drop=True)
             continue

        cv_agents_list = []
        initial_summaries_fold = []

        inner_kf_rs = outer_idx
        inner_kf = KFold(n_splits=config['inner_splits'], shuffle=True, random_state=inner_kf_rs)

        trainval_index = trainval_index_full.copy()
        current_trainval_percent = config.get('trainval_percent', 100) # Pega o % da config atual
        if current_trainval_percent < 100 and current_trainval_percent > 0:
            blocks_total_num = len(trainval_index)
            blocks_subsample_num = np.round((blocks_total_num * current_trainval_percent / 100)).astype(int)
            trainval_index = np.random.choice(trainval_index, size=blocks_subsample_num, replace=False)
            trainval_index.sort()
            assert blocks_subsample_num >= config['inner_splits']

        for inner_idx, (train_idx_inner, val_idx_inner) in enumerate(inner_kf.split(trainval_index)):
            train_index_cv = trainval_index[train_idx_inner]
            val_index_cv = trainval_index[val_idx_inner]

            trainer_data = {}
            trainer_data['train'] = behav_dt.get_behav_data(train_index_cv, config)
            trainer_data['val'] = behav_dt.get_behav_data(val_index_cv, config)
            trainer_data['test'] = behav_dt.get_behav_data(test_index, config)

            for seed in range(seed_num):
                model_name = f'outerfold{outer_idx}_innerfold{inner_idx}_seed{seed}'
                current_config = config | {
                    'outer_fold': outer_idx,
                    'inner_fold': inner_idx,
                    'seed': seed,
                    'train_index': train_index_cv,
                    'val_index': val_index_cv,
                    'test_index': test_index,
                    'model_path': exp_model_path / model_name
                }
                
                a0 = Agent(config['agent_type'], config=current_config)
                test_data_t = trainer_data['test']

                if 'device' in config and config['device'] == 'cuda' and torch.cuda.is_available():
                    a0.model.to('cuda')
                    test_data_t['input'] = test_data_t['input'].to('cuda')
                    if isinstance(test_data_t['target'], torch.Tensor): test_data_t['target'] = test_data_t['target'].to('cuda')
                    if isinstance(test_data_t['mask'], torch.Tensor): test_data_t['mask'] = test_data_t['mask'].to('cuda')
                    elif isinstance(test_data_t['mask'], tuple): test_data_t['mask'] = (test_data_t['mask'][0].to('cuda'), test_data_t['mask'][1].to('cuda'))

                test_pass0 = a0._eval_1step(test_data_t['input'], test_data_t['target'], test_data_t['mask'])
                
                del a0, test_pass0

                a = Agent(config['agent_type'], ['train'], config=current_config)
                if 'model_init_path' in config: a.load(config['model_init_path'])
                a.bind_data(trainer_data)
                cv_agents_list.append(a)

        cv_agents_list = agent_pool_auto_train(cv_agents_list, n_jobs=n_jobs, verbose_level=verbose_level)
        [a.training_diagnose() for a in cv_agents_list]

        pool_training_summary_list = [ag.summary for ag in cv_agents_list] + initial_summaries_fold
        pool_training_summary = pd.DataFrame(pool_training_summary_list)
        pool_training_summary = _ensure_trainval_percent_column(pool_training_summary) # <<< Garante a coluna nos novos

        joblib.dump(pool_training_summary, pool_training_summary_path)

        if verbose_level >= 1:
            with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
                 print(pool_training_summary)

        training_summary = pd.concat([training_summary, pool_training_summary], axis=0, join='outer').drop_duplicates(subset=['outer_fold', 'inner_fold', 'seed', 'trainval_percent'], keep='last').reset_index(drop=True)

    final_summary_df = pd.DataFrame(initial_summaries_all)
    final_summary_df = _ensure_trainval_percent_column(final_summary_df) # <<< Garante a coluna nos 0%

    training_summary = pd.concat([training_summary, final_summary_df], axis=0, join='outer')
    training_summary = _ensure_trainval_percent_column(training_summary) # <<< Garante antes do drop final

    training_summary = training_summary.drop_duplicates(subset=['outer_fold', 'inner_fold', 'seed', 'trainval_percent'], keep='last').reset_index(drop=True)

    joblib.dump(training_summary, training_summary_path)
    print(f"Sumário final (com 0%) salvo em {training_summary_path}")
    return training_summary


def behavior_split_training(config, n_jobs=1, verbose_level=1):
    """ Training of one model on one behavioral dataset with train/val/test split.
    The split is not implemented here, but outside this function.

    This will create nested folders, save models, configs, final results in (for consistency):
        files/trained_models/experiment_name_folder/agent_name_folder/outerfold0_innerfold0_seed?

    Args:
        config: everything
        n_jobs: multiprocessing: -1 for using all cores.
        verbose_level: 0 for no print (maybe seldom used);
            1 for print only the new results;
            2 for print existing+new results.

    """
    # logging_file = Path('logs') / (str(config['model_path']).replace('/','.').replace('\\','.') + '.log.txt')
    # os.makedirs(logging_file.parent, exist_ok=True)
    # sys.stdout = PrinterLogger(sys.stdout, open(logging_file, 'a+'))

    goto_root_dir.run()
    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    for c in configs:
        if config_modifier is not None:
            c = config_modifier(c)
        print(f"\n--- Chamando para config com trainval_percent = {c.get('trainval_percent')} ---")
        behavior_cv_training_test_only(c, n_jobs=n_jobs, verbose_level=verbose_level)


    pp.pprint(config)
    behav_data_spec = config['behav_data_spec']
    if isinstance(behav_data_spec, list): # assemble the dict for behavior data specification
        behav_data_spec = {k: config[k] for k in behav_data_spec}
    behav_dt = Dataset(config['dataset'], behav_data_spec=behav_data_spec, verbose=False)
    behav_dt = behav_dt.behav_to(config) # transform format following specifications
    print('Data block num', behav_dt.batch_size)

    training_summary = pd.DataFrame()

    seed_num = config['seed_num']
    outer_idx = 0
    inner_idx = 0
    agents_list = []
    train_index = config['train_index']
    val_index = config['val_index']
    test_index = config['test_index']
    train_trial_index = config['train_trial_index'] if 'train_trial_index' in config else None
    val_trial_index = config['val_trial_index'] if 'val_trial_index' in config else None
    test_trial_index = config['test_trial_index'] if 'test_trial_index' in config else None
    trainer_data = {}
    trainer_data['train'] = behav_dt.get_behav_data(train_index, config, selected_trial_indices=train_trial_index)
    trainer_data['val'] = behav_dt.get_behav_data(val_index, config, selected_trial_indices=val_trial_index)
    trainer_data['test'] = behav_dt.get_behav_data(test_index, config, selected_trial_indices=test_trial_index)
    if 'distill' in config and config['distill'] == 'student':
        # trainer_data['train']['target'], trainer_data['val']['target'] = _load_teacher_network_score(config, train_index, val_index, trainer_data['train'], trainer_data['val'], outer_idx, )
        # this function will change trainer_data['train']['target'], trainer_data['val']['target']
        # trainer_data['train']['mask'], trainer_data['val']['mask], trainer_data['test']['mask'] are not changed
        _load_teacher_network_score(config,
                         train_index,
                         val_index,
                         trainer_data,
                         outer_idx)
    for seed in range(seed_num):
        a = Agent(config['agent_type'], ['train'],
                            config=config | {
                                'outer_fold': outer_idx,
                                'inner_fold': inner_idx,
                                'seed': seed,
                                'model_path': exp_model_path / f'outerfold{outer_idx}_innerfold{inner_idx}_seed{seed}'})
        if 'model_init_path' in config:
            print('Loading pretrained model from', config['model_init_path'])
            a.load(config['model_init_path'])
        a.bind_data(trainer_data)
        agents_list.append(a)

    agents_list = agent_pool_auto_train(agents_list, n_jobs=n_jobs, verbose_level=verbose_level)

    [a.training_diagnose() for a in agents_list]
    # save summary information in a list of dicts, then transform the list to a pandas dataframe; save a lot of time
    training_summary = [ag.summary for ag in agents_list]
    training_summary = pd.DataFrame(training_summary)
    if verbose_level >= 1:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)
        print(pd.DataFrame(training_summary))

    joblib.dump(training_summary, training_summary_path)
    return training_summary

def behavior_cv_training_config_combination(base_config, config_ranges, n_jobs=1, verbose_level=1, config_modifier=None):
    goto_root_dir.run()
    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    for c in configs:
        if config_modifier is not None:
            c = config_modifier(c)

        current_percent = c.get('trainval_percent', 100)
        print(f"\nDEBUG: Chamando com trainval_percent = {current_percent}")
        if current_percent == 0:
             behavior_cv_training_test_only(c, n_jobs=n_jobs, verbose_level=verbose_level)
        else:
             behavior_cv_training(c, n_jobs=n_jobs, verbose_level=verbose_level)


def get_training_exp_folder_name(file_string):
    """Automatically generate the current training_exp folder

    Args:
        file_string: __file__

    Returns:
        the file name without .py
    """
    return os.path.basename(file_string)[:-3] # remove .py

def _load_teacher_network_score(config, train_index, val_index, trainer_data, outer_idx):
    """Load the teacher network score for distillation.

    Take a student RNN config. The student RNN will be trained on the train_index and val_index.
    So the train and val data trainer_data will be replaced by the teacher network score.
    The teacher network score is loaded from the teacher network model_path.
    """
    train_data, val_data = trainer_data['train'], trainer_data['val']
    model_path = config['teacher_model_path']
    # model_path = model_path.replace('XXX',str(config['trainval_size']))
    model_scores = None
    # assert outer_idx == 0
    outerfold_name = 'outerfold' + str(outer_idx)

    if os.path.exists(ANA_SAVE_PATH / model_path):
        if 'outerfold' not in str(model_path):
            folder_names = [f.name for f in os.scandir(ANA_SAVE_PATH / model_path) if f.is_dir()]
            folder_names = [f for f in folder_names if outerfold_name in f]
            assert  len(folder_names) == 1
            folder_name = folder_names[0]
            the_model_path = ANA_SAVE_PATH / model_path / folder_name
        else:
            the_model_path = ANA_SAVE_PATH / model_path
        with set_os_path_auto():
            model_pass  = joblib.load(the_model_path / f'total_scores.pkl')
        model_scores = model_pass['scores']
        model_mask = model_pass['mask']

        print('Loading model scores from', the_model_path)
        #print(train_data['target'].shape) # (seq_len, batch_size)

        n_actions = model_scores[0].shape[-1]
        student_model = 'rnn'
        if 'target' in train_data: # for RNNs
            assert config['output_h0']
            train_seq_len, train_batch_size = train_data['target'].shape
            val_seq_len, val_batch_size = val_data['target'].shape
        else: # for cog models
            assert isinstance(train_data['input'], list) # list of nn_sessions
            student_model = 'cog'
            train_batch_size = len(train_data['input'])
            train_seq_len = len(train_data['input'][0]['outcomes'])
            val_batch_size = len(val_data['input'])
            val_seq_len = len(val_data['input'][0]['outcomes'])
            assert train_seq_len == val_seq_len
            tch_seq_len = model_scores[train_index[0]].shape[0] - 1 # the last one is for output_h0 = True
            if tch_seq_len == train_seq_len * 2: # for two stages
                train_seq_len *= 2
                val_seq_len *= 2
            elif tch_seq_len == train_seq_len:
                pass
            else:
                raise ValueError(f'tch_seq_len {tch_seq_len} != train_seq_len {train_seq_len}')
        assert len(train_index) == train_batch_size
        assert len(val_index) == val_batch_size
        train_model_scores = np.zeros((train_seq_len,
                                       train_batch_size,
                                       n_actions
                                       )) # (seq_len, batch_size,2)

        for i, t in enumerate(train_index): # replace train data
            model_score = model_scores[t][:-1] # remove the last one for output_h0 = True
            if isinstance(model_mask[t], tuple):  # if the mask for the sub t is tuple
                target_subtractive_mask = model_mask[t][1]
            else:
                target_subtractive_mask = 0
            train_model_scores[:, i, :] = model_score - target_subtractive_mask

        val_model_scores = np.zeros((val_seq_len,
                                     val_batch_size,
                                     n_actions
                                     ))
        for i, t in enumerate(val_index): # replace val data
            model_score = model_scores[t][:-1]
            if isinstance(model_mask[t], tuple):  # if the mask for the sub t is tuple
                target_subtractive_mask = model_mask[t][1]
            else:
                target_subtractive_mask = 0
            val_model_scores[:, i, :] = model_score - target_subtractive_mask

    if model_scores is None:
        raise FileNotFoundError(f'No model score found in {ANA_SAVE_PATH / model_path}')
    train_model_scores = torch.from_numpy(train_model_scores).to(device=config['device'])
    val_model_scores = torch.from_numpy(val_model_scores).to(device=config['device'])
    # print(train_model_scores.shape, val_model_scores.shape)
    # transform score to probability with softmax
    if 'distill_temp' in config:
        distill_temp = config['distill_temp']
    else:
        distill_temp = 1
    sm = torch.nn.Softmax(dim=2)
    train_model_prob = sm(train_model_scores / distill_temp)
    val_model_prob = sm(val_model_scores / distill_temp)
    if 'teacher_prop' in config and config['teacher_prop'] != 1:
        assert 0 <= config['teacher_prop'] <= 1
        assert not np.isclose(config['teacher_prop'], 0.5), 'teacher_prop = 0.5 is not supported, causing key conflict'
        assert student_model == 'rnn'
        trainer_data['train']['target'] = [
            ('tch', config['teacher_prop'], train_model_prob), # teacher logit target
            ('action', 1-config['teacher_prop'], trainer_data['train']['target']), # original action target
        ]
        trainer_data['val']['target'] = [
            ('tch', config['teacher_prop'], val_model_prob),
            ('action', 1-config['teacher_prop'], trainer_data['val']['target']),
        ]
    else:
        if student_model == 'rnn':
            trainer_data['train']['target'] = train_model_prob
            trainer_data['val']['target'] = val_model_prob
        else:
            train_model_prob = train_model_prob.cpu().numpy() # (seq_len, batch_size, n_actions)
            val_model_prob = val_model_prob.cpu().numpy() # (seq_len, batch_size, n_actions)
            for b in range(train_batch_size):
                trainer_data['train']['input'][b]['target'] = train_model_prob[:,b,:]
            for b in range(val_batch_size):
                trainer_data['val']['input'][b]['target'] = val_model_prob[:,b,:]
    return trainer_data

def _load_other_network_score(config, train_index, val_index, train_data, val_data, outer_idx):
    
    model_path = config['model_path']
    # print(model_path)
    model_path = model_path.replace('.trainprob-True','').replace('_dataprop','')
    # delete ".trainval_percent-\d*"
    model_path = re.sub(r'\.trainval_percent-\d*','',model_path)
    # delete ".inner_splits-\d*"
    model_path = re.sub(r'\.inner_splits-\d*','',model_path)
    # replace "l1_weight-[.\de-]+" with "l1_weight-XXX"
    model_path = re.sub(r'l1_weight-[.\de-]+','l1_weight-XXX',model_path)
    model_scores = None
    outerfold_name = 'outerfold' + str(outer_idx)
    for l1_weight in ['1e-05','0.0001','0.001']:
        model_path_l1 = model_path.replace('l1_weight-XXX',f'l1_weight-{l1_weight}')
        # print(model_path_l1)
        if os.path.exists(ANA_SAVE_PATH / model_path_l1):
            # find all folder names in the model_path_l1
            folder_names = [f.name for f in os.scandir(ANA_SAVE_PATH / model_path_l1) if f.is_dir()]
            folder_names = [f for f in folder_names if outerfold_name in f]
            if len(folder_names) == 0:
                continue
            folder_name = folder_names[0]
            the_model_path = ANA_SAVE_PATH / model_path_l1 / folder_name
            model_scores = joblib.load(the_model_path / f'total_scores.pkl')['scores']
            print('Loading model scores from', the_model_path)
            #print(train_data['target'].shape) # (seq_len, batch_size)

            assert len(train_index) == train_data['target'].shape[1] # seq_len, batch_size
            assert len(val_index) == val_data['target'].shape[1]
            train_model_scores = np.zeros((train_data['target'].shape[0], train_data['target'].shape[1],2)) # (seq_len, batch_size,2)
            for i, t in enumerate(train_index):
                model_score = model_scores[t][:-1] # remove the last one for output_h0 = True
                assert train_data['mask'][:,i].sum() == model_score.shape[0], f'{train_data["mask"][:,i].sum()} != {model_score.shape[0]}'
                train_model_scores[:model_score.shape[0], i, :] = model_score
            val_model_scores = np.zeros((val_data['target'].shape[0], val_data['target'].shape[1],2))
            for i, t in enumerate(val_index):
                model_score = model_scores[t][:-1]
                assert val_data['mask'][:,i].sum() == model_score.shape[0], f'{val_data["mask"][:,i].sum()} != {model_score.shape[0]}'
                val_model_scores[:model_score.shape[0], i, :] = model_score
            break
    if model_scores is None:
        raise FileNotFoundError(f'No model score found in {ANA_SAVE_PATH / model_path}')
    train_model_scores = torch.from_numpy(train_model_scores).to(device=config['device'])
    val_model_scores = torch.from_numpy(val_model_scores).to(device=config['device'])
    # print(train_model_scores.shape, val_model_scores.shape)
    # transform score to probability with softmax
    sm = torch.nn.Softmax(dim=2)
    train_model_prob = sm(train_model_scores)
    val_model_prob = sm(val_model_scores)
    return train_model_prob, val_model_prob


def load_teacher_summary(exp_folder, selector):
    with set_os_path_auto():
        teacher_summary = joblib.load(ANA_SAVE_PATH / exp_folder / f'rnn_final_best_summary.pkl')
    for k, v in selector.items():
        if k in teacher_summary.columns:
            teacher_summary = teacher_summary[teacher_summary[k] == v]

    assert len(teacher_summary) == 1
    teacher_summary = teacher_summary.iloc[0]
    teacher_test_indexes = teacher_summary['test_index']
    teacher_model_path = teacher_summary['model_path']
    return teacher_test_indexes, teacher_model_path

# --- Função Principal Adaptada para Slurm ---
def behavior_cv_training_job_slurm(base_config, config_ranges, resource_dict, 
                                     project_root_path, training_script_path, 
                                     python_env_setup, n_jobs=1, 
                                     config_modifier=None, ignore_exist=False):
    """Gera todos os arquivos de config e scripts Slurm para treinamento.

    Args:
        base_config: O arquivo de config base.
        config_ranges: Dicionário de faixas de config.
        resource_dict: Dicionário de recursos {'memory', 'cpu', 'gpu'}.
        project_root_path: Caminho raiz do projeto no cluster Slurm.
        training_script_path: Caminho (relativo à raiz) para o script Python de treino.
        python_env_setup: Comandos shell para configurar o ambiente Python.
        n_jobs: Número de jobs passados como argumento -n (se aplicável).
        config_modifier: Função opcional para modificar configs.
        ignore_exist: Ignorar configs se o resumo já existir.
    """
    # goto_root_dir.run() # Comente ou adapte para o contexto Slurm se necessário.
    
    # --- Você precisa ter 'config_control' e 'MODEL_SAVE_PATH' definidos ---
    # Exemplo (você deve ter o real):
    import sys
    sys.path.append('.') # Adiciona o diretório atual ao path
    import training_experiments.config_control as config_control 
    MODEL_SAVE_PATH = Path('./files/saved_model')
    # --------------------------------------------------------------------

    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    config_paths = []
    
    print(f"Gerando {len(configs)} configurações...")

    for c in configs:
        # Certifique-se que 'model_path' existe na config 'c'
        if 'model_path' not in c:
            # Tenta criar um model_path se não existir (AJUSTE CONFORME NECESSÁRIO)
            c['model_path'] = _standarize_job_name(str(c)).replace('-', '/')
            print(f"Aviso: Gerando model_path: {c['model_path']}")
            
        config_path = Path('./files/saved_model') / c['model_path'] / 'allfold_config.pkl'
        os.makedirs(config_path.parent, exist_ok=True)
        
        if ignore_exist:
            training_summary_path = MODEL_SAVE_PATH / c['model_path'] / 'allfold_summary.pkl'
            if os.path.exists(training_summary_path):
                print(f"Ignorando {c['model_path']}, resumo já existe.")
                continue

        if config_modifier is not None:
            c = config_modifier(c)
            
        joblib.dump(c, config_path)
        config_paths.append(config_path)
        
    if len(config_paths) == 0:
        print('Nenhuma nova configuração gerada.')
        return []

    print(f"{len(config_paths)} novas configurações para processar.")

    # Chame a nova função para gerar scripts Slurm
    sbatch_cmds = write_slurm_scripts(
        config_paths, 
        base_config['exp_folder'], 
        resource_dict, 
        project_root_path,
        training_script_path,
        python_env_setup,
        n_jobs=n_jobs
    )
                                      
    print("\n--- Comandos para submeter os jobs no Slurm (verifique o arquivo .txt) ---")
    for cmd in sbatch_cmds[:5]: # Imprime os 5 primeiros como exemplo
        print(cmd)
    if len(sbatch_cmds) > 5:
        print(f"... e mais {len(sbatch_cmds) - 5} comandos.")
    print("------------------------------------------------------------------------\n")
        
    return sbatch_cmds