import numpy as np
import random
import os
import json
import joblib
from path_settings import *
from utils import *

from agents import Agent

def _rnn_update_session(session, n_trials, trial, trial_event):
    """Update session data.

    Args:
        session (dict): Session data.
        n_trials (int): Number of trials in session.
        trial (int): Trial number.
        trial_event (tuple): Trial event.
    """
    session.setdefault('choices', np.zeros(n_trials, dtype=int))[trial] = trial_event[0]
    session.setdefault('second_steps', np.zeros(n_trials, dtype=int))[trial] = trial_event[1]
    session.setdefault('outcomes', np.zeros(n_trials, dtype=int))[trial] = trial_event[2]

def _rnn_pack_trial_event(c, obs, use_one_hot='false'):
    # obs: usually (s, o)
    if use_one_hot == 'false':
        return np.concatenate([(c,), obs]) # [c, s, o]
    elif use_one_hot == 'co': # used in reversal learning
        s, o = obs
        assert c == s
        output = np.zeros(4)
        output[c * 2 + o] = 1
        return output
    elif use_one_hot == 'cso': # used in reduced two-step task
        s, o = obs
        output = np.zeros(8)
        output[c * 4 + s * 2 + o] = 1
        return output
    else:
        raise ValueError('use_one_hot must be false, co, or cso, but got', use_one_hot)

def simulate_exp(ag, task, config, save=True):
    """Simulate the agent's behavior on the task.

    Parameters should be set before this function; otherwise, the default parameters will be used.

    Args:
        task: the task instance.
        config: the config dict.
            n_blocks: number of blocks to simulate.
            n_trials: number of trials in each block.
            sim_seed: random seed for simulation.
            sim_exp_name: the name of the experiment when saving the results.
            additional_name: additional name to add to the file name when saving the results.
        save: whether to save the results to disk.

    Returns:
        A dictionary of simulation results.
    """
    if hasattr(ag, 'cog_type'):
        ag.model_type = ag.cog_type
        print('Simulating agent', ag.model_type, 'with params', ag.params)
        _pack_trial_event = None
        _update_session = None
    else:
        ag.model_type = ag.rnn_type
        print('Simulating agent', ag.model_type)
        _pack_trial_event = _rnn_pack_trial_event
        _update_session = _rnn_update_session

    n_blocks = config['n_blocks']
    n_trials = config['n_trials']
    sim_seed = config['sim_seed']
    sim_exp_name = config['sim_exp_name']
    additional_name = config['additional_name']
    if len(sim_exp_name) == 0 and save:
        raise ValueError('sim_exp_name must be specified if save is True')
    print('n_blocks', n_blocks, 'n_trials', n_trials, 'sim_seed', sim_seed, 'sim_exp_name', sim_exp_name,
          'additional_name', additional_name)

    behav = {}
    behav['action'] = []
    behav['stage2'] = []
    behav['reward'] = []
    if hasattr(ag, 'cog_type'):
        behav['params'] = list(ag.params)
    else:
        behav['params'] = []
    behav['mid_vars'] = []
    np.random.seed(sim_seed)
    random.seed(sim_seed)

    for _ in range(n_blocks):
        use_one_hot = config['use_one_hot'] if ag.model_type in ['SGRU'] else 'false' # false, co, cso
        DVs = ag.simulate(task, n_trials, get_DVs=True, update_session=_update_session, pack_trial_event=_pack_trial_event, use_one_hot=use_one_hot)
        behav['action'].append(DVs['choices'])
        behav['stage2'].append(DVs['second_steps'])
        behav['reward'].append(DVs['outcomes'])
        for k in ['choices', 'second_steps', 'outcomes']:
            DVs.pop(k)
        behav['mid_vars'].append(DVs)
    if save:
        if len(additional_name) and additional_name[0] != '_':
            additional_name = '_' + additional_name
        sim_path = SIM_SAVE_PATH / sim_exp_name
        os.makedirs(sim_path, exist_ok=True)
        fname = f'{ag.model_type}{additional_name}_seed{sim_seed}'
        save_config(config, sim_path, fname+'_config')
        joblib.dump(behav, sim_path / f'{fname}.pkl')
    return behav


def iterate_each_model(iter_model_infos, task, sim_config):
    for model_type, model_additional_name, this_model_summary, summary_cond in iter_model_infos:
        
        print(len(this_model_summary))
        a = summary_cond(this_model_summary)
        print("Verdadeiro: ", a, "   No ver: ", not a)
        if not a:
            with pd_full_print_context():
                print(this_model_summary)
                print('len=',len(this_model_summary))
                raise ValueError('condition not satisfied')
        config = this_model_summary.iloc[0]['config']
        ag = Agent(config['agent_type'], config=config)
        ag.load(config['model_path'])
        simulate_exp(ag, task, config | sim_config | {'additional_name': model_additional_name}, save=True)