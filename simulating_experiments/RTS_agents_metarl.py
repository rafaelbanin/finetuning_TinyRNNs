"""
Run simulations of the RTS agents, with parameters fit to the meta-RL agents.
"""
import sys
sys.path.append('..')
from agents import Agent
from tasks import akam_tasks as ts
from utils import *
from path_settings import *

goto_root_dir.run()
base_config = {
      ### dataset info
      'dataset': 'MetaTwoStep',
      'behav_format': 'cog_session',
      'behav_data_spec': {'seed': 0},
      ### model info
      'agent_type': 'RTSCog',
      'cog_type': 'MB0',
      'device': 'cpu',
      'outer_fold': 0,
      'inner_fold': 0,
      'seed': 0,
      'exp_folder': 'exp_metatwostep_seed0',
}
sim_config = {
      'task': 'Akam_RTS',
      'com_prob': 0.8, # common transition probability
      'rew_gen': 'blocks',
      'block_length': 50,
      'rew_probs':[0.2, 0.8],
      'n_blocks': 100,
      'n_trials': 100,
      'sim_seed': 0,
      'sim_exp_name': get_current_file_name(__file__),
      'additional_name': '',
}

sim_exp_name = get_current_file_name(__file__)
# We already have the model simulated for 100 blocks, now we simulate for 200 blocks
for n_blocks in [
    100,
    #200,
    # 400,
    # 800,
                 ]:
    if n_blocks != 100:
        sim_config['n_blocks'] = n_blocks
        sim_config['sim_exp_name'] = sim_exp_name + '_nblocks' + str(n_blocks)

    task = ts.Two_step(com_prob=0.8, rew_gen='blocks',
                     block_length=50, probs=[0.2, 0.8])

    for cog_type in ['MB0s', 'LS0', #'LS1',
                     'MB0', 'MB1',
                     #'MB0md', 'RC',
                     'Q(0)', 'Q(1)'
                     ]:
        model_instance_path = Path(f'cog_type-{cog_type}') / f"outerfold{base_config['outer_fold']}_innerfold{base_config['inner_fold']}_seed{base_config['seed']}"
        config = base_config | {
              'cog_type': cog_type,
              'model_path': Path(base_config['exp_folder']) / model_instance_path
              } | sim_config
        ag = Agent(config['agent_type'], config=config)
        ag.load(config['model_path'])
        params = ag.params
        params[-1] = min(params[-1], 3) # shrink the inverse temperature within the normal animal range
        ag.set_params(params)
        ag.simulate(task, config, save=True)
