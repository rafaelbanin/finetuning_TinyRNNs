"""
This file will analyze the task performance of the simulated agents for millerrat.
"""

from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from analyzing_experiments.analyzing_decoding import *
from utils import goto_root_dir
goto_root_dir.run()

df = []

for sub in ['V', 'W',
            ]:
    row = {'dataset': 'BartoloMonkey', 'sub': sub}
    dt = Dataset('BartoloMonkey',
                         behav_data_spec={'animal_name': sub, 'filter_block_type': 'both', 'block_truncation': (10, 70)}, verbose=False)
    sub_reward = np.concatenate(dt.behav['reward']).mean()
    row['sub'] = sub_reward
    for model in [
            'MB0s', 'LS0',
           'MB0', 'MB1',
           'RC',
        'SGRU_1', 'GRU_2',
                ]:
        dt_model = Dataset('SimAgent',
                     behav_data_spec={'agent_path': [f'allagents_monkey{sub}_nblocks100_ntrials60'],
                                      'agent_name': f'{model}_seed0',
                                      }, verbose=False)
        model_reward = np.concatenate(dt_model.behav['reward']).mean()
        row[model] = model_reward
    df.append(row)

for sub in ['55', '64','70', '71',
            ]:
    row = {'dataset': 'MillerRat', 'sub': sub}
    dt = Dataset('MillerRat',
            behav_data_spec={'animal_name': f'm{sub}', #'m64',#'m55'
                             'max_segment_length': 150,
                             }, verbose=False)
    sub_reward = np.concatenate(dt.behav['reward']).mean()
    row['sub'] = sub_reward
    for model in [   'MB0s', 'LS0', 'LS1', 'MB0', 'MB1', #'MB0md',
                     'RC', 'Q(0)', 'Q(1)',
                    'MXs',
                    'GRU_1', 'GRU_2', 'SGRU_1',
                ]:
        dt_model = Dataset('SimAgent',
                     behav_data_spec={'agent_path': [f'allagents_millerrat{sub}_nblocks200_ntrials500'],
                                      'agent_name': f'{model}_seed0',
                                      }, verbose=False)
        model_reward = np.concatenate(dt_model.behav['reward']).mean()
        row[model] = model_reward
    df.append(row)

df = pd.DataFrame(df)
df.to_csv(ANA_SAVE_PATH / 'model_task_perf.csv')
marker = 'o'
label = 'Monkey'
for i, row in df.iterrows():
    marker = 'o' if row['dataset'] == 'BartoloMonkey' else 'x'
    plt.plot([1,2,3], [row['sub'], row['GRU_2'],row['MB1']], marker=marker, label=label)
plt.ylim([0.5, 0.65])
plt.yticks([0.5, 0.55, 0.6, 0.65])
plt.ylabel('Task performance')
plt.xticks([1,2,3],['Subject', 'GRU', 'MF'])
plt.show()