import matplotlib.pyplot as plt
import matplotlib
from utils import goto_root_dir
import joblib
from plotting import *
from path_settings import *
import os

goto_root_dir.run()
exp_folder = 'RTS_agents_millerrat55'
ags = [
        'MB0s_seed0',
        'LS0_seed0',
        'LS1_seed0',
        'MB0_seed0',
        'MB1_seed0',
        # 'MB0md_seed0',
        'RC_seed0',
        'Q(0)_seed0',
        'Q(1)_seed0',
        'm55',
    ]
predictor_included = ['reward-common', 'reward-rare', 'nonreward-common', 'nonreward-rare']
ana_exp_path = ANA_SAVE_PATH / exp_folder
fname = '.'.join(predictor_included)
lag = 10
for ag in ags:
    filename = f'{ag}.lag{lag}.{fname}.pkl'
    lr_results = joblib.load(ana_exp_path / filename)
    coef = lr_results['coef']
    fig, ax = plot_start()
    for j, pred in enumerate(predictor_included):
        plt.plot(range(-lag, 0), coef[:, j], label=pred)
    plt.hlines(0, -lag, 0, color='k', linestyles='dashed')
    plt.xlim([-lag-0.2, -0.2])
    plt.xticks(range(-lag+1, 0, 2))
    plt.xlabel('Lag (trial)')
    plt.ylabel('Logistic coefficient')
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
    leg.set_title('')
    plt.title(ag)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    plt.savefig(fig_exp_path / f'log_coef_{ag}.lag{lag}.{fname}.png', bbox_inches="tight")
    plt.show()
    plt.close()