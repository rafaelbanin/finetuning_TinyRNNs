import numpy as np
import joblib
import matplotlib.pyplot as plt
from path_settings import *
from plotting import *
def sem(x):
    return np.std(x, 0)/np.sqrt(x.shape[0])
def plot_evidence(total_score_path, plot_every_n=1, alpha=0.2, linewidth=0.5, force_color=None, score_labels=[]):
    result = joblib.load(ANA_SAVE_PATH / total_score_path)
    model_labels = np.array(result['label']) # 0, 1
    all_logits = []
    for block_idx in range(0, len(result['scores'])):
        p = result['scores'][block_idx][1 # 0: action score, 1: model label score
            ][1:] # remove the first trial with no evidence
        logits = p[:, 0] - p[:, 1]
        all_logits.append(logits)
        if block_idx % plot_every_n == 0:
            color = f'C{model_labels[block_idx]}' if force_color is None else force_color
            plt.plot(range(1, len(logits)+1),
                     logits, alpha=alpha, color=color, linewidth=linewidth)
    # padding
    n_trial = max([len(i) for i in all_logits])
    for i in range(len(all_logits)):
        all_logits[i] = np.concatenate([all_logits[i], all_logits[i][-1] * np.ones(n_trial-len(all_logits[i]))])
    all_logits = np.array(all_logits)
    model_logits_avg_dict = {}
    for label in np.unique(model_labels):
        color = f'C{label}' if force_color is None else force_color
        model_logits = all_logits[model_labels==label]
        model_logits_mean = np.mean(model_logits,0)
        model_sem = sem(model_logits)
        model_logits_avg_dict[label] = {'mean': model_logits_mean, 'sem': model_sem}
        plt.plot(range(1, model_logits.shape[-1]+1),
                 model_logits_mean, alpha=1, color=color, linewidth=0.5, label=score_labels[label])
        plt.fill_between(range(1, model_logits.shape[-1]+1), model_logits_mean-model_sem,model_logits_mean+model_sem, alpha=0.8, edgecolor=None, facecolor=color)
    return model_logits_avg_dict

def plot_evidence_for_all_scores(ref_score_path='', sub_score_path='', ref_score_labels=[], sub_score_labels=[],
                                 ref_plot_every_n=5, exp_fig_path='', fig_name='model_evidence',xlim_max=None, ylim_max=None):
    fig, ax = plot_start(square=True)
    model_logits_avg_dict = plot_evidence(ref_score_path, plot_every_n=ref_plot_every_n, alpha=0.1, linewidth=0.2, score_labels=ref_score_labels)
    joblib.dump(model_logits_avg_dict, ANA_SAVE_PATH / Path(ref_score_path).parent / 'model_logits_avg_dict.pkl')
    sub_logits_avg_dict = plot_evidence(sub_score_path, plot_every_n=1, alpha=0.1, linewidth=0.2, force_color='C2', score_labels=sub_score_labels)
    joblib.dump(sub_logits_avg_dict, ANA_SAVE_PATH / Path(sub_score_path).parent / 'sub_logits_avg_dict.pkl')
    plt.hlines(0, 0, plt.xlim()[1], linestyles='dashed', colors='k', alpha=0.5, linewidth=0.5)
    if xlim_max is not None:
        plt.xlim(0, xlim_max)
        plt.xticks()
    if ylim_max is None:
        ylim_max = np.max(np.abs(plt.ylim()))
    plt.ylim(-ylim_max, ylim_max)
    plt.xlabel('Trial')
    plt.ylabel('Model evidence')
    plt.legend()
    os.makedirs(FIG_PATH / exp_fig_path, exist_ok=True)
    plt.savefig(FIG_PATH / exp_fig_path / f'{fig_name}.pdf', bbox_inches='tight')
    plt.show()
