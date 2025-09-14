import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",category=mpl.MatplotlibDeprecationWarning)
import numpy as np
import os
from adjustText import adjust_text
from analyzing_experiments.analyzing import pd_full_print_context
from utils import goto_root_dir, highlighted_print
import pathlib
from pathlib import Path
from path_settings import *
import joblib
import pandas as pd
from collections import namedtuple

ModelCurve = namedtuple('ModelCurve', ('name', 'label', 'color', 'alpha', 'marker', 'markersize', 'linewidth', 'linestyle'))

def set_mpl():
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['savefig.dpi'] = 480


def plot_start_scatter_hist(square=True,figsize=None,ticks_pos=True):
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(1.5, 0.8),constrained_layout=True)
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    # ax.set(aspect=1)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig, ax, ax_histx, ax_histy

def infer_color_from_behav(behav): #, include_action=False
    if (behav['action'] == behav['stage2']).all():
        include_action = False
    else:
        include_action = True
    if include_action:
        trial_color = np.char.add('C', (behav['action'] * 4 + behav['stage2'] * 2 + behav['reward']).astype('str'))
    else:
        trial_color = np.char.add('C', (behav['stage2'] * 2 + behav['reward']).astype('str'))
    print('Trial type count',np.unique(trial_color,return_counts=True))
    return trial_color

def control_ticks(x_range,y_range):
    if x_range is None:
        return
    # x_rg = 7
    # y_rg = 7
    plt.xlim([x_range[0]-0.1,x_range[1]+0.1])
    plt.xticks([x_range[0], 0, x_range[1]])
    plt.ylim([y_range[0]-0.1,y_range[1]+0.1])
    plt.yticks([y_range[0], 0, y_range[1]])

def plot_2d_logit_takens(x, trial_types, x_range=None, y_range=None, x_label='', y_label='', title='', ref_line=True,
                         ref_x=0.0, ref_y=0.0,
                         coloring_mapping=None, color_spec=None, labels=None,
                         legend=False, plot_params=None):
    # assume that x and trial_types only come from one block
    y = x[1:]
    trial_types = trial_types[1:]
    x = x[:-1]
    trial_types = trial_types.astype(int)
    unique_trial_types = np.unique(trial_types)
    if coloring_mapping is not None:
        color_spec = []
    elif len(unique_trial_types) == 4:
        labels = ['A1/S1 R=0', 'A1/S1 R=1', 'A2/S2 R=0', 'A2/S2 R=1']
        # color_spec = np.array(['C0', 'C1', 'C5', 'C6'])
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
    elif len(unique_trial_types) == 8:
        labels = ['A1 S1 R=0', 'A1 S1 R=1', 'A1 S2 R=0', 'A1 S2 R=1', 'A2 S1 R=0', 'A2 S1 R=1', 'A2 S2 R=0',
                  'A2 S2 R=1']
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick']) # state coloring
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'tomato', 'firebrick']) # action coloring

    fig, ax = plot_start()
    if plot_params is not None:
        s = plot_params['s']
        alpha = plot_params['alpha']
    else:
        s = 0.5
        alpha = 0.5

    for i in range(len(x) - 1):
        color = color_spec[trial_types[i]] if coloring_mapping is None else coloring_mapping(trial_types[i])
        # ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color=color, alpha=0.5, linewidth=0.5)
        ax.quiver(x[i], y[i], x[i+1]-x[i], y[i+1]-y[i], color=color,
                  angles='xy', scale_units='xy', scale=1, alpha=0.5, width=0.004, headwidth=10, headlength=10)
        # add label for each color
    for i in range(len(color_spec)):
        ax.scatter([], [], color=color_spec[i], label=labels[i])

    if ref_line:
        plt.hlines(ref_y, x_range[0], x_range[1], 'grey', alpha=0.8, linewidth=0.4)
        plt.vlines(ref_x, y_range[0], y_range[1], 'grey', alpha=0.8, linewidth=0.4)

    control_ticks(x_range, y_range)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)



def plot_2d_values(x, y, trial_types, x_range=None, y_range=None, x_label='', y_label='', title='', ref_line=True,
                         ref_x=0.0, ref_y=0.0, ref_diag=False, hist=False, show_dot=True, show_curve=False,
                   coloring_mapping=None, color_spec=None, labels=None,
                   legend=False, plot_params=None):
    trial_types = trial_types.astype(int)
    unique_trial_types = np.unique(trial_types)
    if coloring_mapping is not None:
        color_spec = []
    elif len(unique_trial_types) == 4:
        labels = ['A1/S1 R=0', 'A1/S1 R=1', 'A2/S2 R=0', 'A2/S2 R=1']
        # color_spec = np.array(['C0', 'C1', 'C5', 'C6'])
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
    elif len(unique_trial_types) == 8:
        labels = ['A1 S1 R=0', 'A1 S1 R=1', 'A1 S2 R=0', 'A1 S2 R=1', 'A2 S1 R=0', 'A2 S1 R=1', 'A2 S2 R=0',
                  'A2 S2 R=1']
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick']) # state coloring
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'tomato', 'firebrick']) # action coloring
    if hist:
        fig, ax, ax_histx, ax_histy = plot_start_scatter_hist()
    else:
        fig, ax = plot_start()
    if plot_params is not None:
        s = plot_params['s']
        alpha = plot_params['alpha']
    else:
        s = 0.5
        alpha = 0.5
    if show_dot and coloring_mapping is None:
        ax.scatter(x, y, color=color_spec[trial_types],
                   s=s, alpha=alpha,#facecolors='none', edgecolors=color_spec[trial_types]
                   rasterized=True,
                   )
    elif show_dot and coloring_mapping is not None:
        ax.scatter(x, y, color=coloring_mapping(trial_types),
                   s=s, alpha=alpha,
                   rasterized=True,)
        # add label for each color
        for i in range(len(color_spec)):
            ax.scatter([], [], color=color_spec[i], label=labels[i])
    if show_curve:
        for i in range(len(color_spec)):
            x_ = x[trial_types == i]
            y_ = y[trial_types == i]
            # x_, y_ sort by x_
            sort_idx = np.argsort(x_)
            x_ = x_[sort_idx]
            y_ = y_[sort_idx]
            ax.plot(x_, y_, color=color_spec[i], label=labels[i])
    if ref_line:
        plt.hlines(ref_y, x_range[0], x_range[1], 'grey', alpha=0.8, linewidth=0.4)
        plt.vlines(ref_x, y_range[0], y_range[1], 'grey', alpha=0.8, linewidth=0.4)
    if ref_diag:
        plt.plot([x_range[0], x_range[1]], [y_range[0], y_range[1]], 'k', alpha=0.3)

    control_ticks(x_range, y_range)

    if hist:
        binwidth = 0.05
        x_left, x_right = x.min(), x.max() + binwidth
        y_bottom, y_top = y.min(), y.max() + binwidth
        for tt in np.unique(trial_types):
            ax_histx.hist(x[trial_types == tt], bins=np.arange(x_left, x_right, binwidth), alpha=0.5, color=color_spec[tt],histtype='step')
            ax_histy.hist(y[trial_types == tt], bins=np.arange(y_bottom, y_top, binwidth), alpha=0.5, color=color_spec[tt], orientation='horizontal', histtype='step')
        for ax_hist in [ax_histx, ax_histy]:
            ax_hist.spines["right"].set_visible(False)
            ax_hist.spines["top"].set_visible(False)
            ax_hist.spines["bottom"].set_visible(False)
            ax_hist.spines["left"].set_visible(False)
        # remove tick and tick label
        ax_histx.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_histx.set_yticks([])
        # [ax_histx.spines[loc].set_visible(False) for loc in ['top', 'left', 'right']]
        ax_histy.tick_params(axis="y", labelleft=False, left=False)
        ax_histy.set_xticks([])
        # [ax_histy.spines[loc].set_visible(False) for loc in ['top', 'bottom', 'right']]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)


def plot_perf_curves(rnn_perf, rnn_types=None, model_curves=None, max_hidden_dim=20, perf_type='test_loss'):
    if rnn_types is None:
        rnn_types = pd.unique(rnn_perf['rnn_type'])
    if model_curves is None:
        raise ValueError('model_curves must be provided')
    for rnn_type in rnn_types:
        # if 'finetune' in rnn_perf.columns:
        #     if 'finetune' in rnn_type:
        #         this_rnn_perf = rnn_perf[(rnn_perf['finetune'].isin([True])) & (rnn_perf['rnn_type'] == rnn_type.split('-')[0])]
        #     else:
        #         this_rnn_perf = rnn_perf[(rnn_perf['finetune'] == 'none') & (rnn_perf['rnn_type'] == rnn_type)]
        # else:
        #     this_rnn_perf = rnn_perf[rnn_perf['rnn_type'] == rnn_type]
        if '+' in rnn_type:
            this_rnn_perf = rnn_perf[((rnn_perf['rnn_type']=='GRU') & (rnn_perf['hidden_dim']>1)) | ((rnn_perf['rnn_type']=='SGRU') & (rnn_perf['hidden_dim']==1))]
        else:
            this_rnn_perf = rnn_perf[rnn_perf['rnn_type'] == rnn_type]
            this_rnn_perf = this_rnn_perf[this_rnn_perf['hidden_dim'] <= max_hidden_dim]
        # sort based on hidden_dim
        this_rnn_perf = this_rnn_perf.sort_values(by='hidden_dim')
        hidden_dim = this_rnn_perf['hidden_dim']
        perf = this_rnn_perf[perf_type]
        model_curve = model_curves[rnn_type]
        plt.plot(hidden_dim, perf, label=model_curve.name, color=model_curve.color, alpha=model_curve.alpha,
                 linestyle=model_curve.linestyle, linewidth=model_curve.linewidth,
                 marker=model_curve.marker, markersize=model_curve.markersize)
        if perf_type in ['test_loss', 'test_acc']:
            if perf_type == 'test_loss':
                yerr = this_rnn_perf['test_loss_outer_inner_sem']
                if np.isnan(yerr).any():
                    yerr = this_rnn_perf['test_loss_sub_sem']
            else:
                yerr = this_rnn_perf['test_acc_outer_inner_sem']
                if np.isnan(yerr).any():
                    yerr = this_rnn_perf['test_acc_sub_sem']
            capsize = 2
            capthick = 0.5
            if np.isnan(yerr).any():
                yerr = 0
                capsize = 0
                capthick = 0
            plt.errorbar(hidden_dim, perf, yerr=yerr, color=model_curve.color, alpha=model_curve.alpha,
                         capsize=capsize, capthick=capthick)
            # agg_test_loss_sem= this_rnn_perf['agg_test_loss'].apply(lambda x: np.std(x)/np.sqrt(len(x)))
            # plt.errorbar(hidden_dim, perf, yerr=agg_test_loss_sem, color=model_curve.color, alpha=model_curve.alpha)


# def plot_trial_num_perf(trial_num, perf, model_curve, labels_done):
#     label = model_curve.label
#     if label in labels_done:
#         label = None
#     else:
#         labels_done.append(label)
#     plt.plot(trial_num, perf, label=label, color=model_curve.color, alpha=model_curve.alpha,
#              linestyle=model_curve.linestyle, linewidth=model_curve.linewidth,
#              marker=model_curve.marker, markersize=model_curve.markersize)
#     return labels_done


def transform_xaxis(x, breakpoint=1e8):
    if x <= breakpoint:
        return x
    else:
        return breakpoint + (x - breakpoint) / 3

def plot_trial_num_perf(trial_num, perf, model_curve, labels_done, breakpoint=1e8, yerr=0, capsize=0, capthick=0):
    label = model_curve.label
    if label in labels_done:
        label = None
    else:
        labels_done.append(label)
    # Transform trial_num using the custom transform_x function
    transformed_trial_num = [transform_xaxis(x, breakpoint) for x in trial_num]
    plt.plot(transformed_trial_num, perf, label=label, color=model_curve.color, alpha=model_curve.alpha,
             linestyle=model_curve.linestyle, linewidth=model_curve.linewidth,
             marker=model_curve.marker, markersize=model_curve.markersize)
    if capsize > 0:
        plt.errorbar(transformed_trial_num, perf, yerr=yerr, color=model_curve.color,
                     alpha=model_curve.alpha,
                     capsize=capsize, capthick=capthick)
    return labels_done

def plot_perf_curves_dataprop(model_perf, agent_types=None, model_curves=None, max_hidden_dim=20, perf_type='test_loss'):
    if 'rnn_type' in model_perf.columns:
        perf_col_name = 'rnn_type'
    else:
        perf_col_name = 'cog_type'
        
    if agent_types is None:
        agent_types = pd.unique(model_perf[perf_col_name])
    if model_curves is None:
        raise ValueError('model_curves must be provided')
    with pd_full_print_context():
        print(model_perf)
    labels_done = []
    for agent_type in agent_types:
        this_perf = model_perf[model_perf[perf_col_name] == agent_type]
        this_perf = this_perf[this_perf['hidden_dim'] <= max_hidden_dim]
        # hidden_dim = this_perf['hidden_dim']
        # trainval_percent = this_perf['trainval_percent']
        train_trial_num = this_perf['mean_train_trial_num']
        val_trial_num = this_perf['mean_val_trial_num']
        total_trial_num = train_trial_num + val_trial_num

        perf = this_perf[perf_type]

        yerr = this_perf['test_loss_outer_inner_sem'] if 'test_loss_outer_inner_sem' in this_perf.columns else np.nan
        if np.isnan(yerr).any():
            yerr = this_perf['test_loss_sub_sem'] if 'test_loss_sub_sem' in this_perf.columns else np.nan
        capsize = 2
        capthick = 0.5
        if np.isnan(yerr).all():
            yerr = 0
            capsize = 0
            capthick = 0

        labels_done = plot_trial_num_perf(total_trial_num, perf, model_curves[agent_type], labels_done, yerr=yerr, capsize=capsize, capthick=capthick)

def plot_perf_dots(cog_perf, cog_types=None, cog_ignores=None, model_dots=None, perf_type='test_loss', add_text=False):
    if cog_types is None:
        cog_types = set(pd.unique(cog_perf['cog_type']))
    if cog_ignores is not None:
        cog_types = cog_types - set(cog_ignores)
    texts = []
    labels_done = []
    for cog_type in cog_types:
        this_cog_perf = cog_perf[cog_perf['cog_type'] == cog_type]
        if not(this_cog_perf.empty):
            print(this_cog_perf)
            perf = this_cog_perf[perf_type].values[0]
            hidden_dim = this_cog_perf['hidden_dim'].values[0]
            model_dot = model_dots[cog_type]
            if perf_type=='test_loss' and perf > 1:
                print(f'Warning: test loss {perf} is too large, ignored:', cog_type)
                continue
            label = model_dot.label
            if label in labels_done:
                label = None
            else:
                labels_done.append(label)
            plt.scatter(hidden_dim, perf, color=model_dot.color,
                        # facecolors = 'none', edgecolors=model_dot.color,
                        alpha=model_dot.alpha, marker=model_dot.marker,
                        s=model_dot.markersize, zorder=2, label=label)
            if perf_type in ['test_loss', 'test_acc']:
                if perf_type == 'test_loss':
                    yerr = this_cog_perf['test_loss_outer_inner_sem']
                    if np.isnan(yerr).any():
                        yerr = this_cog_perf['test_loss_sub_sem']
                else:
                    yerr = this_cog_perf['test_acc_outer_inner_sem']
                    if np.isnan(yerr).any():
                        yerr = this_cog_perf['test_acc_sub_sem']
                capsize = 2
                capthick = 0.5
                if np.isnan(yerr).all(): ################
                    yerr = 0
                    capsize = 0
                    capthick = 0
                yerr[yerr > 0.01] = 0
                plt.errorbar(hidden_dim, perf, yerr=yerr, color=model_dot.color,
                            alpha=model_dot.alpha,
                            capsize=capsize, capthick=capthick)
                # agg_test_loss_sem = this_cog_perf['agg_test_loss'].apply(lambda x: np.std(x)/np.sqrt(len(x)))
                # plt.errorbar(hidden_dim, perf, yerr=agg_test_loss_sem, color=model_dot.color, alpha=model_dot.alpha)

            # plt.errorbar(hidden_dim, perf, yerr=this_cog_perf['test_loss_mean_inner_sem'].values[0], color=model_dot.color, alpha=model_dot.alpha)
            if add_text:
                texts.append(plt.text(hidden_dim, perf, model_dot.name, ha='center', va='center', fontsize=5,
                                color=model_dot.color))
                
        else:
            print("Error: the dataframe doesnt exist")
    if add_text:
        ratio=(1.05,4)
        adjust_text(texts, expand_text=ratio,
                    expand_points=ratio,
                    expand_objects=ratio,
                    expand_align=ratio,
                    arrowprops=dict(arrowstyle="-", color='k', lw=0.5))


def set_brief_xlog(xlim=None, xticks=None, minorticks=False):
    if xlim is None:
        xlim = [0.91, 22] # the min value is 1, max value is 20
    if xticks is None:
        xticks = [1, 2, 3, 4, 5, 10, 20]
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # allow us to set xticks with log scale
    if not minorticks:
        plt.minorticks_off() # turn off minor ticks

def set_brief_ylog(ylim=None, yticks=None, minorticks=False):
    ax = plt.gca()
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter()) # allow us to set yticks with log scale
    if not minorticks:
        plt.minorticks_off() # turn off minor ticks

def plot_dim_distribution(exp_folder,save_pdf=True, suffix='',
                          bins=[1,2,3,4,5],
                          xticks=[1.5,2.5,3.5,4.5,5.5],
                          xticks_label=[1,2,3,4,5],
                          yticklabel_every=1,
                          ):
    goto_root_dir.run()
    df = joblib.load(ANA_SAVE_PATH / exp_folder / f'rnn_final_perf_est_dim{suffix}.pkl')
    df = df['dimension'].values
    bin_max = bins[-1]
    df[df > bin_max] = bin_max # set the max value to the last bin
    fig, ax = plot_start(figsize=(1.5, 1.5))
    count, _, _ = plt.hist(df, bins=bins,color='white', linewidth=2, edgecolor='black')
    plt.xticks(xticks, xticks_label)
    plt.yticks(np.arange(0, count.max()+1, yticklabel_every))
    plt.xlabel('Estimated dimensionality')
    plt.ylabel('# Subjects')

    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    fname = f'est_dim{suffix}' + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / fname, bbox_inches="tight")
    plt.show()

def plot_all_model_losses(exp_folder, rnn_types=None, cog_types=None, rnn_filters=None, cog_filters=None, xlim=None, ylim=None,  xticks=None, yticks=None,
                          max_hidden_dim=20, minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', load_file_suffix='', figname='loss_all_models',
                          model_curve_setting=None, add_text=False, save_pdf=False, fname=''):
    if cog_types is None:
        raise ValueError('cog_types must be provided')
    if xlim is None:
        xlim = [0.91, 22]
    if xticks is None:
        xticks = [1, 2, 3, 4, 5, 10, 20]
    if rnn_filters is None:
        rnn_filters = {}
    if cog_filters is None:
        cog_filters = {}
    if figsize is None:
        figsize = (1.5, 1.5)
    assert model_curve_setting is not None

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    if perf_type in ['test_loss','test_acc']:
        rnn_perf = joblib.load(ana_exp_path / f'rnn_final_perf{load_file_suffix}.pkl')
        # if len(rnn_types) == 1 and '+' in rnn_types[0]:
        #     rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf_combine_then_select.pkl')
        cog_perf = joblib.load(ana_exp_path / f'cog_final_perf{load_file_suffix}.pkl')####################
        if 'sub_count' in rnn_perf.columns:
            highlighted_print(rnn_perf, by_key='sub_count')
            highlighted_print(cog_perf, by_key='sub_count')
        else:
            with pd_full_print_context():
                print(rnn_perf)
                print(cog_perf)######################

    elif perf_type in ['mean_R2', 'mean_R2_max', 'population_R2']:
        rnn_perf = joblib.load(ana_exp_path / 'rnn_neuron_decoding_perf_based_on_test.pkl')
        cog_perf = joblib.load(ana_exp_path / 'cog_neuron_decoding_perf_based_on_test.pkl')
    elif perf_type in ['latent_population_R2']:
        rnn_perf = joblib.load(ana_exp_path / f'rnn_neuron_decoding_perf_latent_decode_neuron_latent_value{fname}.pkl')
        cog_perf = joblib.load(ana_exp_path / f'cog_neuron_decoding_perf_latent_decode_neuron_latent_value{fname}.pkl')
        perf_type = 'population_R2' # for compatibility when plotting
    elif perf_type in ['num_params']:
        rnn_perf = joblib.load(ana_exp_path / 'rnn_type_num_params.pkl')
        cog_perf = joblib.load(ana_exp_path / 'cog_type_num_params.pkl')
        with pd_full_print_context():
            print(rnn_perf)
            print(cog_perf)
    else:
        raise ValueError('perf_type not recognized')
    if rnn_types is not None and len(rnn_types) > 0:
        for k, v in rnn_filters.items():
            if k in rnn_perf.columns:
                rnn_perf = rnn_perf[rnn_perf[k] == v]
    for k, v in cog_filters.items():
        cog_perf = cog_perf[cog_perf[k] == v]

    fig, ax = plot_start(figsize=figsize)
    set_brief_xlog(xlim=xlim, xticks=xticks, minorticks=minorticks)
    if perf_type in ['num_params']:
        set_brief_ylog()
    print('Plotting for', exp_folder)
    if rnn_types is not None and len(rnn_types) > 0:
        plot_perf_curves(rnn_perf, rnn_types, model_curve_setting, max_hidden_dim=max_hidden_dim, perf_type=perf_type)
    plot_perf_dots(cog_perf, cog_types, model_dots=model_curve_setting, perf_type=perf_type, add_text=add_text)
    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2' or perf_type == 'mean_R2_max':
        plt.ylabel('R2')
    elif perf_type == 'population_R2':
        plt.ylabel('Population R2')
        plt.hlines(rnn_perf.iloc[0]['population_task_R2'], xlim[0], xlim[1], color='k', linestyle='--', linewidth=1)
    elif perf_type == 'num_params':
        plt.ylabel('# Parameters')
    plt.xlabel('# Dynamical variables (d)')
    if yticks is not None:
        plt.yticks(yticks)
    if ylim is not None:
        plt.ylim(ylim)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    if add_text:
        figname = figname + '_text'
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()

def plot_all_model_losses_dataprop(exp_folder, rnn_types=None, cog_types=None, rnn_filters=None, cog_filters=None, xlim=None, xticks=None,
                          minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models_dataprop',
                                   model_curve_setting=None,
                          save_pdf=False,):
    if rnn_types is None:
        raise ValueError('rnn_types must be provided')
    #if cog_types is None:#####################3
     #   raise ValueError('cog_types must be provided')#############3
    if figsize is None:
        figsize = (1.5, 1.5)
    if rnn_filters is None:
        rnn_filters = {}
    if cog_filters is None:
        cog_filters = {}

    assert perf_type in ['test_loss', 'mean_R2']

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    rnn_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    #cog_perf = joblib.load(ana_exp_path / 'cog_final_perf.pkl')#################
    if rnn_types is not None and len(rnn_types) > 0:
        for k, v in rnn_filters.items():
            if k in rnn_perf.columns:
                rnn_perf = rnn_perf[rnn_perf[k] == v]
    #for k, v in cog_filters.items():#######################
     #   cog_perf = cog_perf[cog_perf[k] == v]#########################33

    fig, ax = plot_start(figsize=figsize)
    # set_brief_xlog(xlim=xlim, xticks=xticks, minorticks=minorticks)
    if xlim is not None:
        plt.xlim(xlim)
    if xticks is not None:
        plt.xticks(xticks)
    print('Plotting for', exp_folder)
    plot_perf_curves_dataprop(rnn_perf, rnn_types, model_curve_setting, perf_type=perf_type)
    #plot_perf_curves_dataprop(cog_perf, cog_types, model_curve_setting, perf_type=perf_type)#############################

    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2':
        plt.ylabel('R2')

    # plt.xlabel('# Trials for training')
    plt.xlabel('# Trials available')
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_all_model_losses_f(exp_folder, finetuned_types=None, finetuned_filters=None, xlim=None, ylim=None,  xticks=None, yticks=None,
                          max_hidden_dim=20, minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', load_file_suffix='', figname='loss_all_models',
                          model_curve_setting=None, add_text=False, save_pdf=False, fname=''):
    if xlim is None:
        xlim = [0.91, 22]
    if xticks is None:
        xticks = [1, 2, 3, 4, 5, 10, 20]
    if finetuned_filters is None:
        finetuned_filters = {}
    if figsize is None:
        figsize = (1.5, 1.5)
    assert model_curve_setting is not None
    print('Rodou')
    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    if perf_type in ['test_loss','test_acc']:
        finetuned_perf = joblib.load(ana_exp_path / f'rnn_final_perf{load_file_suffix}.pkl')
        if 'sub_count' in finetuned_perf.columns:
            highlighted_print(finetuned_perf, by_key='sub_count')
        else:
            with pd_full_print_context():
                print(finetuned_perf)

    elif perf_type in ['mean_R2', 'mean_R2_max', 'population_R2']:
        finetuned_perf = joblib.load(ana_exp_path / 'rnn_neuron_decoding_perf_based_on_test.pkl')
    elif perf_type in ['latent_population_R2']:
        finetuned_perf = joblib.load(ana_exp_path / f'rnn_neuron_decoding_perf_latent_decode_neuron_latent_value{fname}.pkl')
        perf_type = 'population_R2' # for compatibility when plotting
    elif perf_type in ['num_params']:
        finetuned_perf = joblib.load(ana_exp_path / 'rnn_type_num_params.pkl')
        with pd_full_print_context():
            print(finetuned_perf)
    else:
        raise ValueError('perf_type not recognized')
    if finetuned_types is not None and len(finetuned_types) > 0:
        for k, v in finetuned_filters.items():
            if k in finetuned_perf.columns:
                finetuned_perf = finetuned_perf[finetuned_perf[k] == v]


    fig, ax = plot_start(figsize=figsize)
    set_brief_xlog(xlim=xlim, xticks=xticks, minorticks=minorticks)
    if perf_type in ['num_params']:
        set_brief_ylog()
    print('Plotting for', exp_folder)
    if finetuned_types is not None and len(finetuned_types) > 0:
        plot_perf_curves(finetuned_perf, finetuned_types, model_curve_setting, max_hidden_dim=max_hidden_dim, perf_type=perf_type)
    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2' or perf_type == 'mean_R2_max':
        plt.ylabel('R2')
    elif perf_type == 'population_R2':
        plt.ylabel('Population R2')
        plt.hlines(finetuned_perf.iloc[0]['population_task_R2'], xlim[0], xlim[1], color='k', linestyle='--', linewidth=1)
    elif perf_type == 'num_params':
        plt.ylabel('# Parameters')
    plt.xlabel('# Dynamical variables (d)')
    if yticks is not None:
        plt.yticks(yticks)
    if ylim is not None:
        plt.ylim(ylim)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    if add_text:
        figname = figname + '_text'
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()


def plot_all_model_losses_dataprop_f(exp_folder, finetuned_types=None, finetuned_filters=None, xlim=None, xticks=None,
                          minorticks=False, figsize=None, legend=True, perf_type='test_loss', title='', figname='loss_all_models_dataprop',
                                   model_curve_setting=None,
                          save_pdf=False,):
    if finetuned_types is None:
        raise ValueError('rnn_types must be provided')
    if figsize is None:
        figsize = (1.5, 1.5)
    if finetuned_filters is None:
        finetuned_filters = {}

    assert perf_type in ['test_loss', 'mean_R2']

    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    finetuned_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')
    if finetuned_types is not None and len(finetuned_types) > 0:
        for k, v in finetuned_filters.items():
            if k in finetuned_perf.columns:
                finetuned_perf = finetuned_perf[finetuned_perf[k] == v]

    fig, ax = plot_start(figsize=figsize)
    if xlim is not None:
        plt.xlim(xlim)
    if xticks is not None:
        plt.xticks(xticks)
    print('Plotting for', exp_folder)
    plot_perf_curves_dataprop_f(finetuned_perf, finetuned_types, model_curve_setting, perf_type=perf_type)

    if perf_type == 'test_loss':
        plt.ylabel('Negative log likelihood')
    elif perf_type == 'mean_R2':
        plt.ylabel('R2')

    # plt.xlabel('# Trials for training')
    plt.xlabel('# Trials available')
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        # leg.set_title('')
    plt.title(title)
    fig_exp_path = FIG_PATH / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    figname = figname + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_perf_curves_dataprop_f(model_perf, agent_types=None, model_curves=None, max_hidden_dim=20, perf_type='test_loss'):
    perf_col_name = 'model_based'
 
        
    if agent_types is None:
        agent_types = pd.unique(model_perf[perf_col_name])
    if model_curves is None:
        raise ValueError('model_curves must be provided')
    with pd_full_print_context():
        print(model_perf)
    labels_done = []
    for agent_type in agent_types:
        this_perf = model_perf[model_perf[perf_col_name] == agent_type]
        this_perf = this_perf[this_perf['hidden_dim'] <= max_hidden_dim]
        # hidden_dim = this_perf['hidden_dim']
        # trainval_percent = this_perf['trainval_percent']
        train_trial_num = this_perf['mean_train_trial_num']
        val_trial_num = this_perf['mean_val_trial_num']
        total_trial_num = train_trial_num + val_trial_num

        perf = this_perf[perf_type]

        yerr = this_perf['test_loss_outer_inner_sem'] if 'test_loss_outer_inner_sem' in this_perf.columns else np.nan
        if np.isnan(yerr).any():
            yerr = this_perf['test_loss_sub_sem'] if 'test_loss_sub_sem' in this_perf.columns else np.nan
        capsize = 2
        capthick = 0.5
        if np.isnan(yerr).all():
            yerr = 0
            capsize = 0
            capthick = 0

        labels_done = plot_trial_num_perf(total_trial_num, perf, model_curves[agent_type], labels_done, yerr=yerr, capsize=capsize, capthick=capthick)


def plot_test_loss_finetuned_model_based(exp_folder):

    # Lista de modelos de interesse
    model_based_list = ['base', '10_pre_trained', '20_pre_trained', '30_pre_trained', '40_pre_trained', '50_pre_trained',
                        '60_pre_trained', '70_pre_trained', '80_pre_trained', '90_pre_trained', '100_pre_trained']

    # Preparando ambiente
    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    finetuned_perf = joblib.load(ana_exp_path / 'rnn_final_perf.pkl')

    # Filtrar apenas os modelos com trainval_percent == 10
    df_filtered = finetuned_perf[
        (finetuned_perf['trainval_percent'] == 10) &
        (finetuned_perf['model_based'].isin(model_based_list))
    ]

    # Ordenar conforme a lista de modelos
    df_filtered['model_based'] = pd.Categorical(df_filtered['model_based'], categories=model_based_list, ordered=True)
    df_filtered = df_filtered.sort_values('model_based')

    # Obter os valores de test_loss
    test_loss_values = df_filtered['test_loss'].values
    n = [0, 800, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200, 8000]
    # Plot
    fig, ax = plot_start(figsize=(4, 3))
    ax.plot(n, test_loss_values, marker='o', linestyle='-', color='blue')
    ax.set_xlabel('Number pretrained trials')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss vs Number pretrained trials (trainval_percent = 10)')
    ax.set_ylim(min(test_loss_values) * 0.9, max(test_loss_values) * 1.1)
    ax.set_xlim(0, 8500)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




def plot_start(square=True,figsize=None,ticks_pos=True):
    set_mpl()
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    elif square:
        fig = plt.figure(figsize=(1.5, 1.5))
    else:
        fig = plt.figure(figsize=(1.5, 0.8))
    ax = fig.add_axes((0.1,0.1,0.8,0.8))
    if ticks_pos:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    return fig,ax



def plot_initial_loss_for_two_agents(
    exp_folder,
    perf_file_suffix,
    agents_to_plot,
    figsize=(8, 6),
    save_pdf=True
):
    ana_exp_output_path = Path('files/analyzing_results') / exp_folder
    perf_file_basename = f'rnn_final_perf{perf_file_suffix}.pkl'
    perf_file = ana_exp_output_path / perf_file_basename
    print(f"INFO: Tentando carregar e plotar para: {exp_folder}, arquivo: {perf_file_basename}")

    if not perf_file.exists():
        print(f"AVISO: Arquivo de performance não encontrado: {perf_file}. O gráfico não pode ser gerado.")
        return

    try:
        df_perf = joblib.load(perf_file)
        print(f"Arquivo de performance '{perf_file}' carregado com {len(df_perf)} linhas.")
    except Exception as e:
        print(f"Erro ao carregar {perf_file}: {e}.")
        return

    if df_perf.empty:
        print("AVISO: O DataFrame está vazio. O gráfico não será gerado.")
        return

    fig, ax = plot_start(figsize=figsize)

    colors = plt.cm.get_cmap('Paired', len(agents_to_plot))
    model_based_order = ['100_pre_trained', '70_pre_trained', '50_pre_trained', '20_pre_trained']
    x_pos = np.arange(len(model_based_order))

    for i, agent_name in enumerate(agents_to_plot):
        agent_data = df_perf[df_perf['agent_name'] == agent_name].copy()
        
        if agent_data.empty:
            print(f"AVISO: Nenhum dado encontrado para o agente '{agent_name}'.")
            continue

        agent_data['model_based'] = pd.Categorical(agent_data['model_based'], categories=model_based_order, ordered=True)
        agent_data = agent_data.sort_values('model_based')
        
        ax.errorbar(
            x_pos,
            agent_data['test_loss'],
            yerr=agent_data['test_loss_sem'],
            label=agent_name,
            marker=model_curve_setting.get(agent_name, {}).get('marker', 'o'),
            color=colors(i),
            capsize=5
        )

    ax.set_ylabel('Test Loss')
    ax.set_xlabel('Modelos (model_based)')
    ax.set_title('Test Loss Inicial para Agentes Específicos (trainval_percent=0)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_based_order, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(title='Agent')

    plt.tight_layout()
    fig_exp_path = Path('files/figures') / exp_folder
    os.makedirs(fig_exp_path, exist_ok=True)
    figname = f"initial_loss_two_agents_comp.png"
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    print(f"Gráfico salvo em {fig_exp_path / figname}")
    plt.show()

def plot_initial_test_loss_vs_trainval_percent(
    exp_folder,
    perf_file_suffix='_sintetic_initial_test',
    model_based_filter=['sintetic'],
    total_max_trials=8000,
    figsize=(6, 4.5),
    save_pdf=True
):
    ana_exp_output_path = ANA_SAVE_PATH / exp_folder
    perf_file_basename = f'rnn_final_perf{perf_file_suffix}.pkl'
    perf_file = ana_exp_output_path / perf_file_basename
    print(f"INFO: Tentando carregar e plotar para: {exp_folder}, arquivo: {perf_file_basename}")
    df_perf = None
    if perf_file.exists():
        try:
            df_perf = joblib.load(perf_file)
            print(f"Arquivo de performance '{perf_file}' carregado com {len(df_perf)} linhas.")
        except Exception as e:
            print(f"Erro ao carregar {perf_file}: {e}. Usando dados simulados.")
            return
    else:
        print(f"AVISO: Arquivo de performance não encontrado: {perf_file}")
        print("O gráfico não pode ser gerado sem o arquivo de dados.")
        return
    if 'model_based' not in df_perf.columns:
        print(f"ERRO: Coluna 'model_based' não encontrada.")
        return
    
    df_filtered = df_perf[df_perf['model_based'].isin(model_based_filter)].copy()
    if df_filtered.empty:
        print("AVISO: Nenhum dado encontrado após a filtragem. O gráfico não será gerado.")
        return
    fig, ax = plot_start(figsize=figsize)
    unique_models_to_plot = sorted(list(df_filtered['model_based'].unique()))
    colors = plt.cm.get_cmap('viridis', len(unique_models_to_plot))
    for i, model_name in enumerate(unique_models_to_plot):
        model_data = df_filtered[df_filtered['model_based'] == model_name].copy()
        if model_data.empty:
            continue
        for col in ['trainval_percent', 'test_loss', 'test_loss_sem']:
            if col not in model_data.columns:
                if col == 'test_loss_sem':
                    model_data[col] = 0
                    print(f"AVISO: Coluna SEM '{col}' não encontrada para '{model_name}'. Barras de erro serão zero.")
                else:
                    print(f"ERRO: Coluna essencial '{col}' ausente para '{model_name}'.")
                    return
            try:
                model_data[col] = pd.to_numeric(model_data[col])
            except ValueError:
                print(f"ERRO: Não foi possível converter a coluna '{col}' para numérica para o modelo '{model_name}'.")
                return
        model_data.sort_values(by='trainval_percent', inplace=True)
        model_data['num_training_trials'] = (model_data['trainval_percent'] / 100) * total_max_trials
        
        ax.errorbar(model_data['num_training_trials'], model_data['test_loss'],
                    yerr=model_data['test_loss_sem'],
                    marker='o', linestyle='-', capsize=4, elinewidth=1.5, markeredgewidth=1.5,
                    color=colors(i),
                    label=f'{model_name} (Média +/- SEM)')
    ax.set_xlabel(f'Número de Trials de Treino (Total Max = {total_max_trials:,})')
    ax.set_ylabel('Test Loss Inicial (sem treino)')
    ax.set_title(f'Test Loss Inicial vs Número de Trials')
    unique_num_trials = sorted(df_filtered['num_training_trials'].unique())
    if unique_num_trials:
        ax.set_xticks(unique_num_trials)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xticks(rotation=30, ha='right')
        min_x, max_x = min(unique_num_trials), max(unique_num_trials)
        padding_x = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else (0.05 * total_max_trials if total_max_trials > 0 else 500)
        ax.set_xlim(min_x - padding_x if min_x != 0 else - (0.01 * total_max_trials if total_max_trials > 0 else 100), 
                    max_x + padding_x)
    min_y, max_y = df_perf['test_loss'].min(), df_perf['test_loss'].max()
    sem_max = df_perf['test_loss_sem'].max() if 'test_loss_sem' in df_perf.columns else 0
    padding_y = (max_y - min_y) * 0.1 if (max_y - min_y) > 0 else 0.1
    ax.set_ylim(min_y - sem_max - padding_y, max_y + sem_max + padding_y)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig_exp_path = FIG_PATH / Path(exp_folder)
    os.makedirs(fig_exp_path, exist_ok=True)
    figname = "initial_loss_vs_num_trials_all_models" + ('.pdf' if save_pdf else '.png')
    plt.savefig(fig_exp_path / figname, bbox_inches="tight")
    print(f"Gráfico salvo em {fig_exp_path / figname}")
    plt.show()

if __name__ == '__main__':
    pass


