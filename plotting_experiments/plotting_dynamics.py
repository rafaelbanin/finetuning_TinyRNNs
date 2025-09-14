import matplotlib.pyplot as plt

from plotting_experiments.plotting import *
from analyzing_experiments.analyzing_dynamics import *

from utils import goto_root_dir

from statsmodels.stats.contingency_tables import Table2x2
def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)

def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def plt_2d_vector_flow(x1, x1_change, x2, x2_change, color, axis_range, ax=None, arrow_max_num = 200, arrow_alpha = 0.8,
                       plot_n_decimal=1):


    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx]
    ax.quiver(x1, x2, x1_change, x2_change, color=color,
              angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, width=0.004, headwidth=10, headlength=10)
    axis_min, axis_max = axis_range
    # ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--')
    # ax.plot([axis_max, axis_min], [axis_min, axis_max], 'k--')
    # xleft, xright = my_ceil(axis_min, 1), my_floor(axis_max, 1)
    if axis_min < 0 < axis_max:
        axis_abs_max = max(abs(axis_min), abs(axis_max))
        axis_min, axis_max = -axis_abs_max, axis_abs_max
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min,plot_n_decimal), 0, np.round(axis_max,plot_n_decimal)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min,plot_n_decimal), np.round(axis_max,plot_n_decimal)]
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_aspect('equal')


def plt_3d_vector_flow(x1, x1_change, x2, x2_change, x3, x3_change, color, axis_range, ax=None, arrow_max_num=2000, arrow_alpha=0.8):
    if len(x1) > arrow_max_num:
        idx = np.random.choice(len(x1), arrow_max_num, replace=False)
        x1, x1_change, x2, x2_change, x3, x3_change = x1[idx], x1_change[idx], x2[idx], x2_change[idx], x3[idx], x3_change[idx]
    lengths = np.sqrt(x1_change ** 2 + x2_change ** 2 + x3_change ** 2)

    for x1_, y1_, z1_, u1, v1, w1, l in zip(x1, x2, x3, x1_change, x2_change, x3_change, lengths):
        ax.quiver(x1_, y1_, z1_, u1, v1, w1)#, length=l * 0.5)
    ax.scatter(x1, x2, x3, c=color, alpha=1, s=10)
    # ax.quiver(x1, x2, x3,x1_change, x2_change, x3_change,
    #           color=color,angles='xy', scale_units='xy', scale=1, alpha=arrow_alpha, width=0.004, headwidth=10, headlength=10)
    axis_min, axis_max = axis_range
    # ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--')
    # ax.plot([axis_max, axis_min], [axis_min, axis_max], 'k--')
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])
    ax.set_zlim([axis_min, axis_max])
    # xleft, xright = my_ceil(axis_min, 1), my_floor(axis_max, 1)
    if axis_min < 0 < axis_max:
        ticks = [axis_min, 0, axis_max]
        ticklabels = [np.round(axis_min, 1), 0, np.round(axis_max, 1)]
    else:
        ticks = [axis_min, axis_max]
        ticklabels = [np.round(axis_min, 1), np.round(axis_max, 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)
    ax.set_zticklabels(ticklabels)
    ax.set_xlabel('Q(S1)')
    ax.set_ylabel('Q(S2)')
    ax.set_zlabel('Q(S3)')
    ax.set_aspect('equal')

def plt_2d_vector_magnitute(x1, x1_change, x2, x2_change, change_magnitude, change_magnitude_max, ax=None, cbar=True):
    s = int(np.sqrt(len(x1)))
    X_mesh = x1.reshape((s, s))
    Y_mesh = x2.reshape((s, s))
    Z_mesh = change_magnitude.reshape((s, s)) / change_magnitude_max
    # print(X_mesh, Y_mesh, Z_mesh)
    ax.contour(X_mesh, Y_mesh, Z_mesh, levels=20, linewidths=0.5, colors='k', alpha=0.3)
    ctf = ax.contourf(X_mesh, Y_mesh, Z_mesh, levels=20, alpha=0.2, vmin=0, vmax=1)
    # cbar = plt.colorbar()
    if cbar:
        cbar = ax.get_figure().colorbar(ctf, ax=ax)
        change_magnitude_max = np.round(change_magnitude_max, 1)
        cbar.set_label('Speed of dynamics', rotation=270)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([0, change_magnitude_max])

def plt_ev(ev, trial_type,ax):
    value = ev[trial_type]['eigenvalue']
    rvector = ev[trial_type]['eigenvector']
    b = ev[trial_type]['bias']
    if value[0] > value[1]:
        value = value[::-1]
        rvector = rvector[:, ::-1]
    lvector = np.linalg.inv(rvector).T
    # d = np.linalg.inv(vector) @ b
    # d/=np.linalg.norm(d)*2
    for j in range(2):
        if np.isclose(rvector[0, j], lvector[0, j]) and np.isclose(rvector[1, j], lvector[1, j]):
            ax.plot([0, rvector[0, j]], [0, rvector[1, j]], 'w', label=f'ev{value[j]:.3f}', linewidth=1 + j * 2)
        else:
            ax.plot([0, rvector[0, j]], [0, rvector[1, j]], 'w', label=f'rev{value[j]:.3f}', linewidth=1 + j * 2)
            ax.plot([0, lvector[0, j]], [0, lvector[1, j]], 'y', label=f'lev{value[j]:.3f}', linewidth=1 + j * 2)
        # ax.plot([0, d[j]*vector[0,j]], [0, d[j]*vector[1,j]],'y', linewidth=1+j*2)
    # ax.plot([0, b[0]], [0, b[1]], 'y', label=f'b', linewidth=2)
    ax.legend()

def plt_2d_vector_field(model_true_output, model_true_trial_types, model_1step_output=None, model_1step_trial_types=None,
                        readout_vector=None, readout_bias=None, subplot=True, ev=None, title=True, coloring_mapping=None,
                        plot_n_decimal=1):
    x1, x1_change = extract_value_changes(model_true_output, value_type=0)
    x2, x2_change = extract_value_changes(model_true_output, value_type=1)
    axis_max = max(np.max(x1),np.max(x2))
    axis_min = min(np.min(x1),np.min(x2))

    # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick']) # state coloring
    color_spec = np.array(
        ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'tomato',
         'firebrick'])  # action coloring
    if len(model_true_trial_types.shape) == 1:
        unique_trial_types = np.unique(model_true_trial_types)
    elif len(model_true_trial_types.shape) == 2:
        unique_trial_types = np.array([0, 1, 2, 3])
    else:
        raise ValueError('model_true_trial_types should be a vector or a 3D array')
    if len(unique_trial_types) == 4:
        titles = ['A1/S1 R=0', 'A1/S1 R=1', 'A2/S2 R=0', 'A2/S2 R=1']
        row_num, col_num = 4, 1
        locs = [1, 2, 3, 4]
    elif len(unique_trial_types) == 8:
        titles = ['A1,S1,R=0', 'A1,S1,R=1', 'A1,S2,R=0', 'A1,S2,R=1','A2,S1,R=0', 'A2,S1,R=1', 'A2,S2,R=0', 'A2,S2,R=1']
        locs = [1, 3, 2, 4, 5, 7, 6, 8]
        row_num, col_num = 4, 2
    else:
        raise ValueError
    set_mpl()
    if subplot:
        # put all subplots in one figure
        fig = plt.figure(figsize=(4, 8))
        axes = [plt.subplot(row_num, col_num, locs[trial_type]) for trial_type in unique_trial_types]
    else:
        # put each subplot in a separate figure
        axes = [plt.figure(figsize=(2, 2)).gca() for trial_type in unique_trial_types]
    if model_1step_output is not None:
        x1_1step, x1_change_1step = extract_value_changes(model_1step_output, value_type=0)
        x2_1step, x2_change_1step = extract_value_changes(model_1step_output, value_type=1)
        change_magnitude = np.sqrt(x1_change_1step ** 2 + x2_change_1step ** 2)
        change_magnitude_max = np.max(change_magnitude)
    for trial_type in unique_trial_types:
        ax = axes[trial_type]
        if len(model_true_trial_types.shape) == 1:
            idx = model_true_trial_types == trial_type
        elif len(model_true_trial_types.shape) == 2:
            transform_trial_type = (model_true_trial_types[:, 0] * 2 + model_true_trial_types[:, 1]).astype(int)
            idx = transform_trial_type == trial_type
        if model_1step_output is not None:
            if len(model_1step_trial_types.shape) == 1:
                idx_1step = model_1step_trial_types == trial_type
            elif len(model_1step_trial_types.shape) == 2:
                transform_trial_type = (model_1step_trial_types[:, 0] * 2 + model_1step_trial_types[:, 1]).astype(int)
                idx_1step = transform_trial_type == trial_type
            # idx_1step = model_1step_trial_types == trial_type
            plt_2d_vector_magnitute(x1_1step[idx_1step], x1_change_1step[idx_1step],
                                    x2_1step[idx_1step], x2_change_1step[idx_1step],
                                    change_magnitude[idx_1step], change_magnitude_max, ax=ax, cbar=False)
        if coloring_mapping is not None:
            plt_2d_vector_flow(x1[idx], x1_change[idx], x2[idx], x2_change[idx], coloring_mapping(model_true_trial_types[idx]),
                               [axis_min, axis_max], ax=ax, plot_n_decimal=plot_n_decimal)
        else:
            plt_2d_vector_flow(x1[idx], x1_change[idx], x2[idx], x2_change[idx], 'k',#color_spec[trial_type],
                               [axis_min, axis_max], ax=ax, plot_n_decimal=plot_n_decimal)
        if title:
            ax.set_title(titles[trial_type])

        # draw readout vector
        if readout_vector is not None: # w1, w2
            x0, y0 = (axis_min + axis_max) / 2, (axis_min + axis_max) / 2
            w1, w2 = readout_vector
            ax.quiver(x0-w1/2, y0-w2/2, w1, w2, color='darkorange',#'k',
                      angles='xy', scale_units='xy', scale=1, alpha=0.6, headwidth=10, headlength=10)
            # draw decision boundary w1*x1 + w2*x2 + b = 0
            db_x1 = np.linspace(axis_min, axis_max, 100)
            db_x2 = -(w1 * db_x1 + readout_bias) / w2
            ax.plot(db_x1, db_x2, '--',alpha=0.7,color='darkorange',#'k'
                    )

        if ev is not None:
            plt_ev(ev, trial_type, ax)
    return axes


def plt_3d_vector_field_gillan(model_true_output, model_true_trial_types,title=True):
    xs, xs_change = extract_value_changes_gillan(model_true_output, trial_wise=False)
    x1, x2, x3 = xs[0], xs[1], xs[2]
    x1_change, x2_change, x3_change = xs_change[0], xs_change[1], xs_change[2]
    axis_max = max(np.max(x1),np.max(x2),np.max(x3))
    axis_min = min(np.min(x1),np.min(x2),np.min(x3))
    # stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked
    unique_trial_types = [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
                          (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1), (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)]
    titles = [f'A{tt[0]+1}S{tt[1]+2}A{tt[2]+1}R{tt[3]}' for tt in unique_trial_types]
    row_num, col_num = 4, 4
    set_mpl()
    # put all subplots in one figure
    fig = plt.figure(figsize=(15, 15))
    axes = [plt.subplot(row_num, col_num, loc+1, projection='3d') for loc in range(len(unique_trial_types))]
    assert model_true_trial_types.shape[-1] == 4 # (block, trial_num, 4)
    tt_indexing = model_true_trial_types.reshape(-1,4) # (block*trial_num, 4)
    tt_indexing = tt_indexing[:,0]*8 + tt_indexing[:,1]*4 + tt_indexing[:,2]*2 + tt_indexing[:,3] # (block*trial_num,)
    if x1.shape[0] == tt_indexing.shape[0] * 2:
        tt_indexing = np.repeat(tt_indexing[:, np.newaxis], 2, axis=1).flatten()
    elif x1.shape[0] == tt_indexing.shape[0]:
        pass
    else:
        raise ValueError
    for tt_idx, tt in enumerate(unique_trial_types):
        ax = axes[tt_idx]
        idx = tt_indexing == (tt[0]*8 + tt[1]*4 + tt[2]*2 + tt[3])

        plt_3d_vector_flow(x1[idx], x1_change[idx], x2[idx], x2_change[idx], x3[idx], x3_change[idx],
                           'k',#color_spec[trial_type],
                           [axis_min, axis_max], ax=ax)
        if title:
            ax.set_title(titles[tt_idx])

    return axes


def plt_dynamical_regression_gillan(exp_folder, save_pdf=True, model_filters=None, plot_regression=True, plot_perf=True):
    csv_path = ANA_SAVE_PATH / exp_folder / f'dynamical_regression_summary.csv'
    dynamical_regression_summary = pd.read_csv(csv_path)
    if model_filters is not None:
        for key, value in model_filters.items():
            dynamical_regression_summary = dynamical_regression_summary[dynamical_regression_summary[key] == value]

    # stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked
    unique_trial_types = [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
                          (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1), (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)]
    titles = [f'A{tt[0]+1}S{tt[1]+1}{"BC"[tt[1]]}{tt[2]+1}R{tt[3]}' for tt in unique_trial_types]
    # A1/A2 S1/S2 B1/B2/C1/C2 R0/R1

    for i, row in dynamical_regression_summary.iterrows():
        offset = 0.1
        cond_dict = {  # trial type=(stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked):
            # (x base location, x offset, tick label, color name, marker)
            # cornflowerblue for stage_1_selected_stimulus=0 and reward_masked=0
            # mediumblue for stage_1_selected_stimulus=0 and reward_masked=1
            # tomato for stage_1_selected_stimulus=1 and reward_masked=0
            # firebrick for stage_1_selected_stimulus=1 and reward_masked=1
            # -offset and 'o' marker for stage_2_selected_stimulus=0
            # +offset and 's' marker for stage_2_selected_stimulus=1
            (0, 0, 0, 0): (0, -offset*2, 'A1\nS1', 'cornflowerblue', 'o'),
            (0, 0, 0, 1): (0, -offset*1, 'A1\nS1', 'mediumblue', 'o'),
            (0, 0, 1, 0): (0, +offset*1, 'A1\nS1', 'cornflowerblue', 's'),
            (0, 0, 1, 1): (0, +offset*2, 'A1\nS1', 'mediumblue', 's'),
            (0, 1, 0, 0): (2, -offset*2, 'A1\nS2', 'cornflowerblue', 'o'),
            (0, 1, 0, 1): (2, -offset*1, 'A1\nS2', 'mediumblue', 'o'),
            (0, 1, 1, 0): (2, +offset*1, 'A1\nS2', 'cornflowerblue', 's'),
            (0, 1, 1, 1): (2, +offset*2, 'A1\nS2', 'mediumblue', 's'),
            (1, 0, 0, 0): (3, -offset*2, 'A2\nS1', 'tomato', 'o'),
            (1, 0, 0, 1): (3, -offset*1, 'A2\nS1', 'firebrick', 'o'),
            (1, 0, 1, 0): (3, +offset*1, 'A2\nS1', 'tomato', 's'),
            (1, 0, 1, 1): (3, +offset*2, 'A2\nS1', 'firebrick', 's'),
            (1, 1, 0, 0): (1, -offset*2, 'A2\nS2', 'tomato', 'o'),
            (1, 1, 0, 1): (1, -offset*1, 'A2\nS2', 'firebrick', 'o'),
            (1, 1, 1, 0): (1, +offset*1, 'A2\nS2', 'tomato', 's'),
            (1, 1, 1, 1): (1, +offset*2, 'A2\nS2', 'firebrick', 's'),
        }
        fig_exp_path = FIG_PATH / exp_folder / 'dynamical_regression' / f'subblock-{row["block"]}'
        os.makedirs(fig_exp_path, exist_ok=True)
        if 'rnn_type' in row and not pd.isna(row['rnn_type']):
            if row['hidden_dim'] == 3:
                model_name = row['rnn_type']
            else:
                model_name = row['rnn_type'] + str(row['hidden_dim'])
        else:
            assert 'cog_type' in row and not pd.isna(row['cog_type'])
            model_name = row['cog_type']

        for target in ['x1_change', 'x2_change', 'x3_change']:
            if plot_perf:
                fig, ax = plot_start(figsize=(1.5, 1))
                test_loss = row['test_loss']
                plt.bar(0, test_loss, color='k', alpha=0.4)
                plt.text(0, test_loss, f'{test_loss:.2f}', ha='center', va='bottom')
                plt.ylabel('Test loss')
                figname = f'{model_name}_{target}_perf' + ('.pdf' if save_pdf else '.png')
                fig.savefig(fig_exp_path / figname, bbox_inches="tight")
                print('Save figure to', fig_exp_path / figname)
                plt.close(fig)
            if plot_regression:
                for beta in ['b0', 'b1', 'b2', 'b3','score']:
                    fig, ax = plot_start(figsize=(1.5, 1))
                    # max_v = np.max([np.abs(np.mean(row[ticklabels[i]])) for i in range(num_trial_type)])
                    value_list = [row[''.join(str(i) for i in tt) + '_' + target + '_' + beta] for tt in unique_trial_types]
                    plot_ticklabels = [''] * 4
                    if beta == 'n_sample':
                        plt.ylabel('#samples')
                        plt.ylim([0, np.max(value_list)*1.1])
                    elif beta == 'R2':
                        plt.ylabel('R2')
                        plt.ylim([-0.1, 1.1])
                        plt.yticks([0, 0.5, 1])
                    else:
                        plt.hlines([-1, 0, 1], -0.3, 4 - 0.7, 'grey', alpha=0.5, linestyles='dashed',
                                   linewidth=0.5)
                        y_max = np.max(np.abs(value_list))
                        if y_max == 0:
                            y_max = 1
                        plt.ylim([-y_max*1.1, y_max*1.1])
                        plt.yticks([-y_max, 0, y_max])
                        plt.ylabel(f'{target}_{beta}\nPrefer A2          Prefer A1')
                    for tt, point_y in zip(unique_trial_types, value_list):
                        x, offset, tick_label, color, marker = cond_dict[tt]
                        plot_ticklabels[x] = tick_label

                        plt.scatter(x+offset, point_y, s=5, color=color, marker=marker)  # , facecolors='none')
                        # plt.errorbar(x, points_mean, yerr=np.std(points), color=color, capsize=2)  # , label=label)
                    plt.xticks(np.arange(4), plot_ticklabels)
                    plt.xlim([-0.3, 4 - 0.7])
                    plt.xlabel('Common          Rare')

                    figname = f'{model_name}_{target}_{beta}' + ('.pdf' if save_pdf else '.png')
                    fig.savefig(fig_exp_path / figname, bbox_inches="tight")
                    print('Save figure to', fig_exp_path / figname)
                    plt.close(fig)


def plt_dynamical_regression_bahrami(exp_folder, save_pdf=True, model_filters=None, plot_regression=True, plot_perf=True):
    csv_path = ANA_SAVE_PATH / exp_folder / f'dynamical_regression_summary.csv'
    dynamical_regression_summary = pd.read_csv(csv_path)
    if model_filters is not None:
        for key, value in model_filters.items():
            dynamical_regression_summary = dynamical_regression_summary[dynamical_regression_summary[key] == value]

    unique_trial_types = [0, 1, 2, 3] # only consider the 4 actions
    trial_type_names = ['A1', 'A2', 'A3', 'A4']
    cond_dict = {  # trial type=(stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked):
        # (x base location, x offset, tick label, color name, marker)
        # cornflowerblue for stage_1_selected_stimulus=0 and reward_masked=0
        # mediumblue for stage_1_selected_stimulus=0 and reward_masked=1
        # tomato for stage_1_selected_stimulus=1 and reward_masked=0
        # firebrick for stage_1_selected_stimulus=1 and reward_masked=1
        # -offset and 'o' marker for stage_2_selected_stimulus=0
        # +offset and 's' marker for stage_2_selected_stimulus=1
        0: (0, 0, 'A1', 'mediumblue', 'o'),
        1: (1, 0, 'A2', 'firebrick', 'o'),
        2: (2, 0, 'A3', 'green', 'o'),
        3: (3, 0, 'A4', 'gold', 'o'),

    }
    for i, row in dynamical_regression_summary.iterrows():

        fig_exp_path = FIG_PATH / exp_folder / 'dynamical_regression' / f'subblock-{row["block"]}'
        os.makedirs(fig_exp_path, exist_ok=True)
        if 'rnn_type' in row and not pd.isna(row['rnn_type']):
            model_name = row['rnn_type']
        else:
            assert 'cog_type' in row and not pd.isna(row['cog_type'])
            model_name = row['cog_type']

        for beta in ['b0_corrected','b0', 'b1', 'b2', 'b3','b4','br','score']:
            if plot_perf:
                fig, ax = plot_start(figsize=(3, 1))
                test_loss = row['test_loss']
                plt.bar(0, test_loss, color='k', alpha=0.4)
                plt.text(0, test_loss, f'{test_loss:.2f}', ha='center', va='bottom')
                plt.ylabel('Test loss')
                figname = f'{model_name}_perf' + ('.pdf' if save_pdf else '.png')
                fig.savefig(fig_exp_path / figname, bbox_inches="tight")
                print('Save figure to', fig_exp_path / figname)
                plt.close(fig)

            target_names = ['x1_change', 'x2_change', 'x3_change', 'x4_change']
            if plot_regression:
                fig, ax = plot_start(figsize=(2, 1))
                # max_v = np.max([np.abs(np.mean(row[ticklabels[i]])) for i in range(num_trial_type)])
                value_list = np.array([[row[ttn + '_' + target + '_' + beta] for ttn in trial_type_names] for target in target_names ])
                plot_ticklabels = [''] * 4
                if beta == 'score':
                    plt.ylabel('R2')
                    plt.ylim([-0.1, 1.1])
                    plt.yticks([0, 0.5, 1])
                else:
                    y_max = np.max(np.abs(value_list))
                    if y_max < 1e-5:
                        y_max = 1
                    plt.ylim([-y_max*1.1, y_max*1.1])
                    plt.hlines([-y_max, 0, y_max], -0.3, 16 - 0.7, 'grey', alpha=0.5, linestyles='dashed',
                               linewidth=0.5)
                    plt.yticks([-y_max, 0, y_max])
                    plt.ylabel(f'{beta}')
                for target_idx,target in enumerate(target_names):
                    for tt in unique_trial_types:
                        x, offset, tick_label, color, marker = cond_dict[tt]
                        tick_idx = target_idx*4 + x
                        # plot_ticklabels[target_idx] = [r'$\Delta L_1$', r'$\Delta L_2$', r'$\Delta L_3$', r'$\Delta L_4$'
                        #                              ][target_idx] #+ '\n' + tick_label

                        plt.scatter(target_idx + 0.15*x - 0.225, value_list[target_idx][tt], s=5, color=color, marker=marker)  # , facecolors='none')
                        # plt.errorbar(x, points_mean, yerr=np.std(points), color=color, capsize=2)  # , label=label)
                plot_ticklabels = [r'$\Delta L_1$', r'$\Delta L_2$', r'$\Delta L_3$', r'$\Delta L_4$']
                plt.xticks(np.arange(4), plot_ticklabels)
                plt.xlim([-0.3, 4 - 0.7])
                figname = f'{model_name}_{beta}' + ('.pdf' if save_pdf else '.png')
                fig.savefig(fig_exp_path / figname, bbox_inches="tight")
                print('Save figure to', fig_exp_path / figname)
                plt.close(fig)

def wrap_violin(x, pos, width, color):
    violin = plt.violinplot(x, positions=[pos], showextrema=False, showmedians=False,
                            quantiles=[0.25, 0.5, 0.75], widths=width)
    # print(violin.keys())
    # Make all the violin statistics marks grey
    for partname, c in zip(('cquantiles',),
                           ('k',)):
        vp = violin[partname]
        vp.set_edgecolor('k')
        vp.set_linewidth(0.8)
        vp.set_alpha(0.5)

    # Make the violin body blue with a red border:
    for vp in violin['bodies']:
        vp.set_facecolor(color)
        vp.set_edgecolor(color)
        vp.set_linewidth(1)
        vp.set_alpha(1)
    return violin

def detect_column_outliers(df, percent = 1):
    indexes = []
    for col in df.columns:
        col_dt = df[col]
        percentiles = np.percentile(col_dt, [percent, 100 - percent])
        indexes += list(col_dt[(col_dt < percentiles[0]) | (col_dt > percentiles[1])].index)
    return np.unique(indexes)


def plt_dynamical_regression_violin_gillan(exp_folder, save_pdf=True, ignore_poor_score=False, model_filters=None,**kwargs):
    from sigfig import round
    csv_path = ANA_SAVE_PATH / exp_folder / f'dynamical_regression_summary.csv'
    dynamical_regression_summary = pd.read_csv(csv_path)

    if ignore_poor_score:
        dynamical_regression_summary = dynamical_regression_summary[~dynamical_regression_summary['poor_score_flag']].reset_index(drop=True)
    # stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked
    unique_trial_types = [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
                          (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1), (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)]
    titles = [f'A{tt[0]+1}S{tt[1]+1}{"BC"[tt[1]]}{tt[2]+1}R{tt[3]}' for tt in unique_trial_types]
    # A1/A2 S1/S2 B1/B2/C1/C2 R0/R1


    base_loc_scale = 1
    offset = 0.17
    cond_dict = {  # trial type=(stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked):
        # (x base location, x offset, tick label, color name, marker)
        # cornflowerblue for stage_1_selected_stimulus=0 and reward_masked=0
        # mediumblue for stage_1_selected_stimulus=0 and reward_masked=1
        # tomato for stage_1_selected_stimulus=1 and reward_masked=0
        # firebrick for stage_1_selected_stimulus=1 and reward_masked=1
        # -offset and 'o' marker for stage_2_selected_stimulus=0
        # +offset and 's' marker for stage_2_selected_stimulus=1
        (0, 0, 0, 0): (0, -offset * 2, 'A1\nS1\nB1\nR0', 'cornflowerblue', 'o'),
        (0, 0, 0, 1): (0, -offset * 1, 'A1\nS1\nB1\nR1', 'mediumblue', 'o'),
        (0, 0, 1, 0): (0, +offset * 1, 'A1\nS1\nB2\nR0', 'cornflowerblue', 's'),
        (0, 0, 1, 1): (0, +offset * 2, 'A1\nS1\nB2\nR1', 'mediumblue', 's'),
        (0, 1, 0, 0): (2, -offset * 2, 'A1\nS2\nC1\nR0', 'cornflowerblue', 'o'),
        (0, 1, 0, 1): (2, -offset * 1, 'A1\nS2\nC1\nR1', 'mediumblue', 'o'),
        (0, 1, 1, 0): (2, +offset * 1, 'A1\nS2\nC2\nR0', 'cornflowerblue', 's'),
        (0, 1, 1, 1): (2, +offset * 2, 'A1\nS2\nC2\nR1', 'mediumblue', 's'),
        (1, 0, 0, 0): (3, -offset * 2, 'A2\nS1\nB1\nR0', 'tomato', 'o'),
        (1, 0, 0, 1): (3, -offset * 1, 'A2\nS1\nB1\nR1', 'firebrick', 'o'),
        (1, 0, 1, 0): (3, +offset * 1, 'A2\nS1\nB2\nR0', 'tomato', 's'),
        (1, 0, 1, 1): (3, +offset * 2, 'A2\nS1\nB2\nR1', 'firebrick', 's'),
        (1, 1, 0, 0): (1, -offset * 2, 'A2\nS2\nC1\nR0', 'tomato', 'o'),
        (1, 1, 0, 1): (1, -offset * 1, 'A2\nS2\nC1\nR1', 'firebrick', 'o'),
        (1, 1, 1, 0): (1, +offset * 1, 'A2\nS2\nC2\nR0', 'tomato', 's'),
        (1, 1, 1, 1): (1, +offset * 2, 'A2\nS2\nC2\nR1', 'firebrick', 's'),
    }

    # cols_name = [''.join(str(i) for i in tt) + '_' + target + '_' + beta
    #              for target in ['x1_change', 'x2_change', 'x3_change']
    #              for beta in ['b0', 'b1', 'b2', 'b3']
    #              for tt in unique_trial_types
    #              ]
    fname_prefix = ''
    if model_filters is not None:
        for key, value in model_filters.items():
            dynamical_regression_summary = dynamical_regression_summary[dynamical_regression_summary[key] == value]
            fname_prefix += f'{key}-{value}_'
    dynamical_regression_summary = dynamical_regression_summary.reset_index(drop=True)

    for target in ['x1_change', 'x2_change', 'x3_change']:
        for focus_beta, beta_label in zip(['b0','b1','b2', 'b3'],
                              [r'$\beta_0$',r'$\beta_{L_1}$',r'$\beta_{L_2}$',r'$\beta_{L_3}$']):
            cols_name = [''.join(str(i) for i in tt) + '_' + target + '_' + focus_beta
                            for tt in unique_trial_types
                         ] # 16
            print(cols_name)
            X = dynamical_regression_summary[cols_name]
            percent = 5 if 'percent' not in kwargs else kwargs['percent']
            outlier_idx = detect_column_outliers(X, percent=percent)
            print('X.shape', X.shape)
            print('Total:', len(outlier_idx), 'Outlier', outlier_idx)
            with pd_full_print_context():
                print('Outlier\n', X.iloc[outlier_idx])
            # delete outlier_idx rows
            X.drop(outlier_idx, axis=0, inplace=True)

            with pd_full_print_context():
                print(X.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9,0.95,0.99]))
            X = X.values
            # X = np.delete(X, outlier_idx, axis=0)
            # if focus_beta in ['b0','b2', 'b3' ]:
            #     X /= np.max(np.abs(X), axis=1)[..., np.newaxis]
            # fill nan with median
            # for col in range(X.shape[1]):
            #     nan_idx = np.isnan(X[:, col])
            #     if np.any(nan_idx):
            #         X[nan_idx, col] = np.nanmedian(X[:, col])


            fig, ax = plot_start(figsize=(4, 1))
            violins = []
            xticks = []
            xticklabels = []

            for tt_idx, tt in enumerate(unique_trial_types):
                x = X[:, tt_idx]
                base, base_offset, tick_label, color, marker = cond_dict[tt]
                pos = base * base_loc_scale + base_offset
                xticks.append(pos)
                xticklabels.append(tick_label)
                # df = pd.DataFrame(dict(x=[tt_idx]*len(x),
                #                        y=x))
                # sns.violinplot(x="x", y="y", data=df,
                #     color=color, ax=ax, inner='quartile', linewidth=0.5, width=0.7)
                violin = wrap_violin(x, pos, offset, color)
                violins.append(violin)

            # sort xticks and xticklabels based on xticks
            xticks = np.array(xticks)
            xticklabels = np.array(xticklabels)
            idx = np.argsort(xticks)
            xticks = xticks[idx]
            xticklabels = xticklabels[idx]

            xticks_brief = (xticks[::2] + xticks[1::2]) / 2
            xticklabels_brief = [l[:-3] for l in xticklabels[::2]] # remove '\nR0' or '\nR1'
            plt.xticks(xticks_brief, xticklabels_brief)
            plt.ylabel(f'{beta_label}')
            ymax = np.max(np.abs(X))
            ymax = max(ymax, 1)
            ymax = round(ymax, sigfigs=2)
            plt.hlines([-ymax, 0, ymax], -0.3, 4 * base_loc_scale - 0.7,
                       'grey', alpha=0.5, linestyles='dashed',
                        linewidth=0.5)
            plt.ylim([-1.1 * ymax, 1.1 * ymax])
            plt.yticks([-ymax, 0, ymax])

            fig_exp_path = FIG_PATH / exp_folder / 'dynamical_regression_group'
            os.makedirs(fig_exp_path, exist_ok=True)
            figname = f'{fname_prefix}violin_{target}_{focus_beta}' + ('.pdf' if save_pdf else '.png')
            fig.savefig(fig_exp_path / figname, bbox_inches="tight")
            print('Save figure to', fig_exp_path / figname)
            plt.close(fig)

            # corr = np.corrcoef(X.T)
            # if not any(np.isnan(corr.flatten())):
            #     corr = corr[idx, :][:, idx] # sort corr based on xticks
            #     corr, new_idx = cluster_corr(corr)
            #     xticklabels_new = xticklabels[new_idx]
            #     fig, ax = plot_start(figsize=(4, 4))
            #     plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
            #     plt.colorbar()
            #     plt.xticks(np.arange(len(unique_trial_types)), xticklabels_new)
            #     plt.yticks(np.arange(len(unique_trial_types)), [l.replace('\n',' ') for l in xticklabels_new])
            #     plt.savefig(fig_exp_path / (f'{fname_prefix}corr_{target}_{focus_beta}' + ('.pdf' if save_pdf else '.png')), bbox_inches="tight")
            #     plt.close(fig)


def plt_dynamical_regression_violin_multiarm(exp_folder, save_pdf=True, model_filters=None,
                                             n_arm=4,
                                             unique_trial_types = [0, 1, 2, 3], # only consider the 4 actions
                                            trial_type_names = ['A1', 'A2', 'A3', 'A4'],
                                             plot_width=0.15,
                                            all_targets = ['x1_change', 'x2_change', 'x3_change', 'x4_change'], # dependent variable
                                            all_beta = ['b0_corrected','b0', 'b1', 'b2', 'b3','b4','br',], # independent variable
                                            beta_labels=[r'$\beta_0$',r'$\beta_0$',r'$\beta_{L_1}$',r'$\beta_{L_2}$',r'$\beta_{L_3}$',r'$\beta_{L_4}$',r'$\beta_{R}$'],
                                            color_code=['mediumblue', 'firebrick', 'green', 'gold'],
                                            figsize=(3, 1),
                                            plot_ticklabels = [r'$\Delta L_1$', r'$\Delta L_2$', r'$\Delta L_3$', r'$\Delta L_4$'],
                                             hline_edge=0.3,
                                             dist_type='violin',
                                             ignore_poor_score=False,
                                             **kwargs
                                             ):
    from sigfig import round
    n_target = len(all_targets)
    n_trial_types = len(unique_trial_types)
    csv_path = ANA_SAVE_PATH / exp_folder / f'dynamical_regression_summary.csv'
    dynamical_regression_summary = pd.read_csv(csv_path)
    dynamical_regression_summary = dynamical_regression_summary[dynamical_regression_summary['hidden_dim'] == n_arm]
    if ignore_poor_score:
        dynamical_regression_summary = dynamical_regression_summary[~dynamical_regression_summary['poor_score_flag']]
    fname_prefix = ''
    if model_filters is not None:
        for key, value in model_filters.items():
            dynamical_regression_summary = dynamical_regression_summary[dynamical_regression_summary[key] == value]
            fname_prefix += f'{key}-{value}_'
    dynamical_regression_summary = dynamical_regression_summary.reset_index(drop=True)

    # removing outliers based on cols_name values
    # cols_name = [ttn + '_' + target + '_' + beta
    #              for target in all_targets
    #              for beta in all_beta
    #              for ttn in trial_type_names
    #              ]

    for focus_beta, beta_label in zip(all_beta, beta_labels):
        cols_name = sum([
                            [ttn + '_' + target + '_' + beta for ttn in trial_type_names]
                            for beta in [focus_beta]
                            for target in all_targets
                     ], []) # n_target * n_trial_types
        xlocs = sum([
                            [target_idx + plot_width*x - plot_width * (n_trial_types - 1)/2 for x in unique_trial_types]
                 for target_idx in range(len(all_targets))
                    ], []) # n_target * n_trial_types
        colors = sum([
                            [color_code[x] for x in unique_trial_types]
                            for target in all_targets
                    ], [])
        print(cols_name)
        X = dynamical_regression_summary[cols_name]
        percent = 5 if 'percent' not in kwargs else kwargs['percent']
        outlier_idx = detect_column_outliers(X, percent=percent)
        print('X.shape', X.shape)
        print('Total:', len(outlier_idx), 'Outlier', outlier_idx)
        with pd_full_print_context():
            print('Outlier\n', X.iloc[outlier_idx])
        # delete outlier_idx rows
        X.drop(outlier_idx, axis=0, inplace=True)

        with pd_full_print_context():
            print(X.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
        X = X.values

        # if focus_beta in ['b0','b0_corrected','br']:
        #     if np.max(np.abs(X))>1e-3:
        #         X /= np.max(np.abs(X), axis=1)[..., np.newaxis]

        fig, ax = plot_start(figsize=figsize)
        # violins = []

        for col_idx, col in enumerate(cols_name):
            x = X[:, col_idx]
            pos = xlocs[col_idx]
            color = colors[col_idx]
            if dist_type == 'violin':
                violin = wrap_violin(x, pos, plot_width, color)
            elif dist_type in ['std', 'sem', 'none']:
                x_mean = np.mean(x)
                plt.scatter(pos, x_mean, s=5, color=color, marker='o')
                if dist_type == 'std':
                    plt.errorbar(pos, x_mean, yerr=np.std(x, ddof=1), color=color, capsize=2)  # , label=label)
                elif dist_type == 'sem':
                    plt.errorbar(pos, x_mean, yerr=np.std(x, ddof=1)/np.sqrt(len(x)), color=color, capsize=2)
            else:
                raise NotImplementedError
            # violins.append(violin)

        n_labels = len(plot_ticklabels)

        plt.xticks(np.arange(n_labels), plot_ticklabels)
        plt.xlim([-hline_edge, n_labels - 1 + hline_edge])
        plt.ylabel(f'{beta_label}')
        ymax = np.max(np.abs(X))
        ymax = max(ymax, 1)
        ymax = round(ymax, sigfigs=2)
        plt.hlines([-ymax, 0, ymax], -hline_edge, n_labels - 1 + hline_edge,
                   'grey', alpha=0.5, linestyles='dashed',
                    linewidth=0.5)
        plt.ylim([-1.1 * ymax, 1.1 * ymax])
        plt.yticks([-ymax, 0, ymax])

        fig_exp_path = FIG_PATH / exp_folder / 'dynamical_regression_group'
        os.makedirs(fig_exp_path, exist_ok=True)
        figname = fname_prefix + f'violin_{focus_beta}' + ('.pdf' if save_pdf else '.png')
        fig.savefig(fig_exp_path / figname, bbox_inches="tight")
        print('Save figure to', fig_exp_path / figname)
        plt.close(fig)

def plt_dynamical_regression_violin_bartolo(exp_folder, save_pdf=True, model_filters=None, percent=5, hidden_dim=2):
    if hidden_dim == 2:
        plt_dynamical_regression_violin_multiarm(exp_folder, save_pdf=save_pdf, model_filters=model_filters,
                                                 n_arm=2,
                                                 unique_trial_types=[0, 1, 2, 3],  # only consider the 4 actions
                                                 trial_type_names=['A1R0','A1R1', 'A2R0', 'A2R1'],
                                                 plot_width=0.15,
                                                 all_targets=['x1_change', 'x2_change'],
                                                 # dependent variable
                                                 all_beta=['b0_corrected', 'b0', 'b1', 'b2', ],
                                                 beta_labels=[r'$\beta_0$', r'$\beta_0$', r'$\beta_{L1}$', r'$\beta_{L2}$'],
                                                 # independent variable
                                                 color_code=['cornflowerblue', 'mediumblue', 'tomato', 'firebrick',],
                                                 figsize=(1.5, 0.8),
                                                 plot_ticklabels=[r'$\Delta L_1$', r'$\Delta L_2$'],
                                                 hline_edge=0.3,
                                                 dist_type='std', # for one subject
                                                 percent=percent,
                                                 )
    elif hidden_dim == 1:
        plt_dynamical_regression_violin_multiarm(exp_folder, save_pdf=save_pdf, model_filters=model_filters,
                                                 n_arm=1,
                                                 unique_trial_types=[0, 1, 2, 3],  # only consider the 4 actions
                                                 trial_type_names=['A1R0','A1R1', 'A2R0', 'A2R1'],
                                                 plot_width=0.15,
                                                 all_targets=['logit_change'],
                                                 # dependent variable
                                                 all_beta=['b0', 'b1'],
                                                 beta_labels=[r'$\beta_0$', r'$\beta_{L}$'],
                                                 # independent variable
                                                 color_code=['cornflowerblue', 'mediumblue', 'tomato', 'firebrick',],
                                                 figsize=(1.5, 0.8),
                                                 plot_ticklabels=[r'$\Delta L$'],
                                                 hline_edge=0.3,
                                                 dist_type='std', # for one subject
                                                 percent=percent,
                                                 )

def plt_dynamical_regression_violin_bahrami(exp_folder, save_pdf=True, model_filters=None, percent=5):
    plt_dynamical_regression_violin_multiarm(exp_folder, save_pdf=save_pdf, model_filters=model_filters,
                                             n_arm=4,
                                             unique_trial_types=[0, 1, 2, 3],  # only consider the 4 actions
                                             trial_type_names=['A1', 'A2', 'A3', 'A4'],
                                             plot_width=0.15,
                                             all_targets=['x1_change', 'x2_change', 'x3_change', 'x4_change'],
                                             # dependent variable
                                             all_beta=['b0_corrected', 'b0', 'b1', 'b2', 'b3', 'b4', 'br', ],
                                             beta_labels=[r'$\beta_0$', r'$\beta_0$', r'$\beta_{L1}$', r'$\beta_{L2}$', r'$\beta_{L3}$', r'$\beta_{L4}$', r'$\beta_{R}$'],
                                             # independent variable
                                             color_code=['mediumblue', 'firebrick', 'green', 'gold'],
                                             figsize=(3, 1),
                                             plot_ticklabels=[r'$\Delta L_1$', r'$\Delta L_2$', r'$\Delta L_3$',
                                                              r'$\Delta L_4$'],
                                             hline_edge=0.3,
                                             percent=percent,
                                             )



def plt_dynamical_regression_violin_suthaharan(exp_folder, save_pdf=True, model_filters=None, percent=5):
    plt_dynamical_regression_violin_multiarm(exp_folder, save_pdf=save_pdf, model_filters=model_filters,
                                             n_arm=3,
                                             unique_trial_types=np.arange(6),  # only consider the 4 actions
                                             trial_type_names=['A1R0', 'A1R1', 'A2R0','A2R1', 'A3R0', 'A3R1'],
                                             plot_width=0.12,
                                             all_targets=['x1_change', 'x2_change', 'x3_change'],
                                             # dependent variable
                                             all_beta=['b0_corrected','b0', 'b1', 'b2', 'b3'],
                                             beta_labels=[r'$\beta_0$', r'$\beta_0$', r'$\beta_{L1}$', r'$\beta_{L2}$', r'$\beta_{L3}$'],
                                             color_code=['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'springgreen', 'green'],
                                             # independent variable
                                             figsize=(3, 1),
                                             plot_ticklabels=[r'$\Delta L_1$', r'$\Delta L_2$', r'$\Delta L_3$'],
                                             hline_edge=0.5,
                                             ignore_poor_score=False,
                                             percent=percent,
                                             )


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    import scipy
    import scipy.cluster.hierarchy as sch
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx

def plt_dynamical_regression_distribution_gillan(exp_folder, save_pdf=True):
    csv_path = ANA_SAVE_PATH / exp_folder / f'dynamical_regression_summary.csv'
    dynamical_regression_summary = pd.read_csv(csv_path)

    # stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked
    unique_trial_types = [(0,0,0,0), (0,0,0,1), (0,0,1,0), (0,0,1,1), (0,1,0,0), (0,1,0,1), (0,1,1,0), (0,1,1,1),
                          (1,0,0,0), (1,0,0,1), (1,0,1,0), (1,0,1,1), (1,1,0,0), (1,1,0,1), (1,1,1,0), (1,1,1,1)]
    titles = [f'A{tt[0]+1}S{tt[1]+1}{"BC"[tt[1]]}{tt[2]+1}R{tt[3]}' for tt in unique_trial_types]
    # A1/A2 S1/S2 B1/B2/C1/C2 R0/R1


    cols_name = [''.join(str(i) for i in tt) + '_' + target + '_' + beta
                    for target in ['x1_change']
                    for beta in ['b0', 'b1', 'b2', 'b3']
                    for tt in unique_trial_types
                 ] # 16 * 4 = 64
    print(cols_name)
    X = dynamical_regression_summary[cols_name]
    with pd_full_print_context():
        print(X.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9,0.95,0.99]))
    X = X.values
    outlier_idx = np.argwhere(np.max(np.abs(X), axis=1)>=10).flatten()
    print('Total:',len(outlier_idx),'Outlier', outlier_idx)
    with pd_full_print_context():
        print('Outlier', dynamical_regression_summary.iloc[outlier_idx])
    subject_idxes = np.arange(X.shape[0])
    X = np.delete(X, outlier_idx, axis=0)
    subject_idxes = np.delete(subject_idxes, outlier_idx, axis=0)

    # first dimension reduction using pca, then tsne
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    for pca_num in [50, 25, 20, 10, 5]:
        pca = PCA(n_components=pca_num)
        tsne = TSNE(n_components=3)
        X_pca = pca.fit_transform(X)

        # fig, ax = plot_start(figsize=(3, 3))
        # plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), 'o-')
        # plt.xlabel('# PCs')
        # plt.ylabel('Cumulative explained variance')
        # plt.show()
        # X_plot = X_pca
        X_plot = tsne.fit_transform(X_pca)
        fig, ax = plot_start(figsize=(3, 3))
        plt.scatter(X_plot[:, 0], X_plot[:, 1], s=1, color='k', marker='o')
        plt.xlabel('axis 1')
        plt.ylabel('axis 2')
        fig_exp_path = FIG_PATH / exp_folder / 'dynamical_regression_group'
        os.makedirs(fig_exp_path, exist_ok=True)
        figname = f'pca{pca_num}_tsne' + ('.pdf' if save_pdf else '.png')
        fig.savefig(fig_exp_path / figname, bbox_inches="tight")
        print('Save figure to', fig_exp_path / figname)
        plt.close(fig)

def infer_readout_vector(model_output, model_scores):
    from sklearn.linear_model import LinearRegression
    full_values = extract_value_changes(model_output, return_full_dim=True)[-1]
    logits = extract_value_changes(model_scores, value_type='logit')[0]
    reg = LinearRegression().fit(full_values, logits)
    readout_vector = reg.coef_ # w1, w2
    readout_bias = reg.intercept_ # b
    norm_l2 = np.linalg.norm(readout_vector)
    readout_vector /= norm_l2
    readout_bias /= norm_l2
    return readout_vector, readout_bias


def _plot_2d_value_change_whole(model_pass, fig_exp_path):
    trial_types_ori = model_pass['trial_type']
    trial_types = np.concatenate(trial_types_ori)
    hid_state_lb = model_pass['hid_state_lb']
    hid_state_ub = model_pass['hid_state_ub']
    model_output = model_pass['internal']
    hid_state_rg = hid_state_ub - hid_state_lb
    for d in range(2):
        values, values_change = extract_value_changes(model_output, value_type=d)
        plot_2d_values(values, values_change, trial_types,
                       x_range=(hid_state_lb[d], hid_state_ub[d]), y_range=(-hid_state_rg[d], hid_state_rg[d]),
                       x_label=f'{d + 1} Value', y_label=f'{d + 1} Value change', title='',
                       ref_line=True)
        plt.savefig(fig_exp_path / ('2d_values.png'), bbox_inches="tight")
        plt.show()
        plt.close()


def _plot_2d_vector_field_whole(model_pass, model_pass_1step, fig_exp_path, ev=None, save_pdf=False, title=True,
                                coloring_mapping=None, output_h0=True, plot_n_decimal=1):
    trial_types_ori = model_pass['trial_type']
    if not output_h0:
        # we should remove the last time step in trial_types_ori
        # in this case, model_scores and trial_types_ori had the same length
        trial_types_ori = [trial_type[:-1] for trial_type in trial_types_ori]
    trial_types = np.concatenate(trial_types_ori)
    model_output = model_pass['internal']
    model_scores = model_pass['scores']
    if model_pass_1step is not None:
        model_1step_output = model_pass_1step['internal']
        model_1step_trial_types = np.array(model_pass_1step['trial_type']).flatten()
    else:
        model_1step_output = None
        model_1step_trial_types = None
    readout_vector, readout_bias = infer_readout_vector(model_output, model_scores)
    axes = plt_2d_vector_field(model_output, trial_types, model_1step_output=model_1step_output,
                               model_1step_trial_types=model_1step_trial_types, readout_vector=readout_vector,
                               readout_bias=readout_bias, subplot=True, ev=ev, title=title, coloring_mapping=coloring_mapping,
                               plot_n_decimal=plot_n_decimal)
    figs = [ax.get_figure() for ax in axes]
    if figs[0] is figs[1]:
        fig = figs[0]
        figname = f'2d_vector_field' + ('.pdf' if save_pdf else '.png')
        fig.savefig(fig_exp_path / figname, bbox_inches="tight")
        fig.show()
        plt.close(fig)
    else:
        for i_f, fig in enumerate(figs):
            figname = f'2d_vector_field_{i_f}' + ('.pdf' if save_pdf else '.png')
            fig.savefig(fig_exp_path / figname, bbox_inches="tight")
            fig.show()
            plt.close(fig)

def plot_all_models_value_change(exp_folder, plots, # plots pipeline
                                 save_pdf=False, plot_ev=False, plot_max_logit=5,
                                 model_filters=None, plot_params=None, coloring_mapping=None,output_h0=True,
                                 additional_fname='', func='default',
                                 model_list_from='analysis_perf',
                                 ):
    if model_list_from == 'analysis_perf':
        model_summary = get_model_summary(exp_folder)
        if model_filters is None:
            model_filters = {}
        print(f'In {exp_folder}. Found {len(model_summary)} models')
        for k, v in model_filters.items():
            model_summary = model_summary[model_summary[k] == v]
        if plot_ev:
            model_summary = model_summary[(model_summary['rnn_type'] == 'PNR1') & (model_summary['hidden_dim'] == 2)]
        print(f'Found {len(model_summary)} models after filtering')
        iter_list = [(row['model_path'], row['hidden_dim']) for i, row in model_summary.iterrows()]
    elif model_list_from == 'analysis_scores':
        ana_path = ANA_SAVE_PATH / exp_folder
        # iterate over all files in the nested folder
        iter_list = []
        for p in ana_path.rglob("*"):  # recursively search all subfolders
            if p.name == 'total_scores.pkl':
                model_path = p.parent.relative_to(ANA_SAVE_PATH)
                hidden_dim = 0 # not specified
                iter_list.append((model_path, hidden_dim))
    else:
        raise NotImplementedError

    # for i, row in model_summary.iterrows():
    #     model_path = row['model_path']
    #     hidden_dim = row['hidden_dim']
    for model_path, hidden_dim in iter_list:
        if func == 'default':
            plot_one_model_value_change(model_path, hidden_dim, plots,
                                        save_pdf=save_pdf, plot_ev=plot_ev, plot_max_logit=plot_max_logit, plot_params=plot_params,
                                        coloring_mapping=coloring_mapping,output_h0=output_h0, additional_fname=additional_fname,
                                        plot_n_decimal=1)
        elif func == 'gillan': # for gillan's data
            plot_one_model_value_change_gillan(model_path, hidden_dim, plots,
                                        save_pdf=save_pdf, plot_ev=plot_ev, plot_max_logit=plot_max_logit, plot_params=plot_params,
                                        coloring_mapping=coloring_mapping,output_h0=output_h0, additional_fname=additional_fname)
        elif func == 'suthaharan':
            plot_one_model_value_change_suthaharan(model_path, hidden_dim, plots,
                                        save_pdf=save_pdf, plot_ev=plot_ev, plot_max_logit=plot_max_logit, plot_params=plot_params,
                                        coloring_mapping=coloring_mapping,output_h0=output_h0, additional_fname=additional_fname)
        elif func == 'bahrami':
            plot_one_model_value_change_bahrami(model_path, hidden_dim, plots,
                                        save_pdf=save_pdf, plot_ev=plot_ev, plot_max_logit=plot_max_logit, plot_params=plot_params,
                                        coloring_mapping=coloring_mapping,output_h0=output_h0, additional_fname=additional_fname)


def plot_one_model_value_change(model_path, hidden_dim, plots,
                                save_pdf=False, plot_ev=False, plot_max_logit=5, plot_params=None,
                                coloring_mapping=None,output_h0=True, additional_fname='', plot_n_decimal=1):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_scores = model_pass['scores']
    # model_output = model_pass['internal']
    trial_types_ori = model_pass['trial_type']
    if not output_h0:
        # we should remove the last time step in trial_types_ori
        # in this case, model_scores and trial_types_ori had the same length
        trial_types_ori = [trial_type[:-1] for trial_type in trial_types_ori]
    trial_types = np.concatenate(trial_types_ori)
    # print(model_output[0].shape, model_scores[0].shape, trial_types_ori[0].shape)
    fig_exp_path = FIG_PATH / model_path
    os.makedirs(fig_exp_path, exist_ok=True)
    print(f'{model_path} making {plots}')

    if '2d_value_change' in plots and hidden_dim == 2:
        _plot_2d_value_change_whole(model_pass, fig_exp_path)

    if '2d_vector_field' in plots and hidden_dim == 2:
        if os.path.exists(ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl'):
            model_pass_1step = joblib.load(ANA_SAVE_PATH / model_path / f'2d_inits_pass.pkl')
        else:
            model_pass_1step = None
        if plot_ev and os.path.exists(ANA_SAVE_PATH / model_path / f'eigen.pkl'):
            ev = joblib.load(ANA_SAVE_PATH / model_path / f'eigen.pkl')
        else:
            ev = None
        _plot_2d_vector_field_whole(model_pass, model_pass_1step, fig_exp_path, ev=ev, save_pdf=True, title=False,#True
                                    coloring_mapping=coloring_mapping,output_h0=output_h0, plot_n_decimal=plot_n_decimal)


    if '2d_logit_change' in plots or \
        '2d_logit_next' in plots or  \
        '2d_logit_nextpr' in plots or  \
        '2d_logit_nextpr_ci' in plots or  \
        '2d_pr_nextpr' in plots or \
        '2d_pr_change' in plots:
        # logit change
        logits, logits_change = extract_value_changes(model_scores, value_type='logit')
        action_prob = 1 / (1 + np.exp(-logits))
        next_logit = logits+logits_change
        next_action_prob = 1 / (1 + np.exp(-next_logit))
        prob_change = next_action_prob - action_prob

    relative_action = 'relative_action' in plots # False: absolute action, True: relative action
    if relative_action:
        if hidden_dim != 1:
            print('relative_action only works for 1d models; skipping')
            return
        action_type = trial_types > (trial_types.max() / 2)
        logits[action_type] = -logits[action_type]
        logits_change[action_type] = -logits_change[action_type]
        next_logit[action_type] = -next_logit[action_type]
        action_prob[action_type] = 1 - action_prob[action_type]
        next_action_prob[action_type] = 1 - next_action_prob[action_type]
        prob_change[action_type] = -prob_change[action_type]

    hist = 'hist' in plots
    show_curve = 'show_curve' in plots
    legend = 'legend' in plots
    fname = ('_hist' if hist else '') + ('_relaction' if relative_action else '')
    if show_curve:
        if hidden_dim != 1:
            print('show_curve only works for 1d models; skipping')
            return
        fname += '_curve'+('_legend'  if legend else '')+additional_fname+('.pdf' if save_pdf else '.png')
    else: # too many dots to save as pdf; ignore save_pdf
        fname += additional_fname+('.pdf' if save_pdf else '.png')

    logit_range = (-plot_max_logit, plot_max_logit)

    if '2d_logit_takens' in plots:
        logit_block0 = extract_value_changes(model_scores[0:1], value_type='logit')[0] # input is a list; output is logit numpy array
        trial_types_block0 = trial_types_ori[0]
        plot_2d_logit_takens(logit_block0, trial_types_block0, x_range=logit_range, y_range=logit_range, x_label='Logit(t-1)',
                       y_label='Logit(t)', title='',
                       ref_line=True, ref_x=0, ref_y=0,
                       coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits_takens{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_change' in plots:
        plot_2d_values(logits, logits_change, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit',
                       y_label='Logit change', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        if show_curve and legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        plt.savefig(fig_exp_path / (f'2d_logits{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_next' in plots:
        plot_2d_values(logits, next_logit, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit',
                       y_label='Logit (next)', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits_next{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_nextpr' in plots:
        plot_2d_values(logits, next_action_prob, trial_types, x_range=logit_range, y_range=(0, 1), x_label='Logit',
                       y_label='Action prob (next)', title='',
                       ref_line=True, ref_x=0, ref_y=0.5, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits_next_action_prob{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_pr_nextpr' in plots:
        plot_2d_values(action_prob, next_action_prob, trial_types, x_range=(0, 1), y_range=(0, 1),
                       x_label='Action prob', y_label='Action prob (next)', title='',
                       ref_line=True, ref_x=0.5, ref_y=0.5, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_action_prob_next_action_prob{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_pr_change' in plots:
        plot_2d_values(action_prob, prob_change, trial_types, x_range=(0, 1), y_range=(-1, 1), x_label='Action prob',
                       y_label='Action prob change', title='',
                       ref_line=True, ref_x=0.5, ref_y=0, hist=hist, show_dot=not show_curve, show_curve=show_curve, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_action_prob_prob_change{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()
    unique_trial_types = np.unique(trial_types)

    if len(unique_trial_types) == 4:
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
    elif len(unique_trial_types) == 8:
        # color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick']) # state coloring
        color_spec = np.array(
            ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'tomato',
             'firebrick'])  # action coloring
    else:
        color_spec = []

    if '2d_logit_nextpr_ci' in plots and hidden_dim == 1:
        if not show_curve:
            print('2d_logit_nextpr_ci only works with show_curve; skipping')
            return
        plot_2d_values(logits, next_action_prob, trial_types, x_range=logit_range, y_range=(0,1), x_label='Logit', y_label='Action prob (next)', title='',
                        ref_line=True, ref_x=0, ref_y=0.5, hist=hist, show_dot=False, show_curve=True, coloring_mapping=coloring_mapping)
        bin_results = extract_logit_action_freq(model_scores, trial_types_ori)

        for tt in bin_results.keys():
            bin_centers, p, ci_low, ci_upp, action_counts_of_bin = bin_results[tt]
            if relative_action and tt > (trial_types.max() / 2):
                bin_centers = -bin_centers
                p = 1 - p
                ci_low, ci_upp = 1 - ci_upp, 1 - ci_low

            plt.fill_between(bin_centers, ci_low, ci_upp, alpha=0.2, color=color_spec[tt])

        plt.savefig(fig_exp_path / (f'2d_logits_next_action_prob_CI{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

    if '2d_logit_nextpr_ci_log_odds_ratio' in plots and hidden_dim == 1:
        bin_results = extract_logit_action_freq(model_scores, trial_types_ori)
        _plot_action_pair_log_odds_ratio(bin_results, fig_exp_path, save_pdf=save_pdf, color_spec=color_spec, relative_action=relative_action)


def plot_one_model_value_change_gillan(model_path, hidden_dim, plots,
                                save_pdf=False, plot_ev=False, plot_max_logit=5, plot_params=None, coloring_mapping=None,output_h0=True, additional_fname=''):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_scores = model_pass['scores']
    model_internal = model_pass['internal']
    trial_types_ori = model_pass['trial_type']
    model_mask = model_pass['mask']
    assert output_h0
    # print(model_output[0].shape, model_scores[0].shape, trial_types_ori[0].shape)
    fig_exp_path = FIG_PATH / model_path
    os.makedirs(fig_exp_path, exist_ok=True)
    print(f'{model_path} making {plots}')

    def coloring_mapping(trial_types):
        #
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
        # color_spec = np.array(
        #     ['cornflowerblue', 'mediumblue', 'cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'tomato',
        #      'firebrick'])  # action coloring
        colors = color_spec[trial_types]
        return colors

    if '3d_vector_field' in plots:
        axes = plt_3d_vector_field_gillan(#model_scores,
                                        model_internal,
                                          trial_types_ori,title=True)
        figs = [ax.get_figure() for ax in axes]
        if figs[0] is figs[1]:
            fig = figs[0]
            figname = f'3d_vector_field' + ('.pdf' if save_pdf else '.png')
            fig.savefig(fig_exp_path / figname, bbox_inches="tight")
            fig.show()
            plt.close(fig)
        else:
            for i_f, fig in enumerate(figs):
                figname = f'3d_vector_field_{i_f}' + ('.pdf' if save_pdf else '.png')
                fig.savefig(fig_exp_path / figname, bbox_inches="tight")
                fig.show()
                plt.close(fig)

    if '2d_logit_change' in plots:
        for block in range(len(model_scores)):
            # only first stage action of each trial
            logits, logits_change = extract_value_changes([model_scores[block][::2,:2]], value_type='logit')
            mask = model_mask[block][0][::2].astype(bool) # only the mask for first stage action of each trial

            trial_types = trial_types_ori[block] # list of (stage_1_selected_stimulus, stage_2_state, stage_2_selected_stimulus, reward_masked)
            assert len(trial_types) == 200 and len(mask) == 200
            for stage_2_state in [0, 1]:
                for stage_2_selected_stimulus in [0, 1]:
                    trial_filter = np.array([tt[1] == stage_2_state and tt[2] == stage_2_selected_stimulus for tt in trial_types])
                    logits_filtered = logits[trial_filter & mask]
                    logits_change_filtered = logits_change[trial_filter & mask]
                    trial_types_filtered = np.array([tt[0]*2+tt[3] for tt in trial_types])[trial_filter & mask]
                    fname = f'-stage1_block{block}_at_stage2state-{stage_2_state}_stim-{stage_2_selected_stimulus}' + additional_fname+'.png'
                    #print(fname, logits_filtered, logits_change_filtered, trial_types_filtered)

                    logit_range = (-plot_max_logit, plot_max_logit)
                    plot_2d_values(logits_filtered, logits_change_filtered, trial_types_filtered, x_range=logit_range, y_range=logit_range, x_label='Logit',
                                   y_label='Logit change', title='',
                                   ref_line=True, ref_x=0, ref_y=0, hist=False, show_dot=True, show_curve=False, coloring_mapping=coloring_mapping, plot_params=plot_params)
                    plt.title(f'stage 2 state {stage_2_state} stimulus {stage_2_selected_stimulus}')
                    plt.savefig(fig_exp_path / (f'2d_logits{fname}'), bbox_inches="tight")
                    plt.show()
                    plt.close()


def plot_one_model_value_change_suthaharan(model_path, hidden_dim, plots,
                                save_pdf=False, plot_ev=False, plot_max_logit=5, plot_params=None, coloring_mapping=None,output_h0=True, additional_fname=''):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_scores = model_pass['scores']
    # model_output = model_pass['internal']
    trial_types_ori = model_pass['trial_type']
    model_mask = model_pass['mask']
    assert output_h0
    # print(model_output[0].shape, model_scores[0].shape, trial_types_ori[0].shape)
    fig_exp_path = FIG_PATH / model_path
    os.makedirs(fig_exp_path, exist_ok=True)
    print(f'{model_path} making {plots}')

    def coloring_mapping(trial_types):
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick', 'springgreen', 'darkgreen'])
        colors = color_spec[trial_types]
        return colors

    for block in range(len(model_scores)):
        # only first stage action of each trial
        x = model_scores[block]
        x = x[:, 0] - (x[:, 1] + x[:, 2]) / 2
        logits_change = x[1:] - x[:-1]
        logits = x[:-1]

        trial_types = trial_types_ori[block] # 0~5
        assert len(trial_types) == 160

        fname = f'0vsR_block{block}' + additional_fname+'.png'
        #print(fname, logits_filtered, logits_change_filtered, trial_types_filtered)

        logit_range = (-plot_max_logit, plot_max_logit)
        plot_2d_values(logits, logits_change, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit',
                       y_label='Logit change', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=False, show_dot=True, show_curve=False, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()


def plot_one_model_value_change_bahrami(model_path, hidden_dim, plots,
                                save_pdf=False, plot_ev=False, plot_max_logit=6, plot_params=None, coloring_mapping=None,output_h0=True, additional_fname=''):
    model_pass = joblib.load(ANA_SAVE_PATH / model_path / f'total_scores.pkl')
    model_scores = model_pass['scores']
    # model_output = model_pass['internal']
    trial_types_ori = model_pass['trial_type']
    model_mask = model_pass['mask']
    assert output_h0
    # print(model_output[0].shape, model_scores[0].shape, trial_types_ori[0].shape)
    fig_exp_path = FIG_PATH / model_path
    os.makedirs(fig_exp_path, exist_ok=True)
    print(f'{model_path} making {plots}')

    def coloring_mapping(trial_types):
        color_spec = np.array(['mediumblue', 'firebrick', 'darkgreen', 'darkgoldenrod'])
        colors = color_spec[trial_types]
        return colors

    for block in range(len(model_scores)):
        # only first stage action of each trial
        x = model_scores[block]
        mask = model_mask[block].astype(bool)
        x = x[:, 0] - (x[:, 1] + x[:, 2] + x[:, 3]) / 3
        logits_change = (x[1:] - x[:-1])[mask]
        logits = x[:-1][mask]

        trial_types = trial_types_ori[block][mask].astype(np.int32) # 0~3

        fname = f'0vsR_block{block}' + additional_fname+'.png'
        #print(fname, logits_filtered, logits_change_filtered, trial_types_filtered)

        logit_range = (-plot_max_logit, plot_max_logit)
        plot_2d_values(logits, logits_change, trial_types, x_range=logit_range, y_range=logit_range, x_label='Logit',
                       y_label='Logit change', title='',
                       ref_line=True, ref_x=0, ref_y=0, hist=False, show_dot=True, show_curve=False, coloring_mapping=coloring_mapping, plot_params=plot_params)
        plt.savefig(fig_exp_path / (f'2d_logits{fname}'), bbox_inches="tight")
        plt.show()
        plt.close()

def _plot_action_pair_log_odds_ratio(bin_results, fig_exp_path, save_pdf=False, color_spec=None, relative_action=False):
    fname_prefix = 'relative_' if relative_action else ''
    if color_spec is None:
        color_spec = np.array(['cornflowerblue', 'mediumblue', 'tomato', 'firebrick'])
    for tt1 in bin_results.keys():
        for tt2 in bin_results.keys():
            if tt1 >= tt2:
                continue
            bin_centers1, p1, ci_low1, ci_upp1, action_counts_of_bin1 = bin_results[tt1]
            bin_centers2, p2, ci_low2, ci_upp2, action_counts_of_bin2 = bin_results[tt2]

            plt.figure(figsize=(1.5, 3.5))
            plt.subplot(2, 1, 1)
            plt.plot(bin_centers1, p1, color=color_spec[tt1])
            plt.plot(bin_centers2, p2, color=color_spec[tt2])
            plt.vlines(0, 0, 1, color='k', alpha=0.2)
            plt.hlines(0.5, -5, 5, color='k', alpha=0.2)
            plt.xlim(-5, 5)
            plt.ylabel('Action prob')
            plt.subplot(2, 1, 2)
            log_odds_ratio = []
            lcbs = []
            ucbs = []
            for i in range(len(bin_centers1)):
                table = Table2x2([[action_counts_of_bin1[i,0], action_counts_of_bin1[i, 1]],
                                  [action_counts_of_bin2[i,0], action_counts_of_bin2[i,1]]])
                log_odds_ratio.append(table.log_oddsratio)
                lcb, ucb = table.log_oddsratio_confint(0.05, method='normal')
                lcbs.append(lcb)
                ucbs.append(ucb)
            plt.plot(bin_centers1, log_odds_ratio, color='b')
            plt.plot(bin_centers1, np.zeros(len(bin_centers1)), color='k', linestyle='--')
            plt.fill_between(bin_centers1, lcbs, ucbs, alpha=0.2, color='b')
            plt.xlim(-5, 5)
            plt.ylabel('Log odds ratio')
            plt.xlabel('Logit')
            os.makedirs(fig_exp_path / 'log_odds_ratio', exist_ok=True)
            fname = fname_prefix + f'{tt1}vs{tt2}' + ('.pdf' if save_pdf else '.png')
            plt.savefig(fig_exp_path / 'log_odds_ratio' / fname, bbox_inches="tight")
            plt.show()
            plt.close()


def plot_1d_logit_feature_simple(exp_folder, save_pdf=False, legend=True, feature='intercept'):
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    if feature == 'intercept':
        group_summary = joblib.load(ana_exp_path / 'intercept_group_summary.pkl')
    elif feature == 'slope':
        group_summary = joblib.load(ana_exp_path / 'slope_group_summary.pkl')
    else:
        raise ValueError('feature must be either intercept or slope')
    ticklabels = [c for c in group_summary.columns if 4<=len(c)<=6] # column names like 'A1S1R0' or 'S1R1'
    num_trial_type = len(ticklabels)
    cond_dict = { # x location, reward color, tick label
        'A0S0R0': (0, 0, 'A1\nS1','cornflowerblue'),
        'A0S0R1': (0, 1, 'A1\nS1','mediumblue'),
        'A0S1R0': (2, 0, 'A1\nS2','cornflowerblue'),
        'A0S1R1': (2, 1, 'A1\nS2','mediumblue'),
        'A1S0R0': (3, 0, 'A2\nS1','tomato'),
        'A1S0R1': (3, 1, 'A2\nS1','firebrick'),
        'A1S1R0': (1, 0, 'A2\nS2','tomato'),
        'A1S1R1': (1, 1, 'A2\nS2','firebrick'),

        'S0R0': (0, 0, 'A1','cornflowerblue'),
        'S0R1': (0, 1, 'A1','mediumblue'),
        'S1R0': (1, 0, 'A2','tomato'),
        'S1R1': (1, 1, 'A2','firebrick'),
    }
    for i, row in group_summary.iterrows():
        model_name = row['model_name'] if 'model_name' in row else row['model_type']
        if num_trial_type == 8:
            fig, ax = plot_start(figsize=(1, 1))
        else:
            fig, ax = plot_start(figsize=(0.5, 1))
        max_v = np.max([np.abs(np.mean(row[ticklabels[i]])) for i in range(num_trial_type)])
        plot_ticklabels = [''] * (num_trial_type//2)
        if feature == 'intercept':
            plt.hlines([-1, 0, 1], -0.3, num_trial_type//2 - 0.7, 'grey', alpha=0.5, linestyles='dashed', linewidth=0.5)
            plt.ylim([-1.3, 1.3])
            plt.yticks([-1, 0, 1])
            plt.ylabel('Asymptotic preference\nPrefer A2          Prefer A1')
        else:
            plt.ylim([-0.3, 1.3])
            plt.yticks([0, 0.5, 1])
            plt.ylabel('Learning rate')
        for i in range(num_trial_type):
            points = np.array(row[ticklabels[i]])
            if feature == 'intercept':
                points = points / max_v

            points_mean = np.mean(points)
            x, r, tick_label, color = cond_dict[ticklabels[i]]
            plot_ticklabels[x] = tick_label
            if x == 0:
                label = f'Reward {r}'
            else:
                label = None
            if r == 0:
                marker = 'o'#'s'
            else:
                marker = 'o'#'d'
            #color = ['magenta', 'green'][c]
            # plt.boxplot(points, positions=[x], widths=0.1, vert=True, showfliers=True, patch_artist=True, labels=label,
            #             boxprops=dict(facecolor=color, color=color, alpha=0.2),
            #             medianprops=dict(color=color, alpha=1),
            #             whiskerprops=dict(color=color, alpha=0.5),
            #             capprops=dict(color=color, alpha=0.5),
            #             flierprops=dict(color=color, alpha=0.5, marker='o', markersize=1)
            #
            #             )
            plt.scatter(x, points_mean, label=label, s=5, color=color, marker=marker)#, facecolors='none')
            plt.errorbar(x, points_mean, yerr=np.std(points, ddof=1), color=color, capsize=2)#, label=label)
            # plt.scatter(np.ones(len(points))*(x-0.1), points, s=3, color=color, alpha=0.2, marker=marker)
        plt.xticks(np.arange(num_trial_type//2), plot_ticklabels)
        plt.xlim([-0.3, num_trial_type//2-0.7])
        # plt.gca().invert_yaxis()
        # if num_trial_type == 8:
        #     plt.xlabel('Common          Rare')
        # plt.vlines(0, -1, num_trial_type, 'k', alpha=0.5)
        # flip axis

        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=False, ncol=1)
        plt.title(model_name)
        fig_exp_path = FIG_PATH / exp_folder
        os.makedirs(fig_exp_path, exist_ok=True)
        plt.savefig(fig_exp_path / (f'1d_logits_{feature}_{model_name}_simple'+('.pdf' if save_pdf else '.png')), bbox_inches="tight")
        plt.show()
        plt.close()
        print(f'plot {feature}: {model_name} done')