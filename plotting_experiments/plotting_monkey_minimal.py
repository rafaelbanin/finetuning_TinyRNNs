from plotting import *
from plotting_experiments.plotting_dynamics import *
from plotting_experiments.plotting_decoding import *

save_pdf = True
plotting_pipeline = [
     'plot_model_perf_for_each_exp',
    'plot_dynamics_for_each_exp',
    'plot_1d_for_each_exp',
]
dynamics_plot_pipeline = [
    ## global options
    # 'legend', # show legend; only for 2d_logit_change and show_curve

    ## logit and pr analysis
    '2d_logit_change', # logit vs logit change
    ## other analysis
    '2d_vector_field',
    ]
exp_folders = [
    'exp_monkeyV_minimal',
]

goto_root_dir.run()

dot_alpha = 0.9
curve_alpha= 0.9
markersize = 10
curve_markersize = 5
GRU_color = 'C0'
SGRU_color =  'C5'
LS_color = 'C1'
MF_color = 'C4'
MB_color = 'C3'
PNR_color = 'C2'
model_curve_setting = { # for monkey, all MBs are MFs
    'GRU+SGRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU': ModelCurve('SGRU', 'SGRU', SGRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'SGRU-finetune': ModelCurve('SGRU-f', 'SGRU-f',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'PNR1': ModelCurve('SLIN', 'SLIN',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),

    # MF: C4, LS: C5, MB/MFMB: C3, RC: C4
    'MFs': ModelCurve('MFs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0s': ModelCurve('MBs', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0se': ModelCurve('MB0se', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'LS0': ModelCurve('LS0', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'LS1': ModelCurve('LS1', 'LS', LS_color, dot_alpha, 'v', markersize, 1, '-'),
    'MB0': ModelCurve('MB0', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB1': ModelCurve('MB1', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'MB0m': ModelCurve('MB0m', 'MF', MF_color,dot_alpha, 'o', markersize, 1, '-'),
    'MB0md': ModelCurve('MB-GRU', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'RC': ModelCurve('RC', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(0)': ModelCurve('Q(0)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
    'Q(1)': ModelCurve('Q(1)', 'MF', MF_color, dot_alpha, 'o', markersize, 1, '-'),
}



plot_perf_exp_folders = exp_folders if 'plot_model_perf_for_each_exp' in plotting_pipeline else []
plot_perf_exp_folders += ['exp_monkey'] if 'plot_perf_for_all_exps' in plotting_pipeline else []
for exp_folder in plot_perf_exp_folders:
    # perf
    for add_text in [True, False]:
        plot_all_model_losses(exp_folder,
                          rnn_types=[#'GRU', 'SGRU',
                                     #'SGRU-finetune',
                                     #'PNR1'
                                     'GRU+SGRU',
                                     ],
                          cog_types=['MB0s', 'LS0', #'LS1',
                                     'MB0', 'MB1', #'MB0md',#'MB0m',
                                     'RC'],
                          rnn_filters={'readout_FC': True, 'finetune': 'none',
                                       },
                          xlim=[0.91, 22],
                          ylim=[0.4, 0.55],
                          yticks=[0.4, 0.5],
                          xticks=[1, 2, 3, 4, 5, 10, 20],
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=True,
                          title=exp_folder[4:],
                          figname='loss_all_models',
                          model_curve_setting=model_curve_setting,
                          add_text=add_text,
                          save_pdf=save_pdf,
                          )


    # dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=dynamics_plot_pipeline,
         save_pdf=save_pdf)

if 'plot_1d_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        # plot_1d_logit_coef(exp_folder)
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='intercept')
        plot_1d_logit_feature_simple(exp_folder, save_pdf=save_pdf, legend=False, feature='slope')
