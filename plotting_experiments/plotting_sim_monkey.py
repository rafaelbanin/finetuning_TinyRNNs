from plotting import *
from plotting_experiments.plotting_dynamics import *

goto_root_dir.run()

plotting_pipeline = [
    #'plot_model_perf_for_each_exp',
    # 'plot_dynamics_for_each_exp',
    # 'plot_1d_for_each_exp',
    'plot_model_perf_data_proportion_for_each_exp',
]

exp_folders = [
     'exp_sim_monkeyV',
    #'exp_sim_monkeyW',
]

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
    #'GRU+SGRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'GRU': ModelCurve('GRU', 'GRU', GRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    #'SGRU': ModelCurve('SGRU', 'SGRU', SGRU_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    'GRU-f': ModelCurve('GRU-f', 'GRU-f',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),
    #'PNR1': ModelCurve('SLIN', 'SLIN',PNR_color, curve_alpha, 'x', curve_markersize, 1, '-'),

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



if 'plot_model_perf_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        for sim_agent_name in [#'SGRU_1_seed0', 'GRU_2_seed0',
                               #'MB0s_seed0', 'LS0_seed0', #'LS1_seed0',
                            #'MB0_seed0', 
                            'MB1_seed0', #'MB0md_seed0', #'RC_seed0',
                               ]:
            for add_text in [True, False]:
                for perf_type, figname, ylim, yticks in [
                         ('test_loss','loss_all_models', [0.35, 0.65],[0.4, 0.5, 0.6]),
                        # ('test_acc','acc_all_models', [0.65,0.82], [0.7,0.8])
                    ]:
                    plot_all_model_losses(exp_folder,
                          rnn_types=[#'GRU', 'SGRU', #
                                     #'PNR1'
                              'GRU+SGRU',
                                     ],
                          cog_types=[#'MB0s', 'LS0', #'LS1',
                                     #'MB0', 
                                     #'MB1', #'MB0md',
                                     ],
                          rnn_filters={'agent_name': sim_agent_name},
                          cog_filters={'agent_name': sim_agent_name},
                          xlim=[0.91, 5],
                          xticks=[1, 2, 3, 4, #5, 10, 20
                                  ],
                              ylim=ylim,
                            yticks=yticks,
                          max_hidden_dim=20,
                          minorticks=False,
                          figsize=(1.5, 1.5),
                          legend=False,
                          title=f'{sim_agent_name}_fit_{exp_folder[8:]}',
                          perf_type=perf_type,
                          figname=f'{sim_agent_name}_{figname}',
                          add_text=add_text,
                            model_curve_setting=model_curve_setting,
                          save_pdf=True,
                          )
# dynamics
if 'plot_dynamics_for_each_exp' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_all_models_value_change(exp_folder, plots=[
            # '2d_value_change',
            '2d_logit_change',
            # '2d_vector_field',
        ])
        # plot_1d_logit_coef(exp_folder)


if 'plot_model_perf_data_proportion_for_each_exp' in plotting_pipeline:
    exp_folders_dataprop = [
    'exp_sim_monkeyV',
    # 'exp_monkeyW_dataprop',
    ]
    for exp_folder in exp_folders_dataprop:
        plot_all_model_losses_dataprop(exp_folder,
                              rnn_types=['GRU','GRU-f'
                                  # 'SGRU'
                                         ],
                              cog_types=[#'MB0s', 'LS0', #'LS1',
                                         #'MB0', 'MB1', #'MB0md','MB0m',
                                     #'RC'
                              ],
                              xlim=[0, 5000],
                              xticks=[0,1000, 2000,4000,6000,8000],
                              minorticks=False,
                              figsize=(1.5, 1.5),
                              legend=False,
                              title=exp_folder.replace('exp_seg_', ''),
                              figname='loss_all_models_dataprop',
                              model_curve_setting=model_curve_setting,
                          save_pdf=True,
                              )