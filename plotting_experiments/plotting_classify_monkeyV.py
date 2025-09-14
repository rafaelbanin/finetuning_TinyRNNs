from plotting_classify import *


for comp, ref_score_labels, ylim_max in [
    # ('SGRU_1vsMB0s', ['GRU (d=1)', 'MF (d=1)'], 2),
    ('GRU_2vsMB1', ['GRU (d=2)', 'MF (d=2)'], 4),
    ]:
    parent_dir = rf'exp_classify_monkeyV{comp}\rnn_type-GRU.hidden_dim-10.l1_weight-1e-05'
    # for all subfolders in parent_dir, plot the model evidence
    for sub_dir in os.listdir(ANA_SAVE_PATH / parent_dir):
        if not os.path.isdir(ANA_SAVE_PATH / parent_dir / sub_dir):
            continue
        plot_evidence_for_all_scores(rf'{parent_dir}\{sub_dir}\total_scores.pkl',
                                     rf'{parent_dir}\{sub_dir}\total_scores_V.pkl',
                                     ref_score_labels=ref_score_labels, sub_score_labels=['monkey V'],
                                     ref_plot_every_n=20, exp_fig_path=f'exp_classify_monkeyV',
                                     fig_name=f'{sub_dir}_model_evidence_{comp}',xlim_max=60, ylim_max=ylim_max)