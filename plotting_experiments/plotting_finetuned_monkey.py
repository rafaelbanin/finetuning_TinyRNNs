from plotting_experiments.plotting import * 

plotting_pipeline = [
    'plot_single_agent_performance',  # Novo passo de plotagem
]

exp_folders = ['exp_finetuned_monkeyV']

model_curve_setting = {
    '100_pre_trained': ModelCurve('100_pre_trained', '100_pre_trained', 'C9', 0.9, 's', 5, 1, '-'),
    '70_pre_trained': ModelCurve('70_pre_trained', '70_pre_trained', 'C1', 0.9, 'p', 5, 1, '-'),
    '50_pre_trained': ModelCurve('50_pre_trained', '50_pre_trained', 'C4', 0.9, 'd', 5, 1, '-'),
    '20_pre_trained': ModelCurve('20_pre_trained', '20_pre_trained', 'C3', 0.9, '^', 5, 1, '-'),
    'sintetic': ModelCurve('sintetic', 'sintetic', 'green', 0.9, 'v', 5, 1, '-'),
    'MB1_seed0': ModelCurve('MB1_seed0', 'MB1_seed0', 'blue', 0.9, 'o', 5, 1, '-'),
    'Q(0)_seed0': ModelCurve('Q(0)_seed0', 'Q(0)_seed0', 'red', 0.9, 's', 5, 1, '-'),
}

if 'plot_single_agent_performance' in plotting_pipeline:
    for exp_folder in exp_folders:
        plot_single_agent_performance(
            exp_folder=exp_folder,
            perf_file_suffix='_all_models_perf',  # Usa o sufixo do arquivo de análise
            agent_name='MB1_seed0',  # Agente para plotar
            figsize=(8, 6),
            save_pdf=True
        )