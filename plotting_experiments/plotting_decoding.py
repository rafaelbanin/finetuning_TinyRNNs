import matplotlib.pyplot as plt

from plotting_experiments.plotting import *
from analyzing_experiments.analyzing import *

def plot_neural_pca_decoding(exp_folder, neuro_data_spec, session_names=None):
    assert session_names is not None
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    rnn_summary['model_type'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    cog_summary['model_type'] = cog_summary['cog_type']
    model_decoding_perf_sessions = {}
    for session_name in session_names:
        model_decoding_perf = {}
        for i, row in pd.concat([rnn_summary, cog_summary], axis=0, join='outer').iterrows():
            hidden_dim = row['hidden_dim']
            if hidden_dim < 1 or hidden_dim > 2:
                continue
            model_name = row['model_type']
            if model_name not in ['GRU',
                                  'SGRU','MB0','MB0s','MB1','LS0','LS1'
                                  ]:
                continue
            if row['readout_FC'] == False:
                continue

            model_path = transform_model_format(row, source='row', target='path')
            ana_path = ANA_SAVE_PATH / model_path / 'PCA_decoding'
            fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
            R2s = joblib.load(ana_path / f'{session_name}_{fname}_logits_R2s.pkl')

            model_key = f'{model_name}-{hidden_dim}'
            model_decoding_perf.setdefault(model_key, []).append(R2s[0:100])
        plt.figure()
        for model_key in list(model_decoding_perf.keys()):
            assert len(model_decoding_perf[model_key]) == 10

            # plt.figure()
            # for curve in model_decoding_perf[model_key]:
            #     plt.plot(range(10, 100), curve)
            # plt.title(session_name+' '+model_key)
            # plt.show()
            model_decoding_perf[model_key] = np.mean(model_decoding_perf[model_key], 0)
            model_decoding_perf_sessions.setdefault(model_key, []).append(model_decoding_perf[model_key][50])
            plt.plot(range(0, 100), model_decoding_perf[model_key], label=model_key)
        plt.legend()
        plt.title(session_name)
        plt.show()
    return model_decoding_perf_sessions

def plot_neuronal_r2(exp_folder, neuro_data_spec, session_names=None):
    assert session_names is not None
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
    rnn_summary['model_type'] = rnn_summary['rnn_type']
    cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
    cog_summary['model_type'] = cog_summary['cog_type']
    model_decoding_perf_sessions = {}
    fname = '_'.join([f'{k}_{v}' for k, v in neuro_data_spec.items()])
    for session_name in session_names:
        model_decoding_perf = {}
        for i, row in pd.concat([rnn_summary, cog_summary], axis=0, join='outer').iterrows():
            hidden_dim = row['hidden_dim']
            if hidden_dim < 1:
                continue
            model_name = row['model_type']
            if row['readout_FC'] == False:
                continue

            model_path = transform_model_format(row, source='row', target='path')
            ana_model_path = ANA_SAVE_PATH / model_path / 'decoding'
            neuron_decoding = joblib.load(ana_model_path / f'{session_name}_{fname}_task_var_value_decode_neuron.pkl')
            neuron_R2 = neuron_decoding['r2']
            model_key = f'{model_name}-{hidden_dim}'
            model_decoding_perf.setdefault(model_key, []).append(np.mean(neuron_R2))
        plt.figure()
        x_idx = 0
        for model_key in list(model_decoding_perf.keys()):
            assert len(model_decoding_perf[model_key]) == 10 # folds

            # plt.figure()
            # for curve in model_decoding_perf[model_key]:
            #     plt.plot(range(10, 100), curve)
            # plt.title(session_name+' '+model_key)
            # plt.show()
            # model_decoding_perf[model_key] = np.mean(model_decoding_perf[model_key])
            dots = model_decoding_perf[model_key]
            model_decoding_perf_sessions.setdefault(model_key, []).append(dots)
            plt.scatter(np.ones_like(dots)*x_idx, dots, label=model_key)
            x_idx += 1
        plt.xticks(range(len(model_decoding_perf.keys())), list(model_decoding_perf.keys()))
        plt.legend()
        plt.title(session_name)
        plt.show()
    return model_decoding_perf_sessions