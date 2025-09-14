import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import os
import numpy as np

from analyzing_experiments.analyzing_perf import *
from analyzing_experiments.analyzing_check import *
from analyzing_experiments.analyzing_dynamics import *
from plotting_experiments.plotting import *
from plotting_experiments.plotting_dynamics import *

# Importe as funções de controle de configuração do seu projeto
# Supondo que 'config_control' esteja acessível (pode precisar ajustar o import)
# from training_experiments import config_control # Exemplo, ajuste conforme sua estrutura
# Se config_control não estiver acessível, você pode precisar listar os caminhos manualmente
# ou replicar a lógica de vary_config aqui.

# --- Definições de Caminho e Experimento ---
# !!! IMPORTANTE: Ajuste MODEL_SAVE_PATH para o diretório base onde seus
# !!! resultados (saved_model) estão salvos.
# --- Placeholder ---
MODEL_SAVE_PATH = Path('./saved_model')  # <<< AJUSTE ESTE CAMINHO!
# --- Fim das Definições ---

# Função para replicar config_control.vary_config (se não puder importar)
def vary_config_simple(base_config, config_ranges):
    """Versão simplificada de vary_config para gerar combinações."""
    from itertools import product
    configs = []
    # Obtém as chaves e listas de valores dos ranges
    range_keys = list(config_ranges.keys())
    value_lists = [config_ranges[key] for key in range_keys]

    # Gera todas as combinações de valores
    for value_combination in product(*value_lists):
        current_config_variant = {}
        for i, key in enumerate(range_keys):
            current_config_variant[key] = value_combination[i]
        
        # Combina com a base_config, a variante tem precedência
        final_config = base_config.copy()
        final_config.update(current_config_variant)
        configs.append(final_config)
    return configs


def collect_and_analyze_sintetic_initial_loss(
    model_save_base_path: Path,
    base_config_template: dict,
    config_ranges_to_scan: dict,
    filter_criteria: dict = None
):
    """
    Coleta, combina, analisa e plota a test_loss inicial.

    Args:
        model_save_base_path (Path): Caminho base para a pasta 'saved_model'.
        base_config_template (dict): O template da base_config do seu experimento.
        config_ranges_to_scan (dict): Os config_ranges que definem as variações
                                      (especialmente trainval_percent).
        filter_criteria (dict): Critérios para filtrar os dados antes de plotar
                                (ex: {'model_based': 'sintetic'}).
    """
    if filter_criteria is None:
        filter_criteria = {}

    all_summaries_list = []

    # Gera todas as configurações que foram executadas
    # Tenta importar config_control, se não, usa a versão simples
    try:
        from training_experiments import config_control
        configs_to_check = config_control.vary_config(base_config_template, config_ranges_to_scan, mode='combinatorial')
    except ImportError:
        print("Aviso: Não foi possível importar 'config_control'. Usando uma função de variação de config simples.")
        print("Pode ser necessário ajustar a lógica se 'config_control.vary_config' for complexa.")
        configs_to_check = vary_config_simple(base_config_template, config_ranges_to_scan)


    print(f"Verificando {len(configs_to_check)} configurações potenciais...")

    for config_run in configs_to_check:
        # Reconstrói o model_path como foi salvo
        # Isso pode precisar de ajuste fino baseado em como config['model_path'] é exatamente formado
        # no seu script `training.py`.
        # Exemplo: config['model_path'] = f"{config['exp_folder']}/agent_name-{config['agent_name']}..."
        
        # Vamos assumir que 'model_path' na config_run já é o caminho relativo correto
        # como 'exp_finetuned_monkeyV/agent_name-...'
        # Se não, você precisará reconstruí-lo aqui.
        if 'model_path' not in config_run:
            print(f"Aviso: 'model_path' não encontrado na config gerada. Pulando: {config_run}")
            continue

        summary_file_path = model_save_base_path / Path(config_run['model_path']) / 'allfold_summary.pkl'

        if summary_file_path.exists():
            print(f"Carregando: {summary_file_path}")
            try:
                # Assume set_os_path_auto não é crucial para joblib.load simples
                df_summary = joblib.load(summary_file_path)
                all_summaries_list.append(df_summary)
            except Exception as e:
                print(f"Erro ao carregar {summary_file_path}: {e}")
        else:
            print(f"Não encontrado: {summary_file_path}")

    if not all_summaries_list:
        print("Nenhum arquivo de sumário encontrado. Verifique os caminhos e se os testes 0% foram executados.")
        return

    combined_df = pd.concat(all_summaries_list, ignore_index=True)
    print(f"Total de {len(combined_df)} linhas carregadas de todos os sumários.")

    # Adiciona colunas do dicionário 'config' como colunas de nível superior se não existirem
    # para facilitar a filtragem e agrupamento.
    # Especialmente 'model_based' e 'trainval_percent'.
    if 'config' in combined_df.columns:
        # Extrai model_based se não for uma coluna
        if 'model_based' not in combined_df.columns:
            combined_df['model_based'] = combined_df['config'].apply(lambda c: c.get('model_based', 'N/A'))
        # Extrai trainval_percent se não for uma coluna (deve ser, pela nossa modificação)
        if 'trainval_percent' not in combined_df.columns:
            combined_df['trainval_percent'] = combined_df['config'].apply(lambda c: c.get('trainval_percent', -1))


    # Aplica filtros
    filtered_df = combined_df.copy()
    for key, value in filter_criteria.items():
        if key in filtered_df.columns:
            print(f"Filtrando por: {key} == {value}")
            filtered_df = filtered_df[filtered_df[key] == value]
        else:
            print(f"Aviso: Chave de filtro '{key}' não encontrada nas colunas do DataFrame.")
    
    if filtered_df.empty:
        print(f"Nenhum dado encontrado após aplicar os filtros: {filter_criteria}")
        return

    print(f"Dados após filtragem ({filter_criteria}): {len(filtered_df)} linhas.")

    # Garante que test_loss e trainval_percent são numéricos
    filtered_df['test_loss'] = pd.to_numeric(filtered_df['test_loss'], errors='coerce')
    filtered_df['trainval_percent'] = pd.to_numeric(filtered_df['trainval_percent'], errors='coerce')
    filtered_df.dropna(subset=['test_loss', 'trainval_percent'], inplace=True)


    # Agrupa para calcular Média e SEM
    # Adicione outras chaves de agrupamento se necessário (ex: 'rnn_type', 'hidden_dim')
    # para plotar curvas separadas se houver variações nesses parâmetros.
    # Por agora, agrupamos apenas por trainval_percent.
    grouped = filtered_df.groupby('trainval_percent')['test_loss']
    mean_loss = grouped.mean().reset_index()
    sem_loss = grouped.sem().reset_index() # Erro Padrão da Média

    if mean_loss.empty:
        print("Nenhum dado para plotar após agrupamento.")
        return

    # Plotar
    plt.figure(figsize=(10, 6))
    plt.errorbar(mean_loss['trainval_percent'], mean_loss['test_loss'], 
                 yerr=sem_loss['test_loss'], 
                 marker='o', linestyle='-', label='Test Loss Inicial (Média +/- SEM)', capsize=3)

    plt.title(f"Test Loss Inicial vs. % Dados de Treino ({filter_criteria})")
    plt.xlabel("Percentual de Dados de Treino-Validação (%)")
    plt.ylabel("Test Loss Inicial (sem treino)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajusta os ticks do eixo X para mostrar todos os trainval_percent testados
    unique_trainval_percents = sorted(filtered_df['trainval_percent'].unique())
    if unique_trainval_percents:
        plt.xticks(unique_trainval_percents)

    plt.legend()
    plt.tight_layout()

    output_filename = "initial_loss_analysis_plot.png"
    plt.savefig(output_filename)
    print(f"Gráfico salvo como '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    # --- Exemplo de como usar ---
    # 1. Defina a base_config e config_ranges EXATAMENTE como no seu
    #    script exp_finetuned_monkeyV.py para os modelos 'sintetic'.

    sintetic_base_config = {
        'dataset': 'SimAgent',
        'behav_format': 'tensor',
        'behav_data_spec': ['agent_path', 'agent_name'],
        'agent_path': ['allagents_monkeyV_nblocks100_ntrials100'],
        # 'agent_name': 'MB1_seed0', # Será variado por config_ranges
        'agent_type': 'RNN',
        'rnn_type': 'GRU',
        'input_dim': 3,
        'hidden_dim': 2,
        'output_dim': 2,
        'device': 'cuda', # Ou 'cpu' se rodou em cpu
        'output_h0': True,
        'trainable_h0': False,
        'readout_FC': True,
        'one_hot': False,
        'lr': 0.005,
        'l1_weight': 1e-05,
        'weight_decay': 0,
        'penalized_weight': 'rec',
        'max_epoch_num': 2000, # Irrelevante para teste inicial
        'early_stop_counter': 200, # Irrelevante
        'outer_splits': 10,
        'inner_splits': 9, # Irrelevante para teste inicial mas usado na estrutura
        'seed_num': 3,
        'save_model_pass': 'full',
        'training_diagnose': None,
        'exp_folder': 'exp_finetuned_monkeyV', # Importante para o model_path
        'model_based': 'sintetic', # Importante para o model_path e filtro
    }

    # Estes são os ranges que VOCÊ USOU para gerar os dados de 'sintetic'
    # incluindo os diferentes trainval_percent (e 0%).
    sintetic_config_ranges = {
        'agent_name': ['MB1_seed0'], # Se variou, liste aqui
        'hidden_dim': [2],          # Se variou, liste aqui
        # Adicione outras chaves que você variou e que fazem parte do nome do diretório
        'trainval_percent': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # IMPORTANTE
    }

    # Constrói o 'model_path' dinamicamente para cada config
    # Esta função é chamada por config_control.vary_config no script de treino
    # Precisamos replicar como o model_path é formado.
    def set_model_path(config):
        # Exemplo de como o model_path pode ser construído no seu script original.
        # VOCÊ PRECISA AJUSTAR ISSO PARA CORRESPONDER EXATAMENTE!
        tvp = config['trainval_percent']
        mb = config['model_based']
        an = config['agent_name']
        hd = config['hidden_dim']
        rt = config['rnn_type']
        ef = config['exp_folder']
        
        # Atenção: A ordem e os nomes das chaves no nome do arquivo devem ser os mesmos
        # que o script de treinamento usa para criar os diretórios.
        config['model_path'] = f"{ef}/agent_name-{an}.rnn_type-{rt}.hidden_dim-{hd}.model_based-{mb}.trainval_percent-{tvp}"
        return config

    # Aplica a função para construir model_path em todas as configs geradas
    temp_configs_for_path_generation = vary_config_simple(sintetic_base_config, sintetic_config_ranges)
    configs_with_paths = [set_model_path(conf.copy()) for conf in temp_configs_for_path_generation]


    # Agora, a função de análise pode usar 'model_path' de cada config
    # Para simplificar, vamos passar a lista de configs com paths já construídos.
    # A função collect_and_analyze... precisa ser um pouco ajustada para aceitar esta lista
    # ou podemos filtrar a lista aqui.

    # Ajuste para collect_and_analyze_sintetic_initial_loss:
    # Em vez de passar base_config e config_ranges, passamos a lista de configs já com os paths
    
    all_summaries_list_manual = []
    for config_run in configs_with_paths:
        # Filtra apenas para as configs que realmente são 'sintetic' (caso haja outras em configs_with_paths)
        if config_run.get('model_based') != 'sintetic':
            continue

        summary_file_path = MODEL_SAVE_PATH / Path(config_run['model_path']) / 'allfold_summary.pkl'
        if summary_file_path.exists():
            print(f"Carregando: {summary_file_path}")
            try:
                df_summary = joblib.load(summary_file_path)
                # Adiciona o trainval_percent da config principal se não estiver no df
                # (embora devesse estar após as modificações anteriores)
                if 'trainval_percent' not in df_summary.columns:
                    df_summary['trainval_percent'] = config_run['trainval_percent']
                if 'model_based' not in df_summary.columns:
                     df_summary['model_based'] = config_run['model_based']
                all_summaries_list_manual.append(df_summary)
            except Exception as e:
                print(f"Erro ao carregar {summary_file_path}: {e}")
        else:
            print(f"Não encontrado: {summary_file_path}")
    
    if not all_summaries_list_manual:
        print("Nenhum arquivo de sumário 'sintetic' encontrado manualmente. Verifique os caminhos.")
    else:
        final_combined_df = pd.concat(all_summaries_list_manual, ignore_index=True)
        
        # Chamar a função de plotagem com o DataFrame combinado e o filtro
        plot_initial_test_loss_vs_trainval(final_combined_df, 
                                           filter_criteria={'model_based': 'sintetic'})