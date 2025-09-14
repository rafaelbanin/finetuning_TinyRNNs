import os
import joblib
from pathlib import Path
import re
import sys

# --- Cole aqui as funções auxiliares: ---
# _standardize_path, replace_dot_in_negative_numbers, _standarize_job_name
# e a função generate_slurm_script (modificada para chamar run_single_training_job.py)
# -----------------------------------------

# --- Mantenha essas funções do seu código original ---
def _standardize_path(config_path):
    config_path = str(config_path).replace('\\','/')  # for running on linux
    config_path = config_path.replace('(',r'\(').replace(')',r'\)')
    return config_path

def replace_dot_in_negative_numbers(s: str) -> str:
    pattern = re.compile(r'(-\d+\.\d+)')
    matches = pattern.finditer(s)
    for match in matches:
        s = s.replace(match.group(), match.group().replace('.', ''))
    return s

def _standarize_job_name(name):
    name = name.lower()
    name = replace_dot_in_negative_numbers(name) 
    replace_dict = {
        '/': '.', '_': '.', '-': '.',
        'files.': '', '.allfold.config.pkl': '', 'saved.model.': '', '.cognitive.dynamics.': '', 'd:': '',
        'exp.': '', 'rnn.type.': '', 'dim.': '', 'cog.type.': '',
        'monkey': 'mk.', 'true': 't', 'false': 'f', 'weight.': 'wt.',
        'hidden.': 'hd.', 'output.': 'op.', 'input.': 'ip.',
        'readout.': 'ro.', 'polynomial.order.': 'po.',
        'akamrat': 'akr', 'millerrat': 'mlr', 'gillanhuman': 'gih','gillan1': 'gih',
        'trainval.percent.': 'tvpt.', 'dataprop.': 'dp.', 'inner.splits.': 'ins.',
        r'\(': '', r'\)': '',
        'agent.name': 'ag', 'seed': 'sd',
        'rank': 'rk',
        'include.': '', 'embedding': 'ebd',
        'finetune': 'ft',
        'trainprob.t': 'tpt',
        '.distill': '', 'student': 'st', 'teacher': 'tc', 'none': 'no',
        'trainval.size': 'tvs',
        'omni': 'o', 'l1.wt': 'l1',
        'nonlinearity.': '', 'expand.size.': 'epds.',
        'ro.block.num.': 'robn.',
        'aug2.': '', 'augment.2': 'ag2',
        'pretrained': 'pt',
        'suthaharan': 'sth',
        'gillan': 'gln',
    }
    for k, v in replace_dict.items():
        name = name.replace(k, v)
    name = name.replace('.', '-')
    name = name[:62] 
    return name

# --- Função para Gerar Conteúdo Slurm (Adaptada) ---
def generate_slurm_script_content(config_path, job_name, resource_dict, 
                                  project_root_path, training_runner_script_path, 
                                  python_env_setup, log_dir, n_jobs_per_script=1):
    """Gera o conteúdo de um script Slurm (.slurm)."""
    memory = resource_dict.get('memory', 4) # Padrão 4GB
    cpu = resource_dict.get('cpu', 1)       # Padrão 1 CPU
    gpu = resource_dict.get('gpu', 2)       # Padrão 0 GPU
    
    abs_config_path = Path(config_path).resolve()
    abs_runner_script_path = Path(project_root_path) / training_runner_script_path
    
    output_log = log_dir / f"{job_name}.out"
    error_log = log_dir / f"{job_name}.err"

    # Comando para o script executor
    run_command = f"python {abs_runner_script_path} --config_path {abs_config_path} --n_jobs {n_jobs_per_script}"

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_log}
#SBATCH --error={error_log}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpu}
#SBATCH --mem={memory}G 
"""
    if gpu > 0:
        script_content += f"#SBATCH --gpus={gpu}\n"

    script_content += f"""
echo "Iniciando job $SLURM_JOB_ID em $(hostname)"
echo "Config: {abs_config_path}"

# --- Setup do Ambiente ---
{python_env_setup}

# --- Navega para o Diretório Raiz e Executa ---
echo "Navegando para {project_root_path}"
cd {project_root_path}

echo "Executando: {run_command}"
{run_command}

echo "Job $SLURM_JOB_ID concluído com status $?."
"""
    return job_name, script_content

# --- A NOVA FUNÇÃO GERADORA ---
def behavior_cv_slurm_generation(base_config, config_ranges, resource_dict, 
                                 project_root_path, python_env_setup, 
                                 training_runner_script_path='run_single_training_job.py', 
                                 n_jobs_per_script=1, config_modifier=None, 
                                 ignore_exist=False):
    """
    Gera scripts Slurm para cada combinação de configuração, 
    imitando behavior_cv_training_config_combination.

    Args:
        base_config: O arquivo de config base.
        config_ranges: Dicionário de faixas de config.
        resource_dict: Dicionário de recursos {'memory', 'cpu', 'gpu'}.
        project_root_path: Caminho raiz do projeto no cluster Slurm.
        python_env_setup: Comandos shell para configurar o ambiente Python.
        training_runner_script_path: Caminho (relativo à raiz) para o script executor.
        n_jobs_per_script: Quantos cores/jobs passar para a função de treino interna.
        config_modifier: Função opcional para modificar configs.
        ignore_exist: Ignorar configs se o resumo já existir.
    """
    # --- Certifique-se de que config_control e MODEL_SAVE_PATH estão acessíveis ---
    try:
        sys.path.append('.') # Garante que o diretório atual está no path
        import training_experiments.config_control as config_control
        from path_settings import MODEL_SAVE_PATH # Importa de path_settings.py
        print("Módulos config_control e MODEL_SAVE_PATH importados.")
    except ImportError as e:
        print(f"Erro ao importar 'config_control' ou 'MODEL_SAVE_PATH': {e}")
        print("Certifique-se de que está executando do diretório raiz e que path_settings.py existe.")
        return []
    # --------------------------------------------------------------------------------

    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    sbatch_cmds = []
    
    exp_folder = base_config.get('exp_folder', 'slurm_experiment')
    slurm_dir = Path('./files/slurm') / exp_folder
    log_dir = slurm_dir / 'logs'
    config_save_dir = Path('./files/slurm_configs') / exp_folder
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(config_save_dir, exist_ok=True)

    submit_file_path = slurm_dir / f'submit_all.txt'
    cancel_file_path = slurm_dir / f'cancel_all.txt'
    submit_file_path.unlink(missing_ok=True)
    cancel_file_path.unlink(missing_ok=True)

    print(f"Gerando {len(configs)} configurações para Slurm...")

    with open(submit_file_path, 'a') as submit_f, \
         open(cancel_file_path, 'a') as cancel_f:

        for i, c in enumerate(configs):
            if config_modifier is not None:
                c = config_modifier(c)

            # Gera um model_path/job_name se não existir ou for simples
            job_name_base = c.get('model_path', f'config_{i}')
            job_name = _standarize_job_name(job_name_base)
            
            # Garante que o model_path na config seja usado para salvar
            # e que seja único, mesmo que o job_name seja simplificado.
            c['model_path'] = c.get('model_path', Path(exp_folder) / job_name)

            # Define o caminho para o pkl e o sumário
            config_pkl_path = config_save_dir / f"{job_name}.pkl"
            # O caminho do sumário deve ser o mesmo que 'training.py' usaria
            summary_path = MODEL_SAVE_PATH / c['model_path'] / 'allfold_summary.pkl'

            # Verifica se deve ignorar
            if ignore_exist and os.path.exists(summary_path):
                print(f"Ignorando {job_name}, sumário já existe em {summary_path}.")
                continue

            # Salva o arquivo .pkl
            joblib.dump(c, config_pkl_path)

            # Gera o conteúdo do script Slurm
            job_name, script_content = generate_slurm_script_content(
                config_pkl_path, job_name, resource_dict, 
                project_root_path, training_runner_script_path, 
                python_env_setup, log_dir, n_jobs_per_script
            )

            # Escreve o arquivo .slurm
            slurm_script_file_path = slurm_dir / f'{job_name}.slurm'
            with open(slurm_script_file_path, 'w') as f:
                f.write(script_content)

            # Cria e armazena comandos
            sbatch_cmd = f'sbatch {slurm_script_file_path}'
            scancel_cmd = f'scancel --name={job_name}' 
            print(sbatch_cmd, file=submit_f)
            print(scancel_cmd, file=cancel_f)
            sbatch_cmds.append(sbatch_cmd)

    if not sbatch_cmds:
        print("Nenhum script Slurm foi gerado.")
        return []

    print(f"\n{len(sbatch_cmds)} scripts Slurm gerados em: {slurm_dir}")
    print(f"Configs .pkl salvos em: {config_save_dir}")
    print(f"Comandos de submissão em: {submit_file_path}")
    print(f"Comandos de cancelamento em: {cancel_file_path}")
    print("\n--- Primeiros 5 Comandos sbatch ---")
    for cmd in sbatch_cmds[:5]:
        print(cmd)
    if len(sbatch_cmds) > 5:
        print(f"... e mais {len(sbatch_cmds) - 5} comandos.")
    print("----------------------------------\n")
        
    return sbatch_cmds

# --- Nova Função para Escrever Scripts e Comandos Slurm ---
def write_slurm_scripts(config_paths, exp_folder, resource_dict, 
                        project_root_path, training_script_path, 
                        python_env_setup, n_jobs=1):
    """Gera arquivos .slurm e comandos sbatch para todos os config_paths."""
    slurm_dir = Path('./files/slurm') / exp_folder
    log_dir = slurm_dir / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    sbatch_cmds = []
    
    # Arquivos para guardar comandos
    submit_file_path = Path(f'files/slurm/submit_{exp_folder}.txt')
    cancel_file_path = Path(f'files/slurm/cancel_{exp_folder}.txt')

    # Limpa os arquivos antigos ou os cria
    submit_file_path.unlink(missing_ok=True)
    cancel_file_path.unlink(missing_ok=True)

    with open(submit_file_path, 'a') as submit_f, \
         open(cancel_file_path, 'a') as cancel_f:

        for config_path in config_paths:
            # Padroniza caminho e nome
            std_config_path = _standardize_path(config_path)
            job_name = _standarize_job_name(std_config_path)

            # Gera o conteúdo do script
            job_name, script_content = generate_slurm_script(
                config_path, job_name, resource_dict, 
                project_root_path, training_script_path, 
                python_env_setup, log_dir, n_jobs
            )

            # Escreve o arquivo .slurm
            slurm_script_file_path = slurm_dir / f'{job_name}.slurm'
            with open(slurm_script_file_path, 'w') as f:
                f.write(script_content)

            # Cria e armazena comandos
            sbatch_cmd = f'sbatch {slursbatch_cmdsm_script_file_path}'
            scancel_cmd = f'scancel --name={job_name}' # Comando para cancelar

            print(sbatch_cmd, file=submit_f)
            print(scancel_cmd, file=cancel_f)
            sbatch_cmds.append(sbatch_cmd)
            
    print(f"Scripts Slurm gerados em: {slurm_dir}")
    print(f"Comandos de submissão em: {submit_file_path}")
    print(f"Comandos de cancelamento em: {cancel_file_path}")
    
    return sbatch_cmds

from pathlib import Path
from training_experiments.training import *


# --- Exemplo de base_config e config_ranges ---
base_config = { ########### Treino base com dados sintéticos #################
      ### dataset info
      'dataset': 'SimAgent',
      'behav_format': 'tensor',
      'behav_data_spec': ['agent_path', 'agent_name'],
      'agent_path': ['allagents_monkeyV_nblocks100_ntrials100'],
      'agent_name': 'LS0_seed0',
      ### model info
      'agent_type': 'RNN',
      'rnn_type': 'GRU', # which rnn layer to use
      'input_dim': 3,
      'hidden_dim': 2, # dimension of this rnn layer
      'output_dim': 2, # dimension of action
      'device': 'cuda',
      'output_h0': True, # whether initial hidden state included in loss
      'trainable_h0': False, # the agent's initial hidden state trainable or not
      'readout_FC': True, # whether the readout layer is full connected or not
      'one_hot': False, # whether the data input is one-hot or not
      ### training info for one model
      'lr':0.005,
      'l1_weight': 1e-5,
      'weight_decay': 0,
      'penalized_weight': 'rec',
      'max_epoch_num': 2000,
      'early_stop_counter': 200,
      ### training info for many models on dataset
      'trainval_percent': 0, # subsample the training-validation data (percentage) ######
      'outer_splits': 10,
      'inner_splits': 9,
      'seed_num': 3,
      ### additional training info
      'save_model_pass': 'full', # 'full' for saving all results; 'minimal' for saving only the losses; 'none' for not saving results
      'training_diagnose': None, # can be a list of diagnose function strings
      ### current training exp path
      'exp_folder': get_training_exp_folder_name(__file__),
      'model_based': 'sintetic',
}


config_ranges = { # keys are used to generate model names
      'agent_name': [#'GRU_1_seed0',
           # 'GRU_2_seed0',
            #'SGRU_1_seed0',
            #  'RC_seed0',
            # 'MB0s_seed0',
            # 'LS0_seed0',
            # 'MB0_seed0', 
            'MB1_seed0',
                     ],
      'rnn_type': ['GRU'],
      'hidden_dim': [#1,
                     2,
                     #3,4
            #50, 20,10,5
                     ],
      'model_based': ['sintetic'],
      'trainval_percent': [100, 
                         90, 80, 70, 60, 
                         50, 40, 30, 20, 
                         10,
                         0
                         ]

}

# --- Recursos solicitados para cada job ---
resource_dict = {
    'memory': 8,   # em GB
    'cpu': 2,
    'gpu': 2
}

# --- Caminhos principais ---
project_root_path = Path('.').resolve()  # Diretório atual
python_env_setup = 'source /scratch/rc6118/env/Cognicao/bin/activate'  # Ajuste para seu ambiente

# --- Caminho para o script Python que treina o modelo ---
training_runner_script_path = 'run_single_training_job.py'

# --- Instancia a geração dos scripts Slurm ---
behavior_cv_slurm_generation(
    base_config=base_config,
    config_ranges=config_ranges,
    resource_dict=resource_dict,
    project_root_path=project_root_path,
    python_env_setup=python_env_setup,
    training_runner_script_path=training_runner_script_path,
    n_jobs_per_script=1,
    config_modifier=None,     # Você pode passar uma função que modifica os configs aqui
    ignore_exist=False        # Para não pular se já houver sumário
)