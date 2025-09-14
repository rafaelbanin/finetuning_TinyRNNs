# Nome do arquivo: run_single_training_job.py
import argparse
import joblib
import sys
from pathlib import Path

# --- Garanta que o diretório raiz esteja no path ---
# Isso é crucial para que as importações funcionem no Slurm
# Ajuste o número de '..' se este script estiver em um subdiretório
ROOT_DIR = Path(__file__).resolve().parent.parent 
sys.path.append(str(ROOT_DIR))
# ----------------------------------------------------

try:
    # --- Importe as funções de treinamento ---
    # Ajuste o caminho da importação se necessário
    from training_experiments.training import behavior_cv_training, behavior_cv_training_test_only
    from utils import goto_root_dir, set_os_path_auto # Se necessário
    print("Módulos de treinamento importados com sucesso.")
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    print(f"Verifique se o PYTHONPATH está correto ou se {ROOT_DIR} é a raiz.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Executa um único job de treinamento CV a partir de um config .pkl.")
    parser.add_argument('--config_path', type=str, required=True,
                        help='Caminho para o arquivo de configuração .pkl')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Número de jobs para passar para a função de treino (se aplicável).')
    args = parser.parse_args()

    print(f"Carregando configuração de: {args.config_path}")
    
    try:
        # Garante que o path está correto para carregar (pode ser necessário)
        # com set_os_path_auto(): 
        config = joblib.load(args.config_path)
        print("Configuração carregada com sucesso.")
        print("Config:", config)
        
    except Exception as e:
        print(f"Erro ao carregar o arquivo de configuração {args.config_path}: {e}")
        sys.exit(1)

    # Decide qual função chamar com base em 'trainval_percent'
    trainval_percent = config.get('trainval_percent', 100)

    print(f"Trainval Percent: {trainval_percent}. n_jobs: {args.n_jobs}")

    try:
        if trainval_percent == 0:
            print("Chamando behavior_cv_training_test_only...")
            # behavior_cv_training_test_only espera n_jobs? Verifique a assinatura.
            # Se não, remova args.n_jobs
            behavior_cv_training_test_only(config, n_jobs=args.n_jobs) 
        else:
            print("Chamando behavior_cv_training...")
            behavior_cv_training(config, n_jobs=args.n_jobs)
        
        print(f"Treinamento concluído com sucesso para: {args.config_path}")

    except Exception as e:
        print(f"Erro durante a execução do treinamento para {args.config_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()