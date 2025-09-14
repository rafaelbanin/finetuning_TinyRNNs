import os
from training_experiments.training import *
import re

def _standardize_path(config_path):
    config_path = str(config_path).replace('\\','/')  # for running on linux
    config_path = config_path.replace('(',r'\(').replace(')',r'\)')
    return config_path


def replace_dot_in_negative_numbers(s: str) -> str:
    # Define a regex pattern to find negative decimal numbers
    pattern = re.compile(r'(-\d+\.\d+)')
    # Find all occurrences of the pattern in the input string
    matches = pattern.finditer(s)
    # Replace the dot in each match and update the input string
    for match in matches:
        s = s.replace(match.group(), match.group().replace('.', ''))
    return s

def _standarize_job_name(name):
    name = name.lower()
    name = replace_dot_in_negative_numbers(name) # remove "." usually used in the floats, to avoid further problems when extracting the shared name
    replace_dict = {
        '/': '.', '_': '.', '-': '.',  # Nautilus only allow '.' in the name
        'files.': '', '.allfold.config.pkl': '', 'saved.model.': '', '.cognitive.dynamics.': '', 'd:': '',
        'exp.': '', 'rnn.type.': '', 'dim.': '', 'cog.type.': '',  # remove redundant information
        'monkey': 'mk.', 'true': 't', 'false': 'f', 'weight.': 'wt.',  # shorten the name
        'hidden.': 'hd.', 'output.': 'op.', 'input.': 'ip.',
        'readout.': 'ro.', 'polynomial.order.': 'po.',
        'akamrat': 'akr', 'millerrat': 'mlr', 'gillanhuman': 'gih','gillan1': 'gih',
        'trainval.percent.': 'tvpt.', 'dataprop.': 'dp.', 'inner.splits.': 'ins.',
        r'\(': '', r'\)': '',  # remove brackets
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
    assert len(name)<=62, f'Name {name} has length of {len(name)}. The maximum length is 62.'
    return name


def _find_max_shared_string(strings):
    # Split each string by dots
    split_strings = [s.split('.') for s in strings]

    # they should have the same length
    min_len = min([len(s) for s in split_strings])
    max_len = max([len(s) for s in split_strings])
    assert min_len == max_len, f'Lengths of strings are not the same: {split_strings}.'

    shared_string = []
    for i in range(min_len):
        # Get the words at the current position for all strings
        words = [s[i] for s in split_strings]

        if len(set(words)) == 1:
            shared_string.append(words[0])
        else:
            shared_string.append('x')

    return '.'.join(shared_string)


def generate_Nautilus_yaml(config_paths, resource_dict, n_jobs=1, combined_yaml=False):
    """Generate yaml file for a config_path.
    See behavior_cv_training_job_combination.
    config_path: either a Path/string, or a list of Path/string.
    """
    memory, cpu, gpu = resource_dict['memory'], resource_dict['cpu'], resource_dict['gpu']
    if combined_yaml:
        assert isinstance(config_paths, list), f'config_path should be a list when combined_yaml is True. Got {config_paths}.'
    else:
        assert isinstance(config_paths, (Path, str)), f'config_path should be a Path/string when combined_yaml is False. Got {config_paths}.'
        config_paths = [config_paths]
    config_paths = [_standardize_path(cp) for cp in config_paths]
    job_names = [_standarize_job_name(cp) for cp in config_paths]
    combined_job_name = _find_max_shared_string(job_names)

    if n_jobs != 1:
        n_job_cmd = f'-n {n_jobs}'
    else:
        n_job_cmd = ''

    yaml = [
    'apiVersion: batch/v1',
    'kind: Job',
    'metadata:',
    f'  name: {combined_job_name}',
    'spec:',
    '  template:',
    '    spec:',
    '      containers:',
    '      - name: demo',
    '        image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp',
    '        command: ["/bin/bash"]',
    '        args:',
    '          - -c',
    '          - >-',
    '              cd /volume/cognitive_dynamics &&',
    '              pip install pandas==1.5.3 &&',
        ]
    for cp, jn in zip(config_paths, job_names):
        log_path = '/volume/logs/' + jn + '.out'
        yaml += [
    f'              python training_experiments/training_job_from_config.py -t {cp} {n_job_cmd} 1>{log_path} 2>{log_path} &&',]
    yaml[-1] = yaml[-1][:-2] # remove the last '&&'
    yaml += [
    '        volumeMounts:',
    '        - mountPath: /volume',
    '          name: mattarlab-volume',
    '        resources:',
    '          limits:',
    f'            memory: {int(memory*1.2)}Gi',
    f'            cpu: "{cpu}"',
    f'            nvidia.com/gpu: "{gpu}"',
    '          requests:',
    f'            memory: {memory}Gi',
    f'            cpu: "{cpu}"',
    f'            nvidia.com/gpu: "{gpu}"',
    '      volumes:',
    '        - name: mattarlab-volume',
    '          persistentVolumeClaim:',
    '            claimName: mattarlab-volume',
    '      restartPolicy: Never',
    '  backoffLimit: 0',
    ]
    return combined_job_name, yaml

def write_Nautilus_yaml(config_paths, exp_folder, resource_dict, n_jobs=1, combined_yaml=False):
    """Generate yaml files for all config_paths.
    See behavior_cv_training_job_combination.
    """
    os.makedirs('files/kube', exist_ok=True)
    apply_cmds = []
    delete_cmds = []
    with open(f'files/kube/apply_{exp_folder}.txt', 'a+') as apply_f:
        with open(f'files/kube/delete_{exp_folder}.txt', 'a+') as delete_f:
            if combined_yaml:
                config_paths = [config_paths]
                # config_path will iterate over the single element of config_paths; config_path is a list
                # else: config_paths will iterate over all the elements in config_paths; config_path is a string
            for config_path in config_paths:
                job_name, yaml = generate_Nautilus_yaml(config_path, resource_dict, n_jobs=n_jobs, combined_yaml=combined_yaml)
                with open(f'files/kube/{job_name}.yaml', 'w') as f:
                    for y in yaml:
                        print(y, file=f)
                apply_cmd = f'kubectl apply -f {job_name}.yaml'
                delete_cmd = f'kubectl delete -f {job_name}.yaml'
                print(apply_cmd, file=apply_f)
                print(delete_cmd, file=delete_f)
                apply_cmds.append(apply_cmd)
                delete_cmds.append(delete_cmd)
    return apply_cmds, delete_cmds


def behavior_cv_training_job_combination(base_config, config_ranges, resource_dict, n_jobs=1, combined_yaml=False, config_modifier=None,
                                         ignore_exist=False):
    """Generate all files for training jobs.

    Each job has a config file (in files/saved_models/exp_name), a yaml file (in files/kube/), and a apply/delete command (in files/kube/).
    We should run these commands manually to submit the jobs to the cluster.

    Args:
        base_config: the base config file.
        config_ranges: a dictionary of config ranges.
        resource_dict: a dictionary of resource requirements.
            e.g. {'memory': 5, 'cpu': 16, 'gpu': 0}
            memory is in Gi, cpu is in core, gpu is in number.
    """
    goto_root_dir.run()
    configs = config_control.vary_config(base_config, config_ranges, mode='combinatorial')
    config_paths = []
    for c in configs:
        config_path = Path('./files/saved_model') / c['model_path'] / 'allfold_config.pkl'
        os.makedirs(config_path.parent, exist_ok=True)
        if ignore_exist:
            training_summary_path = MODEL_SAVE_PATH / c['model_path'] / f'allfold_summary.pkl'
            if os.path.exists(training_summary_path):
                continue
        if config_modifier is not None:
            c = config_modifier(c)
        joblib.dump(c, config_path)
        config_paths.append(config_path)
    if len(config_paths) == 0:
        # print('No new config generated.')
        return
    apply_cmds, delete_cmds = write_Nautilus_yaml(config_paths, base_config['exp_folder'], resource_dict, n_jobs=n_jobs, combined_yaml=combined_yaml)
    for cmd in apply_cmds:
        print(cmd)
    return apply_cmds, delete_cmds