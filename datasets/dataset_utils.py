
from pathlib import Path
import json
import importlib

def Dataset(dataset='', behav_data_spec=None, neuro_data_spec=None, verbose=True):
    """Get dataset instance from string.

    Args:
        dataset: A string of dataset name.
        behav_data_spec: A dict specifies which behavioral data to load in which format.
        neuro_data_spec: A dict specifies which neural data to load in which format.

    Returns:
        A instance of desired dataset with specified data loaded.
    """
    with open('data_path.json', 'r') as f:
        print(f)#################################################
        data_path_dict = json.load(f)
        data_path = Path(data_path_dict[dataset])

    # first load the dataset file/module, then get the dataset class, and return an instance of the class
    dataset_class_name = dataset + 'Dataset'
    dataset_module = importlib.import_module('.' + dataset_class_name, package='datasets')
    dataset_class = getattr(dataset_module, dataset_class_name)
    dataset_instance = dataset_class(data_path, behav_data_spec, neuro_data_spec, verbose=verbose)
    return dataset_instance
