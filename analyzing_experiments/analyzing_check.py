"""
Function to check the losses of all the rnn models in an experiment.
In order to make sure the models are saved correctly and losses are computed correctly.
"""
from .analyzing import *
from agents import Agent, _tensor_structure_to_numpy
from tqdm import tqdm
def check_missing_models(exp_folder):
    """Check if there are missing models.

    Args:
        exp_folder (str): The folder of the experiment.
    """

    path = MODEL_SAVE_PATH / Path(exp_folder)
    summary_paths = []
    print('Checking missing models in ', path)
    for p in tqdm(path.rglob("*")): # recursively search all subfolders
        if p.name == 'allfold_config.pkl' or p.name == 'outerfold0_summary.pkl':
            # we are in the root folder of the model
            all_summary_path = p.parent / 'allfold_summary.pkl'
            if not all_summary_path.exists():
                print('WARNING: Missing summary: ', all_summary_path)
        if p.name == 'config.pkl':
            # we are in the folder of the model for a fold and a seed
            model_path = p.parent / 'model.ckpt'
            if not model_path.exists():
                print('WARNING: Missing model: ', model_path)


def check_rnn_model_losses(model_path):
    """Check the losses of a specific rnn model.

    Because during training, the losses are computed based on a mask.
    We now check the loss computed is exact the same as the loss computed without mask (i.e. the loss computed block-by-block).

    Args:
        model_path: The path of the model.
    """
    config = transform_model_format(model_path, source='path', target='config')
    ag = transform_model_format(config, source='config', target='agent')
    save_model_pass = joblib.load(model_path / 'best_pass.pkl') # the loss is computed based on the mask

    behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'], verbose=False)
    behav_dt = behav_dt.behav_to(config)
    indices = np.concatenate([config['train_index'],config['val_index'],config['test_index']],0)
    assert len(np.unique(indices)) == len(indices) # no duplicate indices

    losses_block = [] # the loss computed block-by-block
    trial_num_block = [] # the number of trials in each block
    for idx in range(len(indices)):
        data = behav_dt.get_behav_data([idx], config)
        assert data['mask'].all() # the mask should be all 1
        #print('Block ',idx, 'mask shape', data['mask'].shape)
        model_pass = ag._eval_1step(data['input'], data['target'], data['mask'])
        model_pass = _tensor_structure_to_numpy(model_pass)
        trial_num_block.append(data['mask'].shape[0])
        losses_block.append(model_pass['behav_loss'])
    losses_block = np.array(losses_block)
    trial_num_block = np.array(trial_num_block)
    total_loss_block =losses_block*trial_num_block
    print(model_path)
    for k in ['train', 'val', 'test']:
        loss_w_mask =  save_model_pass[k]['behav_loss']
        loss_wo_mask = total_loss_block[config[k + '_index']].sum() / trial_num_block[config[k + '_index']].sum()
        if not np.isclose(loss_w_mask, loss_wo_mask):
            print('--!!!Loss not equal for ', k, 'with mask: ', loss_w_mask, 'without mask: ', loss_wo_mask)


def check_rnn_models_in_exp(exp_folder):
    goto_root_dir.run()
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')

    logging_file = exp_folder + '.log.txt'
    sys.stdout = PrinterLogger(sys.stdout, open(logging_file, 'a+'))

    for i, row in rnn_summary.iterrows():
        model_path = row['model_path']
        check_rnn_model_losses(MODEL_SAVE_PATH / model_path)
