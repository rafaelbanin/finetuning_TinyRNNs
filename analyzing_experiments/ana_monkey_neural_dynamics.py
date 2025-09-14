import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Union, Tuple


for dtn in ['V20161005']:#,'V20160929','V20160930','V20161017']:
    data = joblib.load(rf"D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\analysis\BartoloMonkey\neural_dynamics_modeling\{dtn}_block_type_where_select_bins_[2, 3]_combine_bins_True_pca.pkl"
        # rf"D:\OneDrive\Documents\git_repo\cognitive_dynamics\files\analysis\BartoloMonkey\neural_dynamics_modeling\{dtn}_block_type_where_select_bins_[2]_pca.pkl"
    )

# PCs = data['PC'] # (episode_num, trial_num, pc_num)
# PCs = PCs[:,:,:50]
X = data['X'] # (episode_num, trial_num, neuron_num)
var = data['behav_var'] # (episode_num, trial_num, var_num)
var_num = var.shape[-1]
dt_shape = X.shape # (episode_num, trial_num, feat_num)
episode_num, trial_num, feat_num = dt_shape
bottleneck_dim = 200
dynamics_bottleneck_dim = 10
input_num = feat_num
epochs = 1500

# Splitting data into training and testing
train_idx = np.arange(0, episode_num-2)
test_idx = np.arange(episode_num-2, episode_num)
train_data = X[train_idx]
test_data = X[test_idx]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a dictionary to store masks and mean vectors for each condition
conditions = {(0, 0): {}, (0, 1): {}, (1, 0): {}, (1, 1): {}}

# # condition-averaged neural activity after the task input
# task_input_slice= slice(0, -1)
# neural_slice = slice(1, None)

# condition-averaged neural activity before the task input
task_input_slice= slice(0, None)
neural_slice = slice(0, None)

# Generate masks and compute mean vectors for each condition
for condition, storage in conditions.items():
    action_value, reward_value = condition
    mask = np.logical_and(var[train_idx, task_input_slice, 0] == action_value, var[train_idx, task_input_slice, 1] == reward_value)
    storage['mean_vector'] = np.mean(train_data[:,neural_slice,:][mask], axis=0)

# Making predictions for the test data
predictions = np.zeros_like(test_data[:,neural_slice,:])

for i in range(test_data.shape[0]): # Loop over episodes
    for j in range(test_data[:,neural_slice].shape[1]): # Loop over trials
        condition = (var[test_idx, :, 0][i, j], var[test_idx, :, 1][i, j])
        predictions[i, j] = conditions[condition]['mean_vector']

loss = np.mean((predictions - test_data[:,neural_slice,:]) ** 2)
corr = np.corrcoef(predictions.flatten(), test_data[:,neural_slice,:].flatten())[0, 1]
print(f'Loss: {loss:.4f}, Correlation: {corr:.4f}')
syss
class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__
    def __deepcopy__(self, memo):
        new_instance = AttrDict()
        # Deepcopy all the items in the current dictionary and assign to new instance
        for key, value in self.items():
            new_instance[key] = deepcopy(value, memo)
        return new_instance


def build_network(structure: List[Union[Tuple[int, int], str]]) -> nn.Sequential:
    """Dynamically build a network layer-by-layer based on provided structure."""
    layers = []
    i = 0
    while i < len(structure):
        if isinstance(structure[i], tuple):  # add a Linear layer
            layers.append(nn.Linear(structure[i][0], structure[i][1]))
            i += 1
        elif isinstance(structure[i], str):  # If it's a string, add an activation function
            if structure[i] == 'relu':
                layers.append(nn.ReLU())
            elif structure[i] == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f'Unknown activation function: {structure[i]}')
            i += 1
        else:
            raise ValueError(f'Unknown layer type: {structure[i]}')
    return nn.Sequential(*layers)

class Autoencoder(nn.Module):
    def __init__(self, config: AttrDict):
        super(Autoencoder, self).__init__()
        self.encoder = build_network(config.encoder_structure) #[(720,100), 'relu',(100, 20)],
        self.decoder = build_network(config.decoder_structure) #[(20, 100), 'relu', (100,720)],

        self.mode = config.mode # can be 'autoencoder', 'denoising', 'beta-vae'
        if self.mode == 'beta-vae':
            self.beta = config.beta # beta value for beta-VAE
        elif self.mode == 'denoising':
            self.noise_type = config.noise_type # can be 'random' or 'zero'
            self.noise_factor = config.noise_factor # noise magnitude for denoising autoencoder
            # (0-inf for random, 0-1 for zero)
            if self.noise_type == 'zero':
                self.dropout = nn.Dropout(self.noise_factor)



    def encode(self, x: torch.Tensor):
        if self.mode == 'denoising':
            x = self.add_noise(x)

        if self.mode == 'beta-vae':
            mu, logvar = self.encoder(x).chunk(2, dim=-1)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
        else:
            mu, logvar = self.encoder(x), None
            z = mu

        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def run_dynamics(self, z, temporal_dynamics):
        temporal_input = temporal_dynamics['input']
        F = temporal_dynamics['func']
        z_new = F(z, temporal_input)
        return z_new

    def forward(self, x: torch.Tensor, temporal_dynamics=None):
        mu, logvar, z = self.encode(x)
        if temporal_dynamics is not None:
            z_new = self.run_dynamics(z, temporal_dynamics)
        else:
            z_new = z
        x_new = self.decode(z_new) # reconstructed input if no temporal dynamics, otherwise the reconstructed updated input
        return mu, logvar, z, z_new, x_new

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_type == 'random':
            return x + self.noise_factor * torch.randn_like(x)
        elif self.noise_type == 'zero':
            return self.dropout(x)
        else:
            raise ValueError(f'Unknown noise type: {self.noise_type}')

class Dynamics(nn.Module):
    def __init__(self, structure):
        super(Dynamics, self).__init__()
        self.layers = build_network(structure)

    def forward(self, x, input=None):
        if input is not None:
            x_and_input = torch.cat([x, input], dim=-1)
        else:
            x_and_input = x
        return self.layers(x_and_input) + x

class RNNdynamics(nn.Module):
    """ A RNN network: input layer + recurrent layer + a readout layer. 

    Attributes:
        input_dim:
        hidden_dim:
        output_dim:
        output_h0: whether the output of the network should contain the one from initial networks' hidden state
        rnn: the recurrent layer
        h0: the initial networks' hidden state
        readout_FC: whether the readout layer is full connected or not
        lin: the full connected readout layer
        lin_coef: the inverse temperature of a direct readout layer
    """
    def __init__(self, config):
        super(RNNdynamics, self).__init__()
        self.neural_activation_dim = config['neural_activation_dim']
        self.input_output_dim = config['input_output_dim']
        self.other_input_dim = config['other_input_dim']
        self.rnn_dim = config['rnn_dim']
        self.raw2input_layer = nn.Linear(self.neural_activation_dim, self.input_output_dim)
        self.rnn = nn.GRU(self.input_output_dim + self.other_input_dim, self.rnn_dim, batch_first=True)
        self.rnn2output_layer = nn.Linear(self.rnn_dim, self.input_output_dim)
        self.output2raw_layer = nn.Linear(self.input_output_dim, self.neural_activation_dim)
        
        self.dummy_param = nn.Parameter(torch.empty(0)) # a dummy parameter to store the device of the model

    def forward(self, neural_input, other_input, get_rnnout=False):
        model_device = self.dummy_param.device
        batch_size, seq_len, input_dim = neural_input.shape
        assert other_input.shape == (batch_size, seq_len, self.other_input_dim), f'other_input.shape: {other_input.shape} should be {(batch_size, seq_len, self.other_input_dim)}'
        raw2input = self.raw2input_layer(neural_input) # (batch_size, seq_len, input_output_dim)
        rnn_input = torch.cat((raw2input, other_input), dim=-1) # (batch_size, seq_len, input_output_dim + other_input_dim)
        rnn_out, hn = self.rnn(rnn_input) # (batch_size, seq_len, rnn_dim)
        rnn2output = self.rnn2output_layer(rnn_out) + raw2input # (batch_size, seq_len, input_output_dim)
        output2raw = self.output2raw_layer(rnn2output) # (batch_size, seq_len, neural_activation_dim)
        
        if get_rnnout:
            return output2raw, {'raw2input': raw2input, 'rnn_input': rnn_input, 'rnn_out': rnn_out, 'rnn2output': rnn2output}
        return output2raw

def wrap_plot(n_rows, n_cols, idx,curves, title, xlabel, ylabel):
    plt.subplot(n_rows, n_cols, idx)
    for curve in curves:
        plt.plot(curve['y'], label=curve['label'], color=curve['color'], linestyle=curve['linestyle'])
    plt.ylim([0, plt.ylim()[1]])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

use_gru = True
if use_gru:
    gru_dynamics = RNNdynamics({
        'neural_activation_dim': feat_num,
        'input_output_dim': bottleneck_dim,
        'other_input_dim': var_num,
        'rnn_dim': dynamics_bottleneck_dim,
    }).to(device)
    criterion_all = nn.MSELoss(reduction='none')
    criterion_mean = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(gru_dynamics.parameters(),
                                 lr=0.001)
    # Convert numpy arrays to PyTorch tensors and move to the specified device
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device)  # (episode_num-1, trial_num, feat_num)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    train_dynamics_input = torch.tensor(var[train_idx], dtype=torch.float32).to(
        device)  # (episode_num-1, trial_num, var_num)
    test_dynamics_input = torch.tensor(var[test_idx], dtype=torch.float32).to(device)

    train_dynamics_losses = []
    train_dynamics_corrs = []
    test_dynamics_losses = []
    test_dynamics_corrs = []

    def evaluate_model(gru_dynamics,
                       tensor,  # (episode_num-1, trial_num, feat_num)
                       dynamics_input=None):

        next_decoded = gru_dynamics(tensor[:, :-1], dynamics_input[:, :-1])
        dynamics_loss_all = criterion_all(next_decoded, tensor[:, 1:])
        dynamics_loss = torch.mean(dynamics_loss_all)
        dynamics_corr = torch.corrcoef(torch.stack([tensor[:, 1:].flatten(), next_decoded.flatten()], dim=0))[0, 1]
        return (next_decoded, dynamics_loss, dynamics_corr)


    for epoch in range(epochs):
        gru_dynamics.train()
        next_decoded, train_dynamics_loss, train_dynamics_corr = evaluate_model(
            gru_dynamics, train_tensor, train_dynamics_input)
        train_loss = train_dynamics_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        gru_dynamics.eval()
        with torch.no_grad():
            next_decoded, test_dynamics_loss, test_dynamics_corr = evaluate_model(
                gru_dynamics, test_tensor, test_dynamics_input)
            test_loss = test_dynamics_loss

        # Record losses
        train_dynamics_losses.append(train_dynamics_loss.item())
        train_dynamics_corrs.append(train_dynamics_corr.item())
        test_dynamics_losses.append(test_dynamics_loss.item())
        test_dynamics_corrs.append(test_dynamics_corr.item())

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train dynamics loss: {train_dynamics_loss.item():.4f}, '
                  f'Test dynamics loss: {test_dynamics_loss.item():.4f}, '
                    f'Train dynamics corr: {train_dynamics_corr.item():.4f}, '
                    f'Test dynamics corr: {test_dynamics_corr.item():.4f}'
                  )

    # plot train & test loss in each subplot
    n_rows = 1  # 4
    n_cols = 2  # 6
    plt.figure(figsize=(n_cols * 2, n_rows * 2))

    wrap_plot(n_rows, n_cols, 1, [
        {'y': train_dynamics_losses, 'label': 'train-dyn', 'color': 'C1', 'linestyle': '--'},
        {'y': test_dynamics_losses, 'label': 'test-dyn', 'color': 'C1', 'linestyle': '-'},
    ],
              'Reconstruction loss', 'Epoch', 'Loss')

    wrap_plot(n_rows, n_cols, 2, [
        {'y': train_dynamics_corrs, 'label': 'train-dyn', 'color': 'C1', 'linestyle': '--'},
        {'y': test_dynamics_corrs, 'label': 'test-dyn', 'color': 'C1', 'linestyle': '-'},
    ],
              'Reconstruction correlation', 'Epoch', 'Correlation')

else:
    ae = Autoencoder(AttrDict({
        'encoder_structure': [(input_num, bottleneck_dim), #'relu'
                              ],
        'decoder_structure': [(bottleneck_dim, input_num)],
        'mode': 'denoising',
        'noise_type': 'zero',
        'noise_factor': 0.05,
    })).to(device)
    dynamics = Dynamics(structure=[(bottleneck_dim+2, dynamics_bottleneck_dim), (dynamics_bottleneck_dim, bottleneck_dim), #'relu'
                                   ]
    ).to(device)
    
    criterion_all = nn.MSELoss(reduction='none')
    criterion_mean = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(list(ae.parameters()) +
                                 list(dynamics.parameters()),
                                 lr=0.01)
    
    # Convert numpy arrays to PyTorch tensors and move to the specified device
    train_tensor = torch.tensor(train_data, dtype=torch.float32).to(device) # (episode_num-1, trial_num, feat_num)
    test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)
    train_dynamics_input = torch.tensor(var[train_idx], dtype=torch.float32).to(device) # (episode_num-1, trial_num, var_num)
    test_dynamics_input = torch.tensor(var[test_idx], dtype=torch.float32).to(device)
    
    
    train_reconstruction_losses = []
    train_reconstruction_corrs = []
    train_dynamics_losses = []
    train_dynamics_corrs = []
    test_reconstruction_losses = []
    test_reconstruction_corrs = []
    test_dynamics_losses = []
    test_dynamics_corrs = []
    train_reconstruction_losses_pc = []
    test_reconstruction_losses_pc = []
    train_dynamics_losses_pc = []
    test_dynamics_losses_pc = []
    
    
    def evaluate_model(ae, dynamics,
                       tensor, # (episode_num-1, trial_num, feat_num)
                       dynamics_input=None):
        mu, logvar, z, z_new, x_new = ae(tensor) # no temporal dynamics
        encoded = z
        decoded = x_new
        reconstruction_loss_all = criterion_all(decoded, tensor) # (episode_num-1, trial_num, feat_num)
        reconstruction_loss_pc = torch.mean(reconstruction_loss_all, dim=(0,1)) # (feat_num,)
        reconstruction_loss = torch.mean(reconstruction_loss_all)
        reconstruction_corr = torch.corrcoef(torch.stack([tensor.flatten(), decoded.flatten()], dim=0))[0,1]
        next_encoded = z_new = ae.run_dynamics(encoded[:,:-1], {'input': dynamics_input[:,:-1], 'func': dynamics})
        next_decoded = ae.decode(next_encoded)
        dynamics_loss_all = criterion_all(next_decoded, tensor[:,1:])
        dynamics_loss_pc = torch.mean(dynamics_loss_all, dim=(0,1))
        dynamics_loss = torch.mean(dynamics_loss_all)
        dynamics_corr = torch.corrcoef(torch.stack([tensor[:,1:].flatten(), next_decoded.flatten()], dim=0))[0,1]
        return (encoded, decoded, next_encoded, next_decoded, reconstruction_loss, dynamics_loss, reconstruction_loss_pc, dynamics_loss_pc, reconstruction_corr, dynamics_corr)
    
    
    for epoch in range(epochs):
        ae.train()
        encoded, decoded, next_encoded, next_decoded, train_reconstruction_loss, train_dynamics_loss, train_reconstruction_loss_pc, train_dynamics_loss_pc, train_reconstruction_corr, train_dynamics_corr = evaluate_model(
            ae, dynamics, train_tensor, train_dynamics_input)
        train_loss = train_reconstruction_loss + train_dynamics_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    
        ae.eval()
        with torch.no_grad():
            encoded, decoded, next_encoded, next_decoded, test_reconstruction_loss, test_dynamics_loss, test_reconstruction_loss_pc, test_dynamics_loss_pc, test_reconstruction_corr, test_dynamics_corr = evaluate_model(
                ae, dynamics, test_tensor, test_dynamics_input)
            test_loss = test_reconstruction_loss + test_dynamics_loss
    
        # Record losses
        train_reconstruction_losses.append(train_reconstruction_loss.item())
        train_reconstruction_corrs.append(train_reconstruction_corr.item())
        train_dynamics_losses.append(train_dynamics_loss.item())
        train_dynamics_corrs.append(train_dynamics_corr.item())
        test_reconstruction_losses.append(test_reconstruction_loss.item())
        test_reconstruction_corrs.append(test_reconstruction_corr.item())
        test_dynamics_losses.append(test_dynamics_loss.item())
        test_dynamics_corrs.append(test_dynamics_corr.item())
        train_reconstruction_losses_pc.append(train_reconstruction_loss_pc)
        test_reconstruction_losses_pc.append(test_reconstruction_loss_pc)
        train_dynamics_losses_pc.append(train_dynamics_loss_pc)
        test_dynamics_losses_pc.append(test_dynamics_loss_pc)
    
    
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train reconstruction loss: {train_reconstruction_loss.item():.4f}, '
                    f'Train dynamics loss: {train_dynamics_loss.item():.4f}, '
                    f'Test reconstruction loss: {test_reconstruction_loss.item():.4f}, '
                    f'Test dynamics loss: {test_dynamics_loss.item():.4f}, '
                    f'Train pc loss: {train_reconstruction_loss_pc[:5]}, '
                    f'Test pc loss: {test_reconstruction_loss_pc[:5]}')
    

    # plot train & test loss in each subplot
    n_rows = 1 #4
    n_cols = 2 #6
    plt.figure(figsize=(n_cols*2, n_rows*2))
    
    wrap_plot(n_rows, n_cols, 1, [
            {'y':train_reconstruction_losses, 'label':'train', 'color':'C0', 'linestyle':'--'},
            {'y':test_reconstruction_losses, 'label':'test', 'color':'C0', 'linestyle':'-'},
            {'y':train_dynamics_losses, 'label':'train-dyn', 'color':'C1', 'linestyle':'--'},
            {'y':test_dynamics_losses, 'label':'test-dyn', 'color':'C1', 'linestyle':'-'},
        ],
      'Reconstruction loss', 'Epoch', 'Loss')
    
    wrap_plot(n_rows, n_cols, 2, [
            {'y':train_reconstruction_corrs, 'label':'train', 'color':'C0', 'linestyle':'--'},
            {'y':test_reconstruction_corrs, 'label':'test', 'color':'C0', 'linestyle':'-'},
            {'y':train_dynamics_corrs, 'label':'train-dyn', 'color':'C1', 'linestyle':'--'},
            {'y':test_dynamics_corrs, 'label':'test-dyn', 'color':'C1', 'linestyle':'-'},
        ],
        'Reconstruction correlation', 'Epoch', 'Correlation')
    # for pc_idx in range(n_rows * n_cols - 1):
    #     train_curves = [x[pc_idx].detach().cpu().numpy() for x in train_reconstruction_losses_pc]
    #     test_curves = [x[pc_idx].detach().cpu().numpy() for x in test_reconstruction_losses_pc]
    #     train_curves_dyn = [x[pc_idx].detach().cpu().numpy() for x in train_dynamics_losses_pc]
    #     test_curves_dyn = [x[pc_idx].detach().cpu().numpy() for x in test_dynamics_losses_pc]
    #     wrap_plot(n_rows, n_cols, 2+pc_idx, [
    #         {'y':train_curves, 'label':'train', 'color':'C0', 'linestyle':'--'},
    #         {'y':test_curves, 'label':'test', 'color':'C0', 'linestyle':'-'},
    #         {'y':train_curves_dyn, 'label':'train-dyn', 'color':'C1', 'linestyle':'--'},
    #         {'y':test_curves_dyn, 'label':'test-dyn', 'color':'C1', 'linestyle':'-'},
    #     ],
    #     f'Reconstruction loss PC{pc_idx}', 'Epoch', 'Loss')
    # plt.show()
    
    
    # plt.figure()
    # plt.subplot(1,3,1)
    # encoder_fc = ae.encoder[0].weight.detach().cpu().numpy() # (bottleneck_dim, feat_num)
    # max_val = np.max(np.abs(encoder_fc))
    # plt.imshow(encoder_fc, vmin=-max_val, vmax=max_val, cmap='seismic')
    # plt.colorbar()
    # plt.title('Encoder')
    # plt.ylabel('Bottleneck dim')
    # plt.xlabel('PC')
    #
    # plt.subplot(1,3,2)
    # decoder_fc = ae.decoder[0].weight.detach().cpu().numpy().T # (bottleneck_dim, feat_num)
    # max_val = np.max(np.abs(decoder_fc))
    # plt.imshow(decoder_fc, vmin=-max_val, vmax=max_val, cmap='seismic')
    # plt.colorbar()
    # plt.title('Decoder')
    # plt.ylabel('Bottleneck dim')
    # plt.xlabel('PC')
    #
    # plt.subplot(1,3,3)
    # dynamics_fc = dynamics.fc.weight.detach().cpu().numpy() # (to_bottleneck_dim, from_bottleneck_dim)
    # max_val = np.max(np.abs(dynamics_fc))
    # plt.imshow(dynamics_fc, vmin=-max_val, vmax=max_val, cmap='seismic')
    # plt.colorbar()
    # plt.title('Dynamics')
    # plt.ylabel('To Bottleneck dim')
    # plt.xlabel('From Bottleneck dim')
    # plt.show()
