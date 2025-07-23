import numpy as np
import torch
from torch import nn, optim
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import vtk
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
import random
import os
import sys
import time
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device running:', device)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        # self.enable_dropout = enable_dropout
        # self.dropout_prob = dropout_prob
        self.in_features = in_features
        # if enable_dropout:
        #     if not self.is_first:
        #         self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)


    def forward(self, x):
        x = self.linear(x)
        # if self.enable_dropout:
        #     if not self.is_first:
        #         x = self.dropout(x)
        return torch.sin(self.omega_0 * x)

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        # self.enable_dropout = enable_dropout
        # self.dropout_prob = dropout_prob
        self.features = features
        # if enable_dropout:
        #     self.dropout_1 = nn.Dropout(dropout_prob)
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)
        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        linear_1 = self.linear_1(self.weight_1*input)
        # if self.enable_dropout:
        #     linear_1 = self.dropout_1(linear_1)
        sine_1 = torch.sin(self.omega_0 * linear_1)
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)

class MyResidualSirenNet(nn.Module):
    def __init__(self, obj):
        super(MyResidualSirenNet, self).__init__()
        # self.enable_dropout = obj['enable_dropout']
        # self.dropout_prob = obj['dropout_prob']
        self.Omega_0=30
        self.n_layers = obj['n_layers']
        self.input_dim = obj['dim']
        self.output_dim = obj['total_vars']
        self.neurons_per_layer = obj['n_neurons']
        self.layers = [self.input_dim]
        for i in range(self.n_layers-1):
            self.layers.append(self.neurons_per_layer)
        self.layers.append(self.output_dim)
        self.net_layers = nn.ModuleList()
        for idx in np.arange(self.n_layers):
            layer_in = self.layers[idx]
            layer_out = self.layers[idx+1]
            ## if not the final layer
            if idx != self.n_layers-1:
                ## if first layer
                if idx==0:
                    self.net_layers.append(SineLayer(layer_in,layer_out,bias=True,is_first=idx==0))
                ## if an intermdeiate layer
                else:
                    self.net_layers.append(ResidualSineLayer(layer_in,bias=True,ave_first=idx>1,ave_second=idx==(self.n_layers-2)))
            ## if final layer
            else:
                final_linear = nn.Linear(layer_in,layer_out)
                ## initialize weights for the final layer
                with torch.no_grad():
                    final_linear.weight.uniform_(-np.sqrt(6 / (layer_in)) / self.Omega_0, np.sqrt(6 / (layer_in)) / self.Omega_0)
                self.net_layers.append(final_linear)

    def forward(self,x):
        for net_layer in self.net_layers:
            x = net_layer(x)
        return x


def size_of_network(n_layers, n_neurons, d_in, d_out, is_residual = True):
    # Adding input layer
    layers = [d_in]
    # layers = [3]

    # Adding hidden layers
    layers.extend([n_neurons]*n_layers)
    # layers = [3, 5, 5, 5]

    # Adding output layer
    layers.append(d_out)
    # layers = [3, 5, 5, 5, 1]

    # Number of steps
    n_layers = len(layers)-1
    # n_layers = 5 - 1 = 4

    n_params = 0

    # np.arange(4) = [0, 1, 2, 3]
    for ndx in np.arange(n_layers):

        # number of neurons in below layer
        layer_in = layers[ndx]

        # number of neurons in above layer
        layer_out = layers[ndx+1]

        # max number of neurons in both the layer
        og_layer_in = max(layer_in,layer_out)

        # if lower layer is the input layer
        # or the upper layer is the output layer
        if ndx==0 or ndx==(n_layers-1):
            # Adding weight corresponding to every neuron for every input neuron
            # Adding bias for every neuron in the upper layer
            n_params += ((layer_in+1)*layer_out)

        else:

            # If the layer is residual then proceed as follows as there will be more weights if residual layer is included
            if is_residual:
                # doubt in the following two lines
                n_params += (layer_in*og_layer_in)+og_layer_in
                n_params += (og_layer_in*layer_out)+layer_out

            # if the layer is non residual then simply add number of weights and biases as follows
            else:
                n_params += ((layer_in+1)*layer_out)
            #
        #
    #

    return n_params

def compute_PSNR(arrgt,arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt)-np.min(arrgt))**2
    snr = 10*np.log10(sqd_max_diff/np.mean(diff**2))
    return snr


def findMultiVariatePSNR(var_name, total_vars, actual, pred):
    # print('Printing PSNR')
    tot = 0
    psnr_list = []
    for j in range(total_vars):
        psnr = compute_PSNR(actual[:,j], pred[:,j])
        psnr_list.append(psnr)
        tot += psnr
        print(var_name[j], ' PSNR:', psnr)
    avg_psnr = tot/total_vars
    print('\nAverage psnr : ', avg_psnr)
     #this function is calculating the psnr of final epoch (or whenever it is called) of each variable and then averaging it
     #Thus individual epochs psnr is not calculated

    return psnr_list, avg_psnr

def compute_rmse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

def denormalizeValue(total_vars, to, ref):
    to_arr = np.array(to)
    for i in range(total_vars):
        min_data = np.min(ref[:, i])
        max_data = np.max(ref[:, i])
        to_arr[:, i] = (((to[:, i] * 0.5) + 0.5) * (max_data - min_data)) + min_data
    return to_arr

from argparse import Namespace

# Parameters (simulating argparse in a Jupyter Notebook)
args = Namespace(
    n_neurons=200,
    n_layers=6,
    epochs=1000,  # Required argument: Set the number of epochs
    batchsize=5096,
    lr=0.00005,
    no_decay=False,
    decay_rate=0.8,
    decay_at_interval=True,
    decay_interval=15,
    datapath='sklearn_teardrop_gmm.vti',  # Required: Set the path to your data
    outpath='./models/',
    exp_path='../logs/',
    modified_data_path='./data/',
    dataset_name='3d_data',  # Required: Set the dataset name
    vti_name='teardrop_sgmm_predicted_vti',  # Required: Name of the dataset
    vti_path='./data/'
)

print(args, end='\n\n')

# Assigning parameters to variables
LR = args.lr
BATCH_SIZE = args.batchsize
decay_rate = args.decay_rate
decay_at_equal_interval = args.decay_at_interval

decay = not args.no_decay
MAX_EPOCH = args.epochs

n_neurons = args.n_neurons
n_layers = args.n_layers + 2
decay_interval = args.decay_interval
outpath = args.outpath
exp_path = args.exp_path
datapath = args.datapath
modified_data_path = args.modified_data_path
dataset_name = args.dataset_name
vti_name = args.vti_name
vti_path = args.vti_path

# Displaying the final configuration
print(f"Learning Rate: {LR}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Decay Rate: {decay_rate}")
print(f"Max Epochs: {MAX_EPOCH}")
print(f"Number of Neurons per Layer: {n_neurons}")
print(f"Number of Layers (including input/output): {n_layers}")
print(f"Data Path: {datapath}")
print(f"Output Path: {outpath}")
print(f"Dataset Name: {dataset_name}")
print(f"Vti Name: {vti_name}")

#datasetup
reader=vtk.vtkXMLImageDataReader()
reader.SetFileName(datapath)
reader.update()
image_data=reader.GetOutput()
point_data=image_data.GetPointData()
no_of_arrays=point_data.GetNumberOfArrays()
variables=[]
for i in range(no_of_arrays):
  array=point_data.GetArrayName(i)
  variables.append(array)
  

origin=image_data.GetOrigin()
dim=image_data.GetDimensions()
spacing=image_data.GetSpacing()
num_points=dim[0]*dim[1]*dim[2]
print(num_points)




data=[]
for i in variables:
  vtk_array=point_data.GetArray(i)
  arr=vtk_to_numpy(vtk_array)
  data.append(arr)
print(data)

x = np.array(range(dim[0]))
y = np.array(range(dim[1]))
z = np.array(range(dim[2]))

def scale_to_minus1_1(arr):
    d_min = np.min(arr)
    d_max = np.max(arr)
    arr = (arr - d_min) / (d_max - d_min)  # scale to [0, 1]
    arr = arr * 2 - 1                      # scale to [-1, 1]
    return arr

# Scaling coordinates to [-1, 1]
x = scale_to_minus1_1(x)
y = scale_to_minus1_1(y)
z = scale_to_minus1_1(z)

print(np.min(x), np.max(x))

#scalling coridinates
loc=[]
for i in z:
  for j in y:
    for k in x:
      loc.append([k,j,i])
loc=np.array(loc)
print(loc.shape)
#scalling coridinates

#saclling only mean,stds between -1 and 1 not wieghts
scaled_data=data.copy()
for i in range(6):
  d_min=np.min(scaled_data[i])
  d_max=np.max(scaled_data[i])
  scaled_data[i]=(scaled_data[i]-d_min)/(d_max-d_min)
  scaled_data[i]=scaled_data[i]*2-1
scaled_data=np.array(scaled_data)


t_data=torch.from_numpy(scaled_data.T)
t_loc=torch.from_numpy(loc)


print('Dataset Name:', dataset_name)
print('Total Variables:', no_of_arrays)
print('Variables Name:', variables, end="\n\n")
print('Total Points in Data:', num_points)
print('Dimension of the Dataset:', dim)
print('Number of Dimensions:',len(dim))
print('Coordinate Tensor Shape:', t_loc.shape)
print('Scalar Values Tensor Shape:', t_data.shape)

print('\n###### Data setup is complete, now starting training ######\n')
import random
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

for n_neurons in range(150, 301, 20):
    print(f"\n{'='*40}\nTraining with {n_neurons} neurons\n{'='*40}", flush=True)

    train_dataloader = DataLoader(
        TensorDataset(t_loc, t_data),
        batch_size=BATCH_SIZE,
        pin_memory=True,
        shuffle=True,
        num_workers=4
    )

    obj = {
        'total_vars': no_of_arrays,
        'dim': len(dim),
        'n_neurons': n_neurons,
        'n_layers': n_layers
    }
    print(obj, flush=True)

    model = MyResidualSirenNet(obj).to(device)
    print(model, flush=True)

    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
    print(optimizer, flush=True)

    criterion = nn.MSELoss()
    print(criterion, flush=True)

    group_size = 5000
    univariate = None

    train_loss_list = []
    best_epoch = -1
    best_loss = 1e8

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    training_start_time = time.time()

    for epoch in range(MAX_EPOCH):
        model.train()
        temp_loss_list = []
        epoch_start_time = time.time()

        for X_train, y_train in train_dataloader:
            X_train = X_train.type(torch.float32).to(device)
            y_train = y_train.type(torch.float32).to(device)

            if univariate:
                y_train = y_train.squeeze()

            optimizer.zero_grad()
            predictions = model(X_train).squeeze()

            pred_reg = predictions[:, :6]
            pred_cls_logits = predictions[:, 6:]
            target_reg = y_train[:, :6]
            target_cls_probs = y_train[:, 6:]

            regression_loss = F.mse_loss(pred_reg, target_reg)
            log_probs = F.log_softmax(pred_cls_logits, dim=1)
            classification_loss = F.kl_div(log_probs, target_cls_probs, reduction='batchmean')
            loss = 0.75* regression_loss + 0.25 * classification_loss

            loss.backward()
            optimizer.step()

            temp_loss_list.append(loss.detach().cpu().numpy())

        epoch_loss = np.average(temp_loss_list)

        # Manual LR decay
        if decay:
            if decay_at_equal_interval and epoch >= decay_interval and epoch % decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_rate
            elif epoch > 0 and epoch_loss > train_loss_list[-1]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= decay_rate

        train_loss_list.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1

        epoch_end_time = time.time()
        print(
            f"Epoch: {epoch + 1}/{MAX_EPOCH} | Train Loss: {epoch_loss:.6f} | "
            f"Time: {round(epoch_end_time - epoch_start_time, 2)}s | LR: {optimizer.param_groups[0]['lr']}",
            flush=True
        )

    # Final summary
    print(f"\n[FINAL] Best Epoch: {best_epoch} | Best Loss: {best_loss:.6f}", flush=True)

    # Save final model (only once per n_neurons config)
    final_model_name = f'siren_compressor_{n_neurons}n'
    torch.save(
        {"epoch": MAX_EPOCH, "model_state_dict": model.state_dict()},
        os.path.join(outpath, f'{final_model_name}.pth')
    )

    # Final prediction for PSNR evaluation
    prediction_list = [[] for _ in range(no_of_arrays)]
    with torch.no_grad():
        for i in range(0, t_loc.shape[0], group_size):
            coords = t_loc[i:min(i + group_size, t_loc.shape[0])].type(torch.float32).to(device)
            vals = model(coords)
            gmm_weights = F.softmax(vals[:, 6:], dim=1)
            vals = torch.cat([vals[:, :6], gmm_weights], dim=1).to('cpu')

            for j in range(no_of_arrays):
                prediction_list[j].append(vals[:, j])

    extracted_list = [[] for _ in range(no_of_arrays)]
    for i in range(len(prediction_list[0])):
        for j in range(no_of_arrays):
            el = prediction_list[j][i].detach().numpy()
            extracted_list[j].append(el)

    for j in range(no_of_arrays):
        extracted_list[j] = np.concatenate(extracted_list[j], dtype='float32')

    n_predictions = np.array(extracted_list).T
    actual = np.array(scaled_data).T

    print(f"\n[PSNR] Evaluation for {n_neurons} neurons:", flush=True)
    findMultiVariatePSNR(variables, no_of_arrays, actual, n_predictions)
    rmse = compute_rmse(actual, n_predictions)
    print("Final RMSE:", rmse, flush=True)

    training_end_time = time.time()
    print(f"Training time for {n_neurons} neurons: {round(training_end_time - training_start_time, 2)} seconds\n", flush=True)


