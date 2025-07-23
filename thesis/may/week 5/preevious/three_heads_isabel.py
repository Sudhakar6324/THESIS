import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import time
from argparse import Namespace
from tqdm import tqdm
import random

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print('Device running:', device, flush=True)

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)
        self.weight_1 = 0.5 if ave_first else 1
        self.weight_2 = 0.5 if ave_second else 1
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                          np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                          np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        linear_1 = self.linear_1(self.weight_1 * input)
        sine_1 = torch.sin(self.omega_0 * linear_1)
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2 * (input + sine_2)

class MyResidualSirenNet(nn.Module):
    def __init__(self, obj):
        super().__init__()
        self.Omega_0 = 30
        self.n_layers = obj['n_layers']
        self.input_dim = obj['dim']
        self.neurons_per_layer = obj['n_neurons']
        self.output_components = 3

        self.layers = [self.input_dim]
        for i in range(self.n_layers - 1):
            self.layers.append(self.neurons_per_layer)

        self.net_layers = nn.ModuleList()
        for idx in range(self.n_layers - 1):
            layer_in = self.layers[idx]
            layer_out = self.layers[idx + 1]
            if idx == 0:
                self.net_layers.append(SineLayer(layer_in, layer_out, is_first=True, omega_0=self.Omega_0))
            else:
                self.net_layers.append(ResidualSineLayer(layer_in, omega_0=self.Omega_0,
                                                         ave_first=(idx > 1), ave_second=(idx == self.n_layers - 2)))

        final_hidden_dim = self.layers[-1]
        self.mean_head = nn.Linear(final_hidden_dim, self.output_components)
        self.std_head = nn.Linear(final_hidden_dim, self.output_components)
        self.weight_head = nn.Linear(final_hidden_dim, self.output_components)

        with torch.no_grad():
            for head in [self.mean_head, self.std_head, self.weight_head]:
                head.weight.uniform_(-np.sqrt(6 / final_hidden_dim) / self.Omega_0,
                                     np.sqrt(6 / final_hidden_dim) / self.Omega_0)

    def forward(self, x):
        for layer in self.net_layers:
            x = layer(x)
        mu = self.mean_head(x)
        std = F.softplus(self.std_head(x)) + 1e-8
        weights = F.softmax(self.weight_head(x), dim=-1)
        return mu, std, weights

def compute_PSNR(arrgt, arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt) - np.min(arrgt)) ** 2
    snr = 10 * np.log10(sqd_max_diff / np.mean(diff ** 2))
    return snr

def findMultiVariatePSNR(var_name, total_vars, actual, pred):
    tot = 0
    psnr_list = []
    for j in range(total_vars):
        psnr = compute_PSNR(actual[:, j], pred[:, j])
        psnr_list.append(psnr)
        tot += psnr
        print(var_name[j], ' PSNR:', psnr, flush=True)
    avg_psnr = tot / total_vars
    print('\nAverage psnr : ', avg_psnr, flush=True)
    return psnr_list, avg_psnr

def compute_rmse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

def scale_to_minus1_1(arr):
    d_min = np.min(arr)
    d_max = np.max(arr)
    return (arr - d_min) / (d_max - d_min) * 2 - 1

args = Namespace(
    n_neurons=200,
    n_layers=6,
    epochs=600,
    batchsize=512,
    lr=0.00001,
    no_decay=False,
    decay_rate=0.8,
    decay_at_interval=True,
    decay_interval=15,
    datapath='sklearn_isabel_gmm_multi.vti',
    outpath='./models/',
    exp_path='../logs/',
    modified_data_path='./data/',
    dataset_name='3d_data',
    vti_name='isabel_sgmm_predicted_vti',
    vti_path='./data/'
)

print(args, flush=True)

# Reader setup
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(args.datapath)
reader.Update()
image_data = reader.GetOutput()
point_data = image_data.GetPointData()
no_of_arrays = point_data.GetNumberOfArrays()

variables = [point_data.GetArrayName(i) for i in range(no_of_arrays)]
print('Variables:', variables, flush=True)

origin = image_data.GetOrigin()
dim = image_data.GetDimensions()
spacing = image_data.GetSpacing()
num_points = dim[0] * dim[1] * dim[2]
print('Total Points:', num_points, flush=True)

data = [vtk_to_numpy(point_data.GetArray(var)) for var in variables]
x = scale_to_minus1_1(np.array(range(dim[0])))
y = scale_to_minus1_1(np.array(range(dim[1])))
z = scale_to_minus1_1(np.array(range(dim[2])))

loc = np.array([[k, j, i] for i in z for j in y for k in x])
scaled_data = data.copy()
for i in range(6):  # Scale only means and stds
    d_min = np.min(scaled_data[i])
    d_max = np.max(scaled_data[i])
    scaled_data[i] = (scaled_data[i] - d_min) / (d_max - d_min)

scaled_data = np.array(scaled_data).T

t_data = torch.from_numpy(scaled_data)
t_loc = torch.from_numpy(loc)

print('Dataset Name:', args.dataset_name, flush=True)
print('Total Variables:', no_of_arrays, flush=True)
print('Total Points in Data:', num_points, flush=True)
print('Coordinate Tensor Shape:', t_loc.shape, flush=True)
print('Scalar Values Tensor Shape:', t_data.shape, flush=True)

print('\n###### Data setup is complete, now starting training ######\n', flush=True)

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

train_dataloader = DataLoader(
    TensorDataset(t_loc, t_data),
    batch_size=args.batchsize,
    pin_memory=True,
    shuffle=True,
    num_workers=4
)

obj = {
    'total_vars': no_of_arrays,
    'dim': len(dim),
    'n_neurons': args.n_neurons,
    'n_layers': args.n_layers + 2
}
model = MyResidualSirenNet(obj).to(device)
print(model, flush=True)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()
print(optimizer, flush=True)
print(criterion, flush=True)

train_loss_list = []
best_epoch = -1
best_loss = 1e8
decay = not args.no_decay

if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)

for epoch in tqdm(range(args.epochs)):
    model.train()
    temp_loss_list = []
    start = time.time()

    for X_train, y_train in train_dataloader:
        X_train = X_train.float().to(device)
        y_train = y_train.float().to(device)

        optimizer.zero_grad()
        mu, std, weights = model(X_train)
        predictions = torch.cat([mu, std, weights], dim=-1)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        temp_loss_list.append(loss.item())

    epoch_loss = np.mean(temp_loss_list)

    if decay:
        if args.decay_at_interval:
            if epoch >= args.decay_interval and epoch % args.decay_interval == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.decay_rate
        elif epoch > 0 and epoch_loss > train_loss_list[-1]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.decay_rate

    train_loss_list.append(epoch_loss)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch + 1

    end = time.time()
    print(f"Epoch: {epoch + 1}/{args.epochs} | Train Loss: {epoch_loss:.6f} | "
          f"Time: {round(end - start, 2)}s ({device}) | LR: {optimizer.param_groups[0]['lr']:.2e}", flush=True)

    if (epoch + 1) % 50 == 0:
        model_name = f'train_{args.dataset_name}_{epoch + 1}ep_{args.n_layers}rb_{args.n_neurons}n_{args.batchsize}bs_' \
                     f'{args.lr}lr_{decay}decay_{args.decay_rate}dr_' \
                     f'{"decayingAtInterval" + str(args.decay_interval) if args.decay_at_interval else "decayingWhenLossIncr"}'
        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict()},
                   os.path.join(args.outpath, f'{model_name}.pth'))

print('\nEpoch with Least Loss:', best_epoch, '| Loss:', best_loss, flush=True)

torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()},
           os.path.join(args.outpath, 'siren_compressor.pth'))

# Inference and evaluation
group_size = 5000
prediction_list = [[] for _ in range(9)]

with torch.no_grad():
    for i in range(0, t_loc.shape[0], group_size):
        coords = t_loc[i:i + group_size].float().to(device)
        mu, std, weights = model(coords)
        vals = torch.cat([mu, std, weights], dim=-1).cpu()
        for j in range(9):
            prediction_list[j].append(vals[:, j])

# Concatenate predictions
extracted_list = [np.concatenate([batch.detach().numpy() for batch in prediction_list[j]], dtype='float32')
                  for j in range(9)]
n_predictions = np.stack(extracted_list, axis=1)

# PSNR and RMSE
psnr_list, avg_psnr = findMultiVariatePSNR(variables, 9, scaled_data, n_predictions)
rmse = compute_rmse(scaled_data, n_predictions)
print("RMSE:", rmse, flush=True)
def makeVTI(data, val, n_predictions, n_pts, total_vars, var_name, dim, isMaskPresent, mask_arr, vti_path, vti_name, normalizedVersion = False):
    nn_predictions = denormalizeValue(total_vars, n_predictions, val) if not normalizedVersion else n_predictions
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(vti_path + vti_name)
    img = vtkImageData()
    img.CopyStructure(data)
    if not isMaskPresent:
        for i in range(total_vars):
            f = var_name[i]
            temp = nn_predictions[:, i]
            arr = vtkFloatArray()
            for j in range(n_pts):
                arr.InsertNextValue(temp[j])
            arr.SetName(f)
            img.GetPointData().AddArray(arr)
        # print(img)
        writer.SetInputData(img)
        writer.Write()
        print(f'Vti File written successfully at {vti_path}{vti_name}')
    else:
        for i in range(total_vars):
            f = var_name[i]
            temp = nn_predictions[:, i]
            idx = 0
            arr = vtkFloatArray()
            for j in range(n_pts):
                if(mask_arr[j] == 1):
                    arr.InsertNextValue(temp[idx])
                    idx += 1
                else:
                    arr.InsertNextValue(0.0)
            arr.SetName('p_' + f)
            data.GetPointData().AddArray(arr)
        # print(data)
        writer.SetInputData(data)
        writer.Write()
        print(f'Vti File written successfully at {vti_path}{vti_name}',flush=True)
def getImageData(actual_img, val, n_pts, var_name, isMaskPresent, mask_arr):
    img = vtkImageData()
    img.CopyStructure(actual_img)
    # if isMaskPresent:
    #     img.DeepCopy(actual_img)
    # img.SetDimensions(dim)
    # img.SetOrigin(actual_img.GetOrigin())
    # img.SetSpacing(actual_img.GetSpacing())
    if not isMaskPresent:
        f = var_name
        data = val
        arr = vtkFloatArray()
        for j in range(n_pts):
            arr.InsertNextValue(data[j])
        arr.SetName(f)
        img.GetPointData().SetScalars(arr)
    else:
        f = var_name
        data = val
        idx = 0
        arr = vtkFloatArray()
        for j in range(n_pts):
            if(mask_arr[j] == 1):
                arr.InsertNextValue(data[idx])
                idx += 1
            else:
                arr.InsertNextValue(0.0)
        arr.SetName(f)
        img.GetPointData().SetScalars(arr)
    return img
# # vti saving path
vti_path = args.vti_path
if not os.path.exists(vti_path):
    os.makedirs(vti_path)
# vti name
vti_name = args.vti_name
isMaskPresent=False
mask_arr = []
total_vars=9
makeVTI(image_data,scaled_data, n_predictions, num_points, total_vars, variables, dim, isMaskPresent, mask_arr, vti_path, vti_name)

