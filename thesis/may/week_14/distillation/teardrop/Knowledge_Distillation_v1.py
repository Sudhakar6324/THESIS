"""
Model takes 4 ips and gives 1 output
considers only Teacher data for normalization also 

Consider AVG PSNR for metrics
"""
import numpy as np
import torch
import os
import time
import argparse
import vtk
from vtkmodules.util import numpy_support
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch.optim import lr_scheduler
from model import MyResidualSirenNet
from utils import compute_PSNR, compute_rmse
from tqdm import tqdm
# ------------------------- ARGUMENTS -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--combined_file', required=True)
parser.add_argument('--run_device', required=True)
parser.add_argument('--outpath', required=True)
parser.add_argument('--outdata_path', required=True)
parser.add_argument('--original_file', required=True)

args = parser.parse_args()

# ------------------------- SETUP -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.run_device
learning_rate = 5e-5
MAX_EPOCH = 300
BATCH_SIZE = 2048
number_layers = 6
neurons_per_layer = 150
lr_schedule_stepsize = 15
lr_gamma = 0.8
weight_decay = 0
group_size = 200000

# ------------------------- LOAD DATA -------------------------
def read_vti(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def data_setup_variable_specific_input(recon_vti):
    num_pts = recon_vti.GetNumberOfPoints()
    dims = recon_vti.GetDimensions()
    print(dims)
    point_data_teacher = recon_vti.GetPointData()
    var_names = [point_data_teacher.GetArray(i).GetName() for i in range(point_data_teacher.GetNumberOfArrays())]
    num_vars = len(var_names)
    print(var_names)
    coords_list = []
    values_list = []
    min_max_list = []

    for j, var_name in enumerate(var_names):
        teacher_arr = point_data_teacher.GetArray(var_name)

        teacher_vals = np.array(teacher_arr)
        print(teacher_vals)
        min_val = teacher_vals.min()
        max_val = teacher_vals.max()
        min_max_list.append((min_val, max_val))

        teacher_vals = 2.0 * ((teacher_vals - min_val) / (max_val - min_val) - 0.5)

        coords_for_var = []
        for i in range(num_pts):
            pt = recon_vti.GetPoint(i)
            x, y, z = pt[0]/dims[0], pt[1]/dims[1], pt[2]/dims[2]
            var_idx_norm = j / num_vars
            coords_for_var.append([x, y, z, var_idx_norm])

        coords_list.append(np.array(coords_for_var))      # shape (num_pts, 4)
        values_list.append(teacher_vals.reshape(-1, 1))    # shape (num_pts, 1)

    coords_np = np.stack(coords_list, axis=1)  # (num_pts, num_vars, 4)
    teacher_np = np.stack(values_list, axis=1).squeeze()  # (num_pts, num_vars)
    return coords_np, teacher_np, var_names, min_max_list

def recon_data(model, coords_np, num_vars, group_size, device):
    model.eval()
    num_pts = coords_np.shape[0]
    recon_all_vars = []

    for var_idx in range(num_vars):
        coords_base = coords_np[:, 0, :3]  # common spatial part
        var_idx_norm = var_idx / num_vars 
        var_input = np.full((coords_base.shape[0], 1), var_idx_norm)
        coords_with_var = np.hstack([coords_base, var_input])

        coords_tensor = torch.from_numpy(coords_with_var).float().to(device)
        out_list = []
        with torch.no_grad():
            for i in range(0, len(coords_tensor), group_size):
                x_batch = coords_tensor[i:i+group_size]
                preds = model(x_batch)
                out_list.append(preds.cpu().numpy())
        recon_all_vars.append(np.concatenate(out_list, axis=0))

    recon_np = np.stack(recon_all_vars, axis=-1)  # shape: (num_pts, num_vars)
    return recon_np

def save_volume(template_vti, var_names, values, outpath, name, min_max_list=None):
    output_vti = vtk.vtkImageData()
    output_vti.DeepCopy(template_vti)
    output_vti.Initialize()

    for i in range(values.shape[1]):
        arr = values[:, i].copy()
        if min_max_list is not None:
            min_val, max_val = min_max_list[i]
            arr = ((arr + 1.0) / 2.0) * (max_val - min_val) + min_val

        vtk_arr = numpy_support.numpy_to_vtk(num_array=arr, deep=True)
        vtk_arr.SetName(var_names[i])
        output_vti.GetPointData().AddArray(vtk_arr)

    writer = vtk.vtkXMLImageDataWriter()
    os.makedirs(outpath,exist_ok=True)
    writer.SetFileName(os.path.join(outpath, f"{name}.vti"))
    writer.SetInputData(output_vti)
    writer.Write()

# ------------------------- LOAD VTI -------------------------
recon_vti = read_vti(args.combined_file)
coords_np, teacher_np, var_names, min_max_list = data_setup_variable_specific_input(recon_vti)

# ------------------------- MODEL -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = MyResidualSirenNet(
    num_layers=number_layers,
    neurons_per_layer=neurons_per_layer,
    num_input_dim=4,
    num_output_dim=1
)
model = nn.DataParallel(model).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_schedule_stepsize, gamma=lr_gamma)
criterion = nn.MSELoss()

# ------------------------- DATALOADER -------------------------
coords_tensor = torch.from_numpy(coords_np).float()     # shape: (N, V, 4)
teacher_tensor = torch.from_numpy(teacher_np).float()   # shape: (N, V)
dataset = TensorDataset(coords_tensor, teacher_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# ------------------------- TRAINING -------------------------
print("Training started", flush=True)
t_begin = time.time()

for epoch in tqdm(range(MAX_EPOCH + 1)):
    model.train()
    losses = []

    for x_batch, y_batch in dataloader:
        B, V, _ = x_batch.shape
        x_batch = x_batch.reshape(B * V, 4).to(device)
        y_batch = y_batch.reshape(B * V, 1).to(device)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    scheduler.step()
    print(f"Epoch {epoch} | Loss: {np.mean(losses):.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)

    if epoch == MAX_EPOCH:
        #model_path = os.path.join(args.outpath, f"student_teacher_4ips_teacherloss_updated1_150.pth")
        os.makedirs(args.outpath,exist_ok=True)
        model_path = os.path.join(args.outpath, f"{args.dataset_name}_KnowledgeDistillation_version1.pth")
        torch.save(model.state_dict(), model_path)

t_end = time.time()
mins, secs = divmod(t_end - t_begin, 60)
print(f"Training complete in {int(mins)}m {int(secs)}s", flush=True)

# ------------------------- RECONSTRUCTION -------------------------
recon_np = recon_data(model, coords_np, len(var_names), group_size, device)
psnr = compute_PSNR(teacher_np.squeeze(),recon_np.squeeze())
rmse = compute_rmse(teacher_np.squeeze(),recon_np.squeeze())
print(f"PSNR: {psnr:.2f}, RMSE: {rmse:.4f}", flush=True)


original_vti = read_vti(args.original_file)  
_, original_np, _, _ = data_setup_variable_specific_input(original_vti)
psnr_vs_original = compute_PSNR(original_np.squeeze(), recon_np.squeeze())
rmse_vs_original = compute_rmse(original_np.squeeze(), recon_np.squeeze())
print(f"PSNR vs Original File: {psnr_vs_original:.2f}, RMSE: {rmse_vs_original:.4f}", flush=True)

# ------------------------- SAVE OUTPUT -------------------------
save_volume(recon_vti, var_names, recon_np.squeeze(), args.outdata_path, f"{args.dataset_name}_KnowledgeDistillation_version1", min_max_list)
