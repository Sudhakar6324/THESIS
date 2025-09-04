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

# ------------------------- ARGUMENTS -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--combined_file', required=True)
parser.add_argument('--run_device', required=True)
parser.add_argument('--outpath', required=True)
parser.add_argument('--outdata_path', required=True)
parser.add_argument('--out_name', default=None)      
parser.add_argument('--original_file', required=True)
parser.add_argument('--no_of_neurons',required=True)
args = parser.parse_args()

# ------------------------- SETUP -------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.run_device
learning_rate = 5e-5
MAX_EPOCH = 300
BATCH_SIZE = 2048
number_layers = 6
neurons_per_layer = args.no_of_neurons
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
    point_data_teacher = recon_vti.GetPointData()
    var_names = [point_data_teacher.GetArray(i).GetName() for i in range(point_data_teacher.GetNumberOfArrays())]
    num_vars = len(var_names)

    coords_list = []
    values_list = []
    min_max_list = []

    for j, var_name in enumerate(var_names):
        teacher_arr = point_data_teacher.GetArray(var_name)

        teacher_vals = np.array([teacher_arr.GetTuple1(i) for i in range(num_pts)])

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


# ------------------------- LOAD VTI -------------------------
recon_vti = read_vti(args.combined_file)
coords_np, teacher_np, var_names, min_max_list = data_setup_variable_specific_input(recon_vti)

# ------------------------- MODEL -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
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

for epoch in range(MAX_EPOCH + 1):
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
        model_path = os.path.join(args.outpath, f"{args.dataset_name}_KnowledgeDistillation_v1.pth")
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
#save_volume(recon_vti, var_names, recon_np.squeeze(), args.outdata_path, f"{args.dataset_name}_KnowledgeDistillation_v1", min_max_list)
def read_vti(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    return reader.GetOutput()

def save_volume(template_vti, var_names, values, outpath, name, min_max_list=None):
    """
    values: (num_pts, num_vars) in training scale ([-1,1]).
    If min_max_list is provided, denormalize per variable back to original teacher scale.
    """
    os.makedirs(outpath, exist_ok=True)
    num_pts = template_vti.GetNumberOfPoints()

    if values.shape[0] != num_pts:
        raise ValueError(f"values has {values.shape[0]} pts, but template has {num_pts}")

    # Copy the image structure (extent, spacing, origin) and NOT wipe it.
    out_vti = vtk.vtkImageData()
    out_vti.DeepCopy(template_vti)

    # Clear only point-data arrays so we can add our predictions
    pd = out_vti.GetPointData()
    pd.Initialize()

    for i, vname in enumerate(var_names):
        arr = values[:, i].astype(np.float32).copy()
        if min_max_list is not None:
            vmin, vmax = min_max_list[i]
            # denorm from [-1,1] -> original scale
            arr = ((arr + 1.0) * 0.5) * (vmax - vmin) + vmin
            arr = arr.astype(np.float32, copy=False)

        vtk_arr = numpy_support.numpy_to_vtk(num_array=arr, deep=True)
        vtk_arr.SetName(vname)
        pd.AddArray(vtk_arr)

    # (Optional) choose an active scalars so some tools show something by default
    if len(var_names) > 0:
        pd.SetActiveScalars(var_names[0])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outpath, f"{name}.vti"))
    writer.SetInputData(out_vti)
    writer.Write()

# ------------------------- DATA PREP (MATCH TRAINING) -------------------------
def data_setup_variable_specific_input(recon_vti):
    num_pts = recon_vti.GetNumberOfPoints()
    dims = recon_vti.GetDimensions()
    point_data_teacher = recon_vti.GetPointData()
    var_names = [point_data_teacher.GetArray(i).GetName() for i in range(point_data_teacher.GetNumberOfArrays())]
    num_vars = len(var_names)

    coords_list, values_list, min_max_list = [], [], []

    for j, var_name in enumerate(var_names):
        teacher_arr = point_data_teacher.GetArray(var_name)
        # teacher values
        teacher_vals = np.array([teacher_arr.GetTuple1(i) for i in range(num_pts)], dtype=np.float64)
        vmin, vmax = teacher_vals.min(), teacher_vals.max()
        min_max_list.append((vmin, vmax))

        # normalize teacher values to [-1, 1] (exactly like training)
        teacher_vals = 2.0 * ((teacher_vals - vmin) / (vmax - vmin) - 0.5)

        # coords: x,y,z from GetPoint(i) divided by dims[*]  (same as your training code)
        coords_for_var = []
        for i in range(num_pts):
            px, py, pz = recon_vti.GetPoint(i)
            x = px / max(dims[0], 1)
            y = py / max(dims[1], 1)
            z = pz / max(dims[2], 1)
            var_idx_norm = j / num_vars   # keep exactly as in training
            coords_for_var.append([x, y, z, var_idx_norm])

        coords_list.append(np.asarray(coords_for_var, dtype=np.float32))  # (num_pts, 4)
        values_list.append(teacher_vals.reshape(-1, 1).astype(np.float32))

    coords_np = np.stack(coords_list, axis=1)                 # (num_pts, num_vars, 4)
    teacher_np = np.stack(values_list, axis=1).squeeze(-1)    # (num_pts, num_vars)
    return coords_np, teacher_np, var_names, min_max_list

# ------------------------- RECON -------------------------
def recon_data(model, coords_np, num_vars, group_size, device):
    model.eval()
    num_pts = coords_np.shape[0]
    recon_all_vars = []

    # base spatial coords (shared)
    coords_base = coords_np[:, 0, :3]  # (num_pts, 3)

    for v in range(num_vars):
        var_idx_norm = v / num_vars
        var_col = np.full((num_pts, 1), var_idx_norm, dtype=np.float32)
        x_in = np.hstack([coords_base, var_col]).astype(np.float32)  # (num_pts, 4)

        preds_list = []
        with torch.no_grad():
            for i in range(0, num_pts, group_size):
                xb = torch.from_numpy(x_in[i:i+group_size]).to(device)
                pb = model(xb).detach().cpu().numpy()  # (chunk, 1)
                preds_list.append(pb[:, 0])
        recon_all_vars.append(np.concatenate(preds_list, axis=0))  # (num_pts,)

    recon_np = np.stack(recon_all_vars, axis=1).astype(np.float32)  # (num_pts, num_vars)
    return recon_np

# ------------------------- MAIN -------------------------
print("Loading combined VTI...")
recon_vti = read_vti(args.original_file)
coords_np, teacher_np, var_names, min_max_list = data_setup_variable_specific_input(recon_vti)
num_vars = len(var_names)

print("Loading model...")
model = MyResidualSirenNet(
    num_layers=6,
    neurons_per_layer=neurons_per_layer,
    num_input_dim=4,
    num_output_dim=1
)
model_path = os.path.join(args.outpath, f"{args.dataset_name}_KnowledgeDistillation_v1.pth")
model = nn.DataParallel(model).to(device)
state = torch.load(model_path, map_location=device)

# (works whether saved from DataParallel or single-GPU)
model.load_state_dict(state)
model.eval()

print("Reconstructing predictions from model...")
recon_np = recon_data(model, coords_np, num_vars,group_size, device)
print("Computing metrics per variable...")
for i, vname in enumerate(var_names):
    psnr_val = compute_PSNR(teacher_np[:, i], recon_np[:, i])
    rmse_val = compute_rmse(teacher_np[:, i], recon_np[:, i])
    print(f"{vname} -> PSNR: {psnr_val:.4f}, RMSE: {rmse_val:.6f}")
out_name = args.out_name or f"{args.dataset_name}_Extrasamples"
out_name = args.out_name or f"{args.dataset_name}_Extrasamples"
save_volume(
    template_vti=recon_vti,
    var_names=var_names,
    values=recon_np,                 # values are in [-1,1]
    outpath=args.outdata_path,
    name=out_name,
    min_max_list=min_max_list        # denorm back to original scale
)
print(f"Saved VTI -> {os.path.join(args.outdata_path, out_name + '.vti')}")
