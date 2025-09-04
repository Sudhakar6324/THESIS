# eval_and_export_from_pth.py
"""
nohup python pth_model.py --dataset_name=isabel --combined_file=../data/combined_isabel_float32.vti --run_device=0,1 --model_path=../models_week/Extra_Samples.pth --outdata_path=../outputs_week/ --out_name=isabel_Extra_Samples>pth_model_Extra_samples.log 2>&1 &
nohup python pth_model.py --dataset_name=combustion --combined_file=../data/combined_combustion_float32.vti --run_device=0,1 --model_path=../models_week/ExtraSamples0.03_combustion.pth --outdata_path=../outputs_week/ --out_name=ExtraSamples0.03_combustion>ExtraSamples0.03_combustion.log 2>&1 &

"""
import numpy as np
import torch
import os
import argparse
import vtk
from vtkmodules.util import numpy_support
from torch import nn
from model import MyResidualSirenNet
from utils import compute_PSNR, compute_rmse

# ------------------------- ARGS -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=True)
parser.add_argument('--combined_file', required=True)
parser.add_argument('--run_device', required=True)          # e.g. "0,1" or "0"
parser.add_argument('--model_path', required=True)          # .pth you saved
parser.add_argument('--outdata_path', required=True)        # folder for vti + metrics
parser.add_argument('--out_name', default=None)             # optional vti basename
parser.add_argument('--group_size', type=int, default=200000)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.run_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------- IO HELPERS -------------------------
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
recon_vti = read_vti(args.combined_file)
coords_np, teacher_np, var_names, min_max_list = data_setup_variable_specific_input(recon_vti)
num_vars = len(var_names)

print("Loading model...")
model = MyResidualSirenNet(
    num_layers=6,
    neurons_per_layer=200,
    num_input_dim=4,
    num_output_dim=1
)
model = nn.DataParallel(model).to(device)
state = torch.load(args.model_path, map_location=device)

# (works whether saved from DataParallel or single-GPU)
model.load_state_dict(state)
model.eval()

print("Reconstructing predictions from model...")
recon_np = recon_data(model, coords_np, num_vars, args.group_size, device)
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
