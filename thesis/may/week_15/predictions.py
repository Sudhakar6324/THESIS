import numpy as np
import torch
import os
import vtk
from vtkmodules.util import numpy_support
from torch import nn
from model import MyResidualSirenNet   # make sure this file is available
from utils import compute_rmse        # if you need RMSE too

# ------------------------- PSNR Function -------------------------
def compute_PSNR(arr_gt, arr_recon):
    diff = arr_gt - arr_recon
    sqd_max_diff = (np.max(arr_gt) - np.min(arr_gt)) ** 2
    snr = 10 * np.log10(sqd_max_diff / np.mean(diff ** 2))
    return snr

# ------------------------- Load VTI -------------------------
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
        arr = point_data_teacher.GetArray(var_name)
        vals = np.array([arr.GetTuple1(i) for i in range(num_pts)])

        min_val = vals.min()
        max_val = vals.max()
        min_max_list.append((min_val, max_val))

        vals = 2.0 * ((vals - min_val) / (max_val - min_val) - 0.5)

        coords_for_var = []
        for i in range(num_pts):
            pt = recon_vti.GetPoint(i)
            x, y, z = pt[0]/dims[0], pt[1]/dims[1], pt[2]/dims[2]
            var_idx_norm = j / num_vars
            coords_for_var.append([x, y, z, var_idx_norm])

        coords_list.append(np.array(coords_for_var))      # (num_pts, 4)
        values_list.append(vals.reshape(-1, 1))           # (num_pts, 1)

    coords_np = np.stack(coords_list, axis=1)   # (num_pts, num_vars, 4)
    values_np = np.stack(values_list, axis=1).squeeze()  # (num_pts, num_vars)
    return coords_np, values_np, var_names, min_max_list

# ------------------------- Model Prediction -------------------------
def recon_data(model, coords_np, num_vars, group_size, device):
    model.eval()
    num_pts = coords_np.shape[0]
    recon_all_vars = []

    for var_idx in range(num_vars):
        coords_base = coords_np[:, 0, :3]
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

    recon_np = np.stack(recon_all_vars, axis=-1)

    # Ensure shape is (num_pts, num_vars)
    if recon_np.ndim == 3 and recon_np.shape[-1] == 1:
        recon_np = recon_np.squeeze(-1)

    return recon_np
                  # (num_pts, num_vars)

# ------------------------- Save VTI -------------------------
def save_vti_like(template_vti, var_names, values, outpath, name, min_max_list=None):
    output_vti = vtk.vtkImageData()
    output_vti.DeepCopy(template_vti)
    output_vti.GetPointData().Initialize()

    for i in range(values.shape[1]):
        arr = values[:, i].copy()
        if min_max_list is not None:
            min_val, max_val = min_max_list[i]
            arr = ((arr + 1.0) / 2.0) * (max_val - min_val) + min_val

        vtk_arr = numpy_support.numpy_to_vtk(num_array=arr, deep=True)
        vtk_arr.SetName(var_names[i])
        output_vti.GetPointData().AddArray(vtk_arr)

    os.makedirs(outpath, exist_ok=True)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(outpath, f"{name}.vti"))
    writer.SetInputData(output_vti)
    writer.Write()

# ------------------------- Main -------------------------
def main():
    model_path = "models/isabel_KnowledgeDistillation_v1.pth"
    combined_vti = "predicted_Isabel_3D_gaussian.vti"
    original_vti = "Isabel_3D_gaussian.vti"
    out_vti_path = "data/"
    out_name = "isabel_knowledge_distillation_reconstruction"

    # Load data
    recon_vti = read_vti(combined_vti)
    coords_np, teacher_np, var_names, min_max_list = data_setup_variable_specific_input(recon_vti)

    original_vti_data = read_vti(original_vti)
    _, original_np, _, _ = data_setup_variable_specific_input(original_vti_data)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = MyResidualSirenNet(
        num_layers=6,
        neurons_per_layer=150,
        num_input_dim=4,
        num_output_dim=1
    )
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    recon_np = recon_data(model, coords_np, len(var_names), group_size=200000, device=device)

    # Save reconstructed VTI
    save_vti_like(recon_vti, var_names, recon_np, out_vti_path, out_name, min_max_list)

    # Denormalize for PSNR
    recon_denorm = []
    for i in range(recon_np.shape[1]):
        min_val, max_val = min_max_list[i]
        arr = ((recon_np[:, i] + 1.0) / 2.0) * (max_val - min_val) + min_val
        recon_denorm.append(arr)
    recon_denorm = np.stack(recon_denorm, axis=1)   # (num_pts, num_vars)

    # Compute PSNR vs Original
    psnr_list = []
    for i, vname in enumerate(var_names):
        psnr_val = compute_PSNR(original_np[:, i], recon_denorm[:, i])
        psnr_list.append(psnr_val)
        print(f"Variable {vname}: PSNR vs Original = {psnr_val:.2f} dB")

    print(f"\nAverage PSNR vs Original: {np.mean(psnr_list):.2f} dB")

if __name__ == "__main__":
    main()
