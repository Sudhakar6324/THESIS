"""It takes a Gaussian VTI file and produces a VTI file containing a GMM with 3 components(9 variables)."""
"""Place the correct path of the input VTI file, which contains mean and std as variables, and use the command python3 batched_gmm.py to run."""
"""Note the variables in the VTI file should be in the form of variableName_Mean (e.g., Pressure_Mean and Pressure_Std) in the VTI file."""
import numpy as np
import torch
import vtk
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42)  

def batched_fit_1d_gmm(data, n_components=3, max_iter=50, tol=1e-6):
    """
    Fit 1D GMMs in batch for multiple 1D datasets.
    Input:
        data: shape (B, N) — B batches of N samples
    Output:
        weights, means, stds: each of shape (B, K)
    """
    B, N = data.shape
    K = n_components
    torch.manual_seed(123)

    # Initialization
    data_mean = data.mean(dim=1, keepdim=True)         # (B,1)
    data_std = data.std(dim=1, keepdim=True) + 1e-6     # (B,1)

    weights = torch.ones(B, K, device=device) / K       # (B,K)
    means = torch.randn(B, K, device=device) * data_std + data_mean  # (B,K)
    stds = data_std.repeat(1, K)                        # (B,K)

    prev_log_likelihood = torch.full((B,), -np.inf, device=device)

    for iteration in range(max_iter):
        data_exp = data.unsqueeze(2)        # (B,N,1)
        means_exp = means.unsqueeze(1)      # (B,1,K)
        stds_exp = stds.unsqueeze(1)        # (B,1,K)
        weights_exp = weights.unsqueeze(1)  # (B,1,K)

        # E-step: Log-probabilities
        log_probs = -0.5 * np.log(2 * np.pi) - torch.log(stds_exp) - 0.5 * ((data_exp - means_exp) ** 2 / stds_exp ** 2)
        log_probs += torch.log(weights_exp)  # (B,N,K)

        # Normalize using log-sum-exp
        max_log = torch.max(log_probs, dim=2, keepdim=True)[0]  # (B,N,1)
        log_probs_norm = log_probs - max_log
        resp_unnorm = torch.exp(log_probs_norm)  # (B,N,K)
        resp_denom = torch.sum(resp_unnorm, dim=2, keepdim=True)  # (B,N,1)
        responsibilities = resp_unnorm / resp_denom  # (B,N,K)

        # M-step
        Nk = responsibilities.sum(dim=1)  # (B,K)
        weights = Nk / N  # (B,K)

        means = (responsibilities * data_exp).sum(dim=1) / Nk  # (B,K)
        var = (responsibilities * (data_exp - means.unsqueeze(1)) ** 2).sum(dim=1) / Nk  # (B,K)
        stds = torch.sqrt(var.clamp(min=1e-8))  # (B,K)

        # Log-likelihood
        log_likelihood = torch.sum(torch.log(torch.sum(torch.exp(log_probs_norm), dim=2)) + max_log.squeeze(2), dim=1)
        if torch.max(torch.abs(log_likelihood - prev_log_likelihood)) < tol:
            break
        prev_log_likelihood = log_likelihood

    return weights, means, stds
def create_vtk_array(name, num_points):
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(num_points)
    return arr

scale_factor = 2.0
n_samples_per_point = 500
batch_size = 512

input_filename = "/content/isabel_gaussian.vti"
output_filename = "isabel_gmm_batched.vti"

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_filename)
reader.Update()
image_data = reader.GetOutput()
dims = image_data.GetDimensions()
num_points = dims[0] * dims[1] * dims[2]

pd = image_data.GetPointData()
array_names = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
variables = {name.replace("_Mean", "") for name in array_names if name.endswith("_Mean") and (name.replace("_Mean", "") + "_Std") in array_names}

print(f"Detected variables: {variables}")

for var in variables:
    print(f"\nProcessing variable: {var}")
    mean_array_vtk = pd.GetArray(var + "_Mean")
    std_array_vtk = pd.GetArray(var + "_Std")

    # Generate samples for all points
    all_samples = np.zeros((num_points, n_samples_per_point), dtype=np.float32)
    for pt_id in range(num_points):
        mu = mean_array_vtk.GetValue(pt_id)
        stdv = std_array_vtk.GetValue(pt_id)
        scaled_std = scale_factor * stdv
        all_samples[pt_id] = np.random.normal(loc=mu, scale=scaled_std, size=n_samples_per_point)
    
    all_samples_tensor = torch.tensor(all_samples, device=device)

    # Prepare output arrays
    out_arrays = {
        "mean0": create_vtk_array(var + "_GMM_Mean0", num_points),
        "mean1": create_vtk_array(var + "_GMM_Mean1", num_points),
        "mean2": create_vtk_array(var + "_GMM_Mean2", num_points),
        "std0": create_vtk_array(var + "_GMM_Std0", num_points),
        "std1": create_vtk_array(var + "_GMM_Std1", num_points),
        "std2": create_vtk_array(var + "_GMM_Std2", num_points),
        "w0": create_vtk_array(var + "_GMM_Weight0", num_points),
        "w1": create_vtk_array(var + "_GMM_Weight1", num_points),
        "w2": create_vtk_array(var + "_GMM_Weight2", num_points),
    }

    # Process in batches
    for i in tqdm(range(0, num_points, batch_size)):
        batch = all_samples_tensor[i:i+batch_size]
        weights, means, stds = batched_fit_1d_gmm(batch, n_components=3)
        
        # Sort by mean
        order = torch.argsort(means, dim=1)
        means = torch.gather(means, 1, order)
        stds = torch.gather(stds, 1, order)
        weights = torch.gather(weights, 1, order)

        for j in range(batch.shape[0]):
            idx = i + j
            out_arrays["mean0"].SetValue(idx, means[j, 0].item())
            out_arrays["mean1"].SetValue(idx, means[j, 1].item())
            out_arrays["mean2"].SetValue(idx, means[j, 2].item())
            out_arrays["std0"].SetValue(idx, stds[j, 0].item())
            out_arrays["std1"].SetValue(idx, stds[j, 1].item())
            out_arrays["std2"].SetValue(idx, stds[j, 2].item())
            out_arrays["w0"].SetValue(idx, weights[j, 0].item())
            out_arrays["w1"].SetValue(idx, weights[j, 1].item())
            out_arrays["w2"].SetValue(idx, weights[j, 2].item())

    # Add to PointData
    for arr in out_arrays.values():
        pd.AddArray(arr)

    pd.RemoveArray(var + "_Mean")
    pd.RemoveArray(var + "_Std")

# Save output
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_filename)
writer.SetInputData(image_data)
writer.Write()

print(f"Done. Output written to: {output_filename}")
