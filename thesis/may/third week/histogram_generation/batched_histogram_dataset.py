import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import torch
from tqdm import tqdm
import random
# Parameters
num_samples = 2000  # Number of samples per grid point
num_bins = 10       # Number of bins for histograms
batch_size = 10000  # Batch size for processing
vti_file_path = "/content/isabel_gaussian.vti"
output_vti_file = "histogram_dataset_batched.vti"


# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
########################################
# Step 1: Read the Input VTI File
########################################

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(vti_file_path)
reader.Update()
data = reader.GetOutput()

n_pts = data.GetNumberOfPoints()
dim = data.GetDimensions()
origin = data.GetOrigin()
spacing = data.GetSpacing()
pdata = data.GetPointData()

print(f"Number of points: {n_pts}, Dimensions: {dim}")

########################################
# Step 2: Detect Variable Pairs (XXX_Mean + XXX_Std)
########################################

array_names = [pdata.GetArrayName(i) for i in range(pdata.GetNumberOfArrays())]
variables = set()

for name in array_names:
    if name.endswith("_Mean"):
        base = name[:-5]
        if f"{base}_Std" in array_names:
            variables.add(base)

print(f"Detected variables: {variables}")

########################################
# Step 3: Create Output vtkImageData
########################################

imageData = vtk.vtkImageData()
imageData.SetDimensions(dim)
imageData.SetOrigin(origin)
imageData.SetSpacing(spacing)
print(f"Origin: {origin}, Spacing: {spacing}")

########################################
# Step 4: Generate Histograms for Each Variable (Batched on GPU)
########################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

for var in variables:
    print(f"\nProcessing variable: {var}")

    mean_array = vtk_to_numpy(pdata.GetArray(f"{var}_Mean"))
    std_array = vtk_to_numpy(pdata.GetArray(f"{var}_Std"))

    assert mean_array.shape[0] == n_pts, f"Mismatch in {var}_Mean"
    assert std_array.shape[0] == n_pts, f"Mismatch in {var}_Std"

    histograms = np.zeros((n_pts, num_bins), dtype=np.int32)

    # Estimate global min/max for consistent bins across batches
    sample_mean = mean_array[:1000].reshape(-1, 1)
    sample_std = std_array[:1000].reshape(-1, 1)
    sample_subset = np.random.normal(loc=sample_mean, scale=sample_std, size=(1000, num_samples))

    global_min = sample_subset.min()
    global_max = sample_subset.max()
    bin_edges = torch.linspace(global_min, global_max, num_bins + 1, device=device)

    for start in tqdm(range(0, n_pts, batch_size)):
        end = min(start + batch_size, n_pts)
        batch_len = end - start

        # Load batch to GPU
        mean_tensor = torch.from_numpy(mean_array[start:end]).float().to(device).unsqueeze(1)
        std_tensor = torch.from_numpy(std_array[start:end]).float().to(device).unsqueeze(1)

        # Generate samples and compute histogram
        samples = torch.normal(mean_tensor.expand(-1, num_samples), std_tensor.expand(-1, num_samples))
        digitized = torch.bucketize(samples, bin_edges) - 1
        digitized = digitized.clamp(0, num_bins - 1)

        batch_hist = torch.zeros((batch_len, num_bins), dtype=torch.int32, device=device)
        for i in range(batch_len):
            batch_hist[i] = torch.bincount(digitized[i], minlength=num_bins)

        histograms[start:end] = batch_hist.cpu().numpy()

    # Save histogram bins as separate VTK arrays
    for bin_idx in range(num_bins):
        bin_array = vtk.vtkIntArray()
        bin_array.SetNumberOfComponents(1)
        bin_array.SetNumberOfTuples(n_pts)
        bin_array.SetName(f"{var}_bin{bin_idx+1}")

        for i in range(n_pts):
            bin_array.SetTuple1(i, int(histograms[i, bin_idx]))

        imageData.GetPointData().AddArray(bin_array)

########################################
# Step 5: Write Output VTI File
########################################

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_vti_file)
writer.SetInputData(imageData)
writer.Write()

print(f"\nHistogram VTI file saved as '{output_vti_file}'")
