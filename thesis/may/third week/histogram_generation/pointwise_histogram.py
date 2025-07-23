import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import random
from tqdm import tqdm
# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
# Parameters
num_samples = 2000  # Number of samples per grid point
num_bins = 10       # Number of bins for histograms
vti_file_path = "/content/isabel_gaussian.vti"
output_vti_file = "histogram_dataset.vti"

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
# Step 4: Generate Histograms for Each Variable
########################################

for var in variables:
    print(f"\nProcessing variable: {var}")

    mean_array = vtk_to_numpy(pdata.GetArray(f"{var}_Mean"))
    std_array = vtk_to_numpy(pdata.GetArray(f"{var}_Std"))

    assert mean_array.shape[0] == n_pts, f"Mismatch in {var}_Mean"
    assert std_array.shape[0] == n_pts, f"Mismatch in {var}_Std"

    histograms = np.zeros((n_pts, num_bins), dtype=np.int32)

    for i in tqdm(range(n_pts)):
        mean = mean_array[i]
        std = std_array[i]
        samples = np.random.normal(mean, std, num_samples)
        hist, _ = np.histogram(samples, bins=num_bins)
        histograms[i, :] = hist

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
