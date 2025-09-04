import vtk
import numpy as np

# ----------------------------
# Load VTI file
# ----------------------------
def load_vti(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

# ----------------------------
# Extract arrays as numpy
# ----------------------------
def vtk_to_numpy_array(vti_data, array_name):
    vtk_array = vti_data.GetPointData().GetArray(array_name)
    if vtk_array is None:
        raise ValueError(f"Array '{array_name}' not found in {vti_data}")
    return np.array([vtk_array.GetTuple1(i) for i in range(vtk_array.GetNumberOfTuples())])

# ----------------------------
# Metrics
# ----------------------------
def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    BC = np.sqrt((2 * sigma1 * sigma2) / (sigma1**2 + sigma2**2)) * \
         np.exp(-(mu1 - mu2)**2 / (4 * (sigma1**2 + sigma2**2)))
    return -np.log(BC)

def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

def kl_div(mu1, sigma1, mu2, sigma2):
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5

def sym_kl(mu1, sigma1, mu2, sigma2):
    return 0.5 * (kl_div(mu1, sigma1, mu2, sigma2) +
                  kl_div(mu2, sigma2, mu1, sigma1))

# ----------------------------
# Main
# ----------------------------
# Load your two grids
grid1 = load_vti("data\GT_teardrop_128x128x128_gaussian.vti")
grid2 = load_vti("data\predicted_GT_teardrop_128x128x128_gaussian.vti")

print("Arrays in file1:", [grid1.GetPointData().GetArrayName(i) for i in range(grid1.GetPointData().GetNumberOfArrays())])
print("Arrays in file2:", [grid2.GetPointData().GetArrayName(i) for i in range(grid2.GetPointData().GetNumberOfArrays())])

# Replace with the actual names (example: "mean", "std")
mu1 = vtk_to_numpy_array(grid1, "Mean")
sigma1 = vtk_to_numpy_array(grid1, "Std")
mu2 = vtk_to_numpy_array(grid2, "Mean")
sigma2 = vtk_to_numpy_array(grid2, "Std")

# Compute metrics for each grid point
bhatt = bhattacharyya_distance(mu1, sigma1, mu2, sigma2)
wasser = wasserstein_distance(mu1, sigma1, mu2, sigma2)
skl = sym_kl(mu1, sigma1, mu2, sigma2)

# Aggregate (average)
print("Average Bhattacharyya distance:", np.mean(bhatt))
print("Average Wasserstein distance:", np.mean(wasser))
print("Average Symmetric KL divergence:", np.mean(skl))
