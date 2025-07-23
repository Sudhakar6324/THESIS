import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from tqdm import tqdm
import os

# === Configuration ===
no_of_samples = 750
scale_factor = 2
input_file = "new_isabel_week_6_gaussian.vti"    
output_file = "new_gmm_isabel_week_6.vti"   # Output filename

# === Load VTI Data ===
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
image_data = reader.GetOutput()
point_data = image_data.GetPointData()

dims = image_data.GetDimensions()
num_points = dims[0] * dims[1] * dims[2]
print(f"Loaded grid: {dims}, total points: {num_points}")

# === Get Mean and Std Arrays ===
mean = vtk_to_numpy(point_data.GetArray(0))
std = vtk_to_numpy(point_data.GetArray(1))

# === Fit GMM Per Point (Parallel) ===
def fit_gmm_for_point(mu, sigma, scale_factor, n_samples=500):
    sigma = sigma * scale_factor
    
    samples = np.random.normal(mu, sigma, size=n_samples).reshape(-1, 1)
    gmm = GaussianMixture(n_components=3, covariance_type='diag', max_iter=50, random_state=42)
    gmm.fit(samples)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    stds=np.maximum(stds,1e-8)
    weights = gmm.weights_.flatten()
    
    order = np.argsort(means)
    return means[order], stds[order], weights[order]

print("Fitting GMMs in parallel...")
results = Parallel(n_jobs=-1, backend='loky')(delayed(fit_gmm_for_point)(
    mean[i], std[i], scale_factor, no_of_samples) for i in tqdm(range(num_points)))

# === Create Output VTK Arrays ===
def create_vtk_array(name):
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(num_points)
    return arr

gmm_mean0 = create_vtk_array("GMM_Mean0")
gmm_mean1 = create_vtk_array("GMM_Mean1")
gmm_mean2 = create_vtk_array("GMM_Mean2")
gmm_std0  = create_vtk_array("GMM_Std0")
gmm_std1  = create_vtk_array("GMM_Std1")
gmm_std2  = create_vtk_array("GMM_Std2")
gmm_w0    = create_vtk_array("GMM_Weight0")
gmm_w1    = create_vtk_array("GMM_Weight1")
gmm_w2    = create_vtk_array("GMM_Weight2")

# === Fill VTK Arrays with Fitted GMM Data ===
for pt_id, (means, stds, weights) in enumerate(results):
    gmm_mean0.SetValue(pt_id, means[0])
    gmm_mean1.SetValue(pt_id, means[1])
    gmm_mean2.SetValue(pt_id, means[2])

    gmm_std0.SetValue(pt_id, stds[0])
    gmm_std1.SetValue(pt_id, stds[1])
    gmm_std2.SetValue(pt_id, stds[2])

    gmm_w0.SetValue(pt_id, weights[0])
    gmm_w1.SetValue(pt_id, weights[1])
    gmm_w2.SetValue(pt_id, weights[2])

# === Attach Arrays to VTK Data ===
point_data.AddArray(gmm_mean0)
point_data.AddArray(gmm_mean1)
point_data.AddArray(gmm_mean2)
point_data.AddArray(gmm_std0)
point_data.AddArray(gmm_std1)
point_data.AddArray(gmm_std2)
point_data.AddArray(gmm_w0)
point_data.AddArray(gmm_w1)
point_data.AddArray(gmm_w2)
point_data.RemoveArray("Pressure_Mean")
point_data.RemoveArray("Pressure_Std")
# === Write Output ===
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(image_data)
writer.Write()

print(f"Done. GMM output saved to '{output_file}'")
