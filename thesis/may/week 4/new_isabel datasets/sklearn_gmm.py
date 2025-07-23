import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import random
import warnings
import logging

# === Logging Setup ===
log_file = "gmm_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# === Ignore sklearn warnings (like convergence issues) ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Seed for reproducibility ===
random.seed(42)
np.random.seed(42)

# === Configuration ===
no_of_samples = 500
scale_factor = 2
input_file = "isabel_gaussian.vti"
output_file = "sklearn_isabel_gmm_gpu3.vti"

# === Load VTI Data ===
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
image_data = reader.GetOutput()
point_data = image_data.GetPointData()

origin = image_data.GetOrigin()
spacing = image_data.GetSpacing()
dims = image_data.GetDimensions()
num_points = dims[0] * dims[1] * dims[2]

print(num_points, dims, flush=True)
logger.info(f"Loaded data: {num_points} points, dimensions: {dims}")

# === Get Variable Names ===
no_of_arrays = point_data.GetNumberOfArrays()
variables = [point_data.GetArrayName(i) for i in range(no_of_arrays)]
print(variables, flush=True)
logger.info(f"Variables: {variables}")

# === Extract Mean and Std Arrays ===
mean = vtk_to_numpy(point_data.GetArray(0))
std = vtk_to_numpy(point_data.GetArray(1))
print(std.shape, flush=True)
logger.info(f"Standard deviation shape: {std.shape}")

# === Create Output Arrays ===
def create_vtk_array(name):
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(num_points)
    return arr

gmm_mean0_vtk = create_vtk_array("GMM_Mean0")
gmm_mean1_vtk = create_vtk_array("GMM_Mean1")
gmm_mean2_vtk = create_vtk_array("GMM_Mean2")
gmm_std0_vtk  = create_vtk_array("GMM_Std0")
gmm_std1_vtk  = create_vtk_array("GMM_Std1")
gmm_std2_vtk  = create_vtk_array("GMM_Std2")
gmm_w0_vtk    = create_vtk_array("GMM_Weight0")
gmm_w1_vtk    = create_vtk_array("GMM_Weight1")
gmm_w2_vtk    = create_vtk_array("GMM_Weight2")

# === Fit GMM for Each Point ===
with tqdm(total=num_points, desc="Fitting GMMs") as pbar:
    for pt_id in range(num_points):
        mu = mean[pt_id]
        sigma = std[pt_id]
        scaled_sigma = sigma * scale_factor

        samples = np.random.normal(loc=mu, scale=scaled_sigma, size=no_of_samples).reshape(-1, 1)

        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(samples)

        weights = gmm.weights_.flatten()
        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())

        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        stds = stds[order]

        gmm_mean0_vtk.SetValue(pt_id, means[0])
        gmm_mean1_vtk.SetValue(pt_id, means[1])
        gmm_mean2_vtk.SetValue(pt_id, means[2])

        gmm_std0_vtk.SetValue(pt_id, stds[0])
        gmm_std1_vtk.SetValue(pt_id, stds[1])
        gmm_std2_vtk.SetValue(pt_id, stds[2])

        gmm_w0_vtk.SetValue(pt_id, weights[0])
        gmm_w1_vtk.SetValue(pt_id, weights[1])
        gmm_w2_vtk.SetValue(pt_id, weights[2])

        if pt_id % 10000 == 0 and pt_id != 0:
            msg = f"Processed {pt_id} / {num_points} points"
            tqdm.write(msg)
            logger.info(msg)

        pbar.update(1)

# === Add Arrays to VTK ===
point_data.AddArray(gmm_mean0_vtk)
point_data.AddArray(gmm_mean1_vtk)
point_data.AddArray(gmm_mean2_vtk)
point_data.AddArray(gmm_std0_vtk)
point_data.AddArray(gmm_std1_vtk)
point_data.AddArray(gmm_std2_vtk)
point_data.AddArray(gmm_w0_vtk)
point_data.AddArray(gmm_w1_vtk)
point_data.AddArray(gmm_w2_vtk)

# === Remove Original Variables ===
point_data.RemoveArray(variables[0])
point_data.RemoveArray(variables[1])
logger.info(f"Removed arrays: {variables[0]}, {variables[1]}")

# === Write Output ===
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(image_data)
writer.Write()

print(f"Done. Output written to {output_file}", flush=True)
logger.info(f"Done. Output written to {output_file}")
