import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import random
import warnings
import logging
import multiprocessing as mp
import time

# === Start timer ===
start_time = time.time()

# === Logging Setup ===
log_file = "gmm_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

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

# === Define worker function for multiprocessing ===
def _gmm_worker(pt_id):
    mu = mean[pt_id]
    sigma = std[pt_id]
    scaled_sigma = sigma * scale_factor

    try:
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

        return pt_id, means[0], means[1], means[2], stds[0], stds[1], stds[2], weights[0], weights[1], weights[2]
    except Exception as e:
        logger.warning(f"GMM failed at point {pt_id}: {str(e)}")
        return pt_id, *[np.nan]*9

# === Fit GMMs in parallel ===
processed_count = 0

with mp.Pool(processes=16) as pool, tqdm(total=num_points, desc="Fitting GMMs") as pbar:
    for result in pool.imap_unordered(_gmm_worker, range(num_points), chunksize=1000):
        pt_id, m0, m1, m2, s0, s1, s2, w0, w1, w2 = result

        gmm_mean0_vtk.SetValue(pt_id, m0)
        gmm_mean1_vtk.SetValue(pt_id, m1)
        gmm_mean2_vtk.SetValue(pt_id, m2)
        gmm_std0_vtk.SetValue(pt_id, s0)
        gmm_std1_vtk.SetValue(pt_id, s1)
        gmm_std2_vtk.SetValue(pt_id, s2)
        gmm_w0_vtk.SetValue(pt_id, w0)
        gmm_w1_vtk.SetValue(pt_id, w1)
        gmm_w2_vtk.SetValue(pt_id, w2)

        processed_count += 1

        if processed_count % 10000 == 0:
            msg = f"[Progress] Processed {processed_count} / {num_points} points"
            print(msg, flush=True)
            logger.info(msg)

        if processed_count % 100000 == 0:
            elapsed = time.time() - start_time
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            logger.info(f"[Time] Elapsed after {processed_count} points: {elapsed:.2f} sec ({elapsed_str})")

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

# === Remove Original Arrays ===
point_data.RemoveArray(variables[0])
point_data.RemoveArray(variables[1])
logger.info(f"Removed arrays: {variables[0]}, {variables[1]}")

# === Write Output ===
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(image_data)
writer.Write()

# === Final Time Logging ===
end_time = time.time()
elapsed = end_time - start_time
elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
logger.info(f"Total time taken: {elapsed:.2f} seconds ({elapsed_str})")

print(f"Done. Output written to {output_file}", flush=True)
logger.info(f"Done. Output written to {output_file}")
