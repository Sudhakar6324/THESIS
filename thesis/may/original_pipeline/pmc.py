import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from tqdm import tqdm
from itertools import product
import multiprocessing
import os
import glob

# === CONFIG ===
ISOLEVEL = 159.9798
N_SAMPLES = 10000
vti_dir = "data"

def get_cell_vertices(i, j, k):
    return [
        (i,   j,   k),
        (i+1, j,   k),
        (i+1, j+1, k),
        (i,   j+1, k),
        (i,   j,   k+1),
        (i+1, j,   k+1),
        (i+1, j+1, k+1),
        (i,   j+1, k+1),
    ]

def build_covariance(sigma, rho=0.6):
    cov = np.full((8, 8), rho)
    np.fill_diagonal(cov, 1.0)
    return (sigma[:, None] * sigma[None, :]) * cov

def sample_crossing_probability(mu, sigma, isovalue, n_samples, rho):
    cov = build_covariance(sigma, rho)
    try:
        samples = np.random.multivariate_normal(mu, cov, size=n_samples)
    except np.linalg.LinAlgError:
        samples = np.random.normal(loc=mu, scale=sigma, size=(n_samples, 8))
    cross = np.any(samples < isovalue, axis=1) & np.any(samples > isovalue, axis=1)
    return np.mean(cross)

def compute_gaussian_cell(args):
    i, j, k, mean_3d, std_3d, correlation = args
    verts = get_cell_vertices(i, j, k)
    mu = np.array([mean_3d[x, y, z] for (x, y, z) in verts], dtype=np.float64)
    sigma = np.array([std_3d[x, y, z] for (x, y, z) in verts], dtype=np.float64)
    prob = sample_crossing_probability(mu, sigma, ISOLEVEL, N_SAMPLES, correlation)
    return (i, j, k, prob)

def compute_gmm_cell(args):
    i, j, k, gmm_means_3d, gmm_stds_3d, gmm_weights_3d = args
    verts = get_cell_vertices(i, j, k)
    means_list, covs_list, weights_raw = [], [], []
    for c in range(3):
        mu_c = [gmm_means_3d[c][x, y, z] for (x, y, z) in verts]
        std_c = [gmm_stds_3d[c][x, y, z] for (x, y, z) in verts]
        cov_c = np.diag(np.square(std_c))
        means_list.append(np.array(mu_c))
        covs_list.append(cov_c)
        w_c = [gmm_weights_3d[c][x, y, z] for (x, y, z) in verts]
        weights_raw.append(np.prod(w_c))
    weights_array = np.array(weights_raw)
    weights_array /= np.sum(weights_array)
    samples = sample_8d_gmm(weights_array, means_list, covs_list, N_SAMPLES)
    cross = np.any(samples < ISOLEVEL, axis=1) & np.any(samples > ISOLEVEL, axis=1)
    return (i, j, k, np.mean(cross))

def sample_8d_gmm(weights, means, covs, n_samples):
    comps = np.random.choice(3, size=n_samples, p=weights)
    samples = np.empty((n_samples, 8))
    for c in range(3):
        idx = np.where(comps == c)[0]
        if len(idx) > 0:
            samples[idx] = np.random.multivariate_normal(mean=means[c], cov=covs[c], size=len(idx))
    return samples

def write_vti(prob_grid, spacing, origin, output_path):
    prob_image = vtk.vtkImageData()
    prob_image.SetDimensions(prob_grid.shape)
    prob_image.SetSpacing(spacing)
    prob_image.SetOrigin(origin)
    vtk_array = numpy_to_vtk(prob_grid.ravel(order='F'), deep=True)
    vtk_array.SetName("crossing_probability")
    prob_image.GetPointData().AddArray(vtk_array)
    prob_image.GetPointData().SetActiveScalars("crossing_probability")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(prob_image)
    writer.Write()
    print(f"Saved output to {output_path}")

def process_gaussian(vti_file):
    print(f"Processing Gaussian: {vti_file}")
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_file)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()
    mean_3d = vtk_to_numpy(image.GetPointData().GetArray(0)).reshape(dims[::-1], order='F')
    std_3d = vtk_to_numpy(image.GetPointData().GetArray(1)).reshape(dims[::-1], order='F')
    output_shape = tuple(d - 1 for d in mean_3d.shape)
    all_indices = [(i, j, k, mean_3d, std_3d, 0.0) for i, j, k in product(*map(range, output_shape))]
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_gaussian_cell, all_indices)
    prob_grid = np.zeros(output_shape, dtype=np.float32)
    for i, j, k, p in results:
        prob_grid[i, j, k] = p
    write_vti(prob_grid, image.GetSpacing(), image.GetOrigin(), os.path.join("isosurfaces", f"isosurface_{os.path.basename(vti_file)}"))

def process_gmm(vti_file):
    print(f"Processing GMM: {vti_file}")
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_file)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()
    gmm_means_3d = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Mean{i}")).reshape(dims[::-1], order='F') for i in range(3)]
    gmm_stds_3d = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Std{i}")).reshape(dims[::-1], order='F') for i in range(3)]
    gmm_weights_3d = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Weight{i}")).reshape(dims[::-1], order='F') for i in range(3)]
    output_shape = tuple(d - 1 for d in dims[::-1])
    all_indices = [(i, j, k, gmm_means_3d, gmm_stds_3d, gmm_weights_3d) for i, j, k in product(*map(range, output_shape))]
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_gmm_cell, all_indices)
    prob_grid = np.zeros(output_shape, dtype=np.float32)
    for i, j, k, p in results:
        prob_grid[i, j, k] = p
    write_vti(prob_grid, image.GetSpacing(), image.GetOrigin(), os.path.join("isosurfaces", f"isosurface_{os.path.basename(vti_file)}"))

if __name__ == '__main__':
    vti_files = glob.glob(os.path.join(vti_dir, "*.vti"))
    os.makedirs("isosurfaces",exist_ok=True)
    for vti_file in vti_files:
        basename = os.path.basename(vti_file).lower()
        if "gaussian" in basename:
            process_gaussian(vti_file)
        elif "gmm" in basename:
            process_gmm(vti_file)
