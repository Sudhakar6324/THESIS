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
vti_dir = "/content/data"  # Change to your actual path if needed
vti_files = glob.glob(os.path.join(vti_dir, "*.vti"))
for vti_file in vti_files:
    INPUT_VTI = os.path.basename(vti_file)
    OUTPUT_VTI  = f"isosurface_{INPUT_VTI }"
    INPUT_VTI = vti_file
    OUTPUT_VTI = os.path.join(vti_dir,OUTPUT_VTI )

    print(f"Processing: {INPUT_VTI} -> {OUTPUT_VTI}")
    # === Load VTI File ===
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(INPUT_VTI)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()

    # === Load GMM Arrays (Means, Stds, Weights) ===
    means = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Mean{i}")) for i in range(3)]
    stds = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Std{i}")) for i in range(3)]
    weights = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Weight{i}")) for i in range(3)]
    print(len(means), len(means[0]), len(stds), len(stds[0]), len(weights), len(weights[0]))

    # === Reshape Arrays to 3D Grids ===
    gmm_means_3d = [arr.reshape(dims[::-1], order='F') for arr in means]
    gmm_stds_3d = [arr.reshape(dims[::-1], order='F') for arr in stds]
    gmm_weights_3d = [arr.reshape(dims[::-1], order='F') for arr in weights]
    print(len(gmm_means_3d), len(gmm_means_3d[0]), len(gmm_stds_3d), len(gmm_stds_3d[0]), len(gmm_weights_3d), len(gmm_weights_3d[0]))
    for i, mean in enumerate(gmm_means_3d):
        print(f"Component {i} shape: {mean.shape}")

    # === Cell Vertex Indexing ===
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

    # === GMM Component Sampler ===
    def sample_8d_gmm(weights, means, covs, n_samples):
        comps = np.random.choice(3, size=n_samples, p=weights)
        samples = np.empty((n_samples, 8))
        for c in range(3):
            idx = np.where(comps == c)[0]
            if len(idx) > 0:
                samples[idx] = np.random.multivariate_normal(mean=means[c], cov=covs[c], size=len(idx))
        return samples

    # === Probability Estimator ===
    def sample_crossing_probability_joint_gmm(weights, means, covs, isovalue, n_samples):
        samples = sample_8d_gmm(weights, means, covs, n_samples)
        cross = np.any(samples < isovalue, axis=1) & np.any(samples > isovalue, axis=1)
        return np.mean(cross)

    # === Main Computation Function ===
    def compute_probability_at_cell(idx):
        i, j, k = idx
        verts = get_cell_vertices(i, j, k)

        means_list = []
        covs_list = []
        weights_raw = []

        for c in range(3):
            mu_c = [gmm_means_3d[c][x, y, z] for (x, y, z) in verts]
            std_c = [gmm_stds_3d[c][x, y, z] for (x, y, z) in verts]
            cov_c = np.diag(np.square(std_c))

            means_list.append(np.array(mu_c))
            covs_list.append(cov_c)

            w_c = [gmm_weights_3d[c][x, y, z] for (x, y, z) in verts]
            w_product = np.prod(w_c)
            weights_raw.append(w_product)

        weights_array = np.array(weights_raw)
        weights_array /= np.sum(weights_array)

        prob = sample_crossing_probability_joint_gmm(weights_array, means_list, covs_list, ISOLEVEL, N_SAMPLES)
        return (i, j, k, prob)

    # === Prepare Grid and Indices ===
    output_shape = tuple(d - 1 for d in dims[::-1])
    prob_grid = np.zeros(output_shape, dtype=np.float32)
    all_indices = list(product(range(output_shape[0]),
                              range(output_shape[1]),
                              range(output_shape[2])))

    # === Run Multiprocessing ===
    if __name__ == '__main__':
        with multiprocessing.Pool() as pool:
            for i, j, k, prob in pool.imap_unordered(compute_probability_at_cell, all_indices):
                prob_grid[i, j, k] = prob

        # === Convert to VTK ImageData ===
        prob_image = vtk.vtkImageData()
        prob_image.SetDimensions(prob_grid.shape)
        prob_image.SetSpacing(image.GetSpacing())
        prob_image.SetOrigin(image.GetOrigin())

        flat = prob_grid.ravel(order='F')
        vtk_array = numpy_to_vtk(flat, deep=True)
        vtk_array.SetName("crossing_probability")
        prob_image.GetPointData().AddArray(vtk_array)
        prob_image.GetPointData().SetActiveScalars("crossing_probability")

        # === Write Output ===
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(OUTPUT_VTI)
        writer.SetInputData(prob_image)
        writer.Write()

        print(f"Saved output to {OUTPUT_VTI}")
