# -*- coding: utf-8 -*-
"""
Unified Isosurface Probability Generator
- Runs Probabilistic Marching Cubes if filename contains "gaussian"
- Runs GMM-based estimation if filename contains "gmm"
- Processes all .vti files in a given folder
"""

import os
import argparse
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from joblib import Parallel, delayed
from tqdm import tqdm


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


# ================================
#   METHOD 1: GAUSSIAN
# ================================
def process_gaussian(file_path, output_path, isolevel, n_samples, n_jobs):
    print(f"\n[GAUSSIAN] Processing {file_path}")

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()

    mean_arr = vtk_to_numpy(image.GetPointData().GetArray(0))
    std_arr  = vtk_to_numpy(image.GetPointData().GetArray(1))

    mean_3d = mean_arr.reshape(dims[::-1])
    std_3d  = std_arr.reshape(dims[::-1])

    def sample_crossing_probability(mu, sigma, isovalue, n_samples):
        samples = mu + sigma * np.random.randn(n_samples, 8)
        mask = (samples.min(axis=1) < isovalue) & (samples.max(axis=1) > isovalue)
        return mask.mean()

    def process_cell(i, j, k):
        verts = get_cell_vertices(i, j, k)
        mu = np.array([mean_3d[x, y, z] for (x, y, z) in verts])
        sigma = np.array([std_3d[x, y, z] for (x, y, z) in verts])
        prob = sample_crossing_probability(mu, sigma, isolevel, n_samples)
        return (i, j, k, prob)

    output_shape = tuple(d - 1 for d in mean_3d.shape)
    prob_grid = np.zeros(output_shape, dtype=np.float32)

    tasks = [(i, j, k) for i in range(output_shape[0])
                       for j in range(output_shape[1])
                       for k in range(output_shape[2])]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i, j, k) for (i, j, k) in tqdm(tasks, desc="Gaussian Cells")
    )

    for i, j, k, prob in results:
        prob_grid[i, j, k] = prob

    prob_image = vtk.vtkImageData()
    prob_image.SetDimensions(prob_grid.shape)
    prob_image.SetSpacing(image.GetSpacing())
    prob_image.SetOrigin(image.GetOrigin())

    flat = prob_grid.ravel(order='F')
    vtk_array = numpy_to_vtk(flat, deep=True)
    vtk_array.SetName("crossing_probability")
    prob_image.GetPointData().AddArray(vtk_array)
    prob_image.GetPointData().SetActiveScalars("crossing_probability")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(prob_image)
    writer.Write()
    print(f"[GAUSSIAN] Saved to {output_path}")


# ================================
#   METHOD 2: GMM
# ================================
def process_gmm(file_path, output_path, isolevel, n_samples, n_jobs):
    print(f"\n[GMM] Processing {file_path}")

    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    image = reader.GetOutput()
    dims = image.GetDimensions()

    means = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Mean{i}")).reshape(dims[::-1]) for i in range(3)]
    stds = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Std{i}")).reshape(dims[::-1]) for i in range(3)]
    weights = [vtk_to_numpy(image.GetPointData().GetArray(f"GMM_Weight{i}")).reshape(dims[::-1]) for i in range(3)]

    def sample_8d_gmm(weights, means, stds, n_samples):
        comps = np.random.choice(3, size=n_samples, p=weights)
        samples = np.empty((n_samples, 8), dtype=np.float32)
        for c in range(3):
            idx = np.where(comps == c)[0]
            if len(idx) > 0:
                mu = means[c]
                sigma = stds[c]
                samples[idx] = np.random.normal(loc=mu, scale=sigma, size=(len(idx), 8))
        return samples

    def sample_crossing_probability_joint_gmm(weights, means, stds, isovalue, n_samples):
        samples = sample_8d_gmm(weights, means, stds, n_samples)
        mins = samples.min(axis=1)
        maxs = samples.max(axis=1)
        return np.mean((mins < isovalue) & (maxs > isovalue))

    def process_cell(i, j, k):
        verts = get_cell_vertices(i, j, k)
        means_list, stds_list, weights_raw = [], [], []
        for c in range(3):
            mu_c = [means[c][x, y, z] for (x, y, z) in verts]
            std_c = [stds[c][x, y, z] for (x, y, z) in verts]
            means_list.append(np.array(mu_c))
            stds_list.append(np.array(std_c))
            w_c = [weights[c][x, y, z] for (x, y, z) in verts]
            weights_raw.append(np.prod(w_c))
        weights_array = np.array(weights_raw)
        weights_array /= np.sum(weights_array)
        prob = sample_crossing_probability_joint_gmm(weights_array, means_list, stds_list, isolevel, n_samples)
        return (i, j, k, prob)

    output_shape = tuple(d - 1 for d in dims[::-1])
    prob_grid = np.zeros(output_shape, dtype=np.float32)

    tasks = [(i, j, k) for i in range(output_shape[0])
                       for j in range(output_shape[1])
                       for k in range(output_shape[2])]

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cell)(i, j, k) for (i, j, k) in tqdm(tasks, desc="GMM Cells")
    )

    for i, j, k, prob in results:
        prob_grid[i, j, k] = prob

    prob_image = vtk.vtkImageData()
    prob_image.SetDimensions(prob_grid.shape)
    prob_image.SetSpacing(image.GetSpacing())
    prob_image.SetOrigin(image.GetOrigin())

    flat = prob_grid.ravel(order='F')
    vtk_array = numpy_to_vtk(flat, deep=True)
    vtk_array.SetName("crossing_probability")
    prob_image.GetPointData().AddArray(vtk_array)
    prob_image.GetPointData().SetActiveScalars("crossing_probability")

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(prob_image)
    writer.Write()
    print(f"[GMM] Saved to {output_path}")


# ================================
#   MAIN DRIVER
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isosurface Probability Generator")
    parser.add_argument("--input", type=str, required=True, help="Input folder with .vti files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for results")
    parser.add_argument("--isolevel", type=float, required=True, help="Isovalue for surface extraction")
    parser.add_argument("--samples", type=int, default=200, help="Monte Carlo samples per cell")
    args = parser.parse_args()
    jobs=-1
    os.makedirs(args.output, exist_ok=True)
    vti_files = [f for f in os.listdir(args.input) if f.endswith(".vti")]

    for f in vti_files:
        file_path = os.path.join(args.input, f)
        out_name = os.path.splitext(f)[0] + "_isosurface.vti"
        out_path = os.path.join(args.output, out_name)

        if "gaussian" in f.lower():
            process_gaussian(file_path, out_path, args.isolevel, args.samples, jobs)
        elif "gmm" in f.lower():
            process_gmm(file_path, out_path, args.isolevel, args.samples, jobs)
        else:
            print(f"[SKIP] {f} (no matching method)")
