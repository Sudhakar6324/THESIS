import vtk
import numpy as np
import os

# === Load the original VTI file ===
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("data\GT_teardrop_Gaussian.vti")  
reader.Update()

image_data = reader.GetOutput()
dims = image_data.GetDimensions()
n_points = image_data.GetNumberOfPoints()

# === Extract 'mean' and 'std' arrays from point data ===
point_data = image_data.GetPointData()
mean_array = point_data.GetArray("Mean")   # <-- replace with actual name
std_array = point_data.GetArray("Std")     # <-- replace with actual name

if mean_array is None or std_array is None:
    raise ValueError("Missing 'mean' or 'std' arrays in the input VTI file.")

# === Convert VTK arrays to NumPy ===
mean_np = np.array([mean_array.GetTuple1(i) for i in range(n_points)])
std_np  = np.array([std_array.GetTuple1(i) for i in range(n_points)])

# === Output directory ===
output_dir = "samples_vti"
os.makedirs(output_dir, exist_ok=True)

# === Generate 100 samples ===
for i in range(1, 101):
    sampled_np = np.random.normal(loc=mean_np, scale=std_np)

    # Create a new VTK float array to hold the sample
    sampled_array = vtk.vtkFloatArray()
    sampled_array.SetName("sample")
    sampled_array.SetNumberOfTuples(n_points)
    for j in range(n_points):
        sampled_array.SetValue(j, sampled_np[j])

    # Create a new vtkImageData to hold the sampled volume
    sample_data = vtk.vtkImageData()
    sample_data.DeepCopy(image_data)  # Copy geometry (extent, spacing, origin)
    sample_data.GetPointData().SetScalars(sampled_array)

    # Write to VTI
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(os.path.join(output_dir, f"sample_{i:03d}.vti"))
    writer.SetInputData(sample_data)
    writer.Write()

    print(f"Saved sample {i:03d}")
