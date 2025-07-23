import vtk
import numpy as np

def read_vtk_array(filename, array_name):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    image_data = reader.GetOutput()
    array = image_data.GetPointData().GetArray(array_name)
    return np.array([array.GetValue(i) for i in range(array.GetNumberOfTuples())])

def print_stats(arr, label):
    print(f"  ➤ {label}:")
    print(f"     min   : {np.nanmin(arr):.4f}")
    print(f"     max   : {np.nanmax(arr):.4f}")
    print(f"     mean  : {np.nanmean(arr):.4f}")
    print(f"     NaNs  : {np.isnan(arr).sum()}")
    print("")

# === Input files ===
file_pointwise = "gmm_modelling\isabel_gmm_batched_2.vti"
file_batched = "gmm_modelling\isabel_gmm_batched.vti"

# === Variable to analyze ===
var = "Pressure"  # Change as needed

# === GMM parameter suffixes ===
params = [
    "GMM_Mean0", "GMM_Mean1", "GMM_Mean2",
    "GMM_Std0",  "GMM_Std1",  "GMM_Std2",
    "GMM_Weight0", "GMM_Weight1", "GMM_Weight2"
]

print(f"🔍 Summary stats for variable: {var}\n")

for param in params:
    array_name = f"{var}_{param}"
    arr_point = read_vtk_array(file_pointwise, array_name)
    arr_batch = read_vtk_array(file_batched, array_name)

    print(f"📌 {array_name}")
    print_stats(arr_point, "2nd batch EM")
    print_stats(arr_batch, "Batched EM")
