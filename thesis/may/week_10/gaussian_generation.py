"""
This script generates a Gaussian distribution for each grid point in a VTI dataset.
The mean is set as the grid point's own value, and the standard deviation is computed
based on the values of its neighboring points.

Usage:
1. Set the correct file path to the input VTI data file.
2. Run the script using the command:
   python3 distribution_generation.py
"""

import vtk
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from numba import njit


# load the data
input_file="data/GT_teardrop_128x128x128.vti"
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
data = reader.GetOutput()
dim = data.GetDimensions()
print(dim)

spacing = data.GetSpacing()
origin = data.GetOrigin()
print(spacing)
print(origin)

# getting the data
Number_of_array = data.GetPointData().GetNumberOfArrays()
print("no of arrays", Number_of_array)
total_data = []
array_names = []
for i in range(Number_of_array):
    curr_arr = data.GetPointData().GetArray(i)
    arr = vtk_to_numpy(curr_arr)
    array_names.append(data.GetPointData().GetArrayName(i))

print(array_names)

x = []
for i in range(dim[0]):
    x.append(i)

y = []
for j in range(dim[1]):
    y.append(j)

z = []
for k in range(dim[2]):
    z.append(k)

@njit
def compute_mean_std(arr, x, y, z, mode):
    shape = arr.shape
    mean = []
    std = []
    global_std = np.std(arr)
    print(global_std)

    def in_bounds(z_, y_, x_):
        return 0 <= z_ < shape[0] and 0 <= y_ < shape[1] and 0 <= x_ < shape[2]

    def get_neighbours(z, y, x):
        n = []
        max_radius = mode // 6
        remainder = mode % 6

        for i in range(1, max_radius + 1):
            candidates = [
                (z, y, x - i),
                (z, y, x + i),
                (z, y - i, x),
                (z, y + i, x),
                (z - i, y, x),
                (z + i, y, x)
            ]
            for cz, cy, cx in candidates:
                if in_bounds(cz, cy, cx):
                    n.append(arr[cz, cy, cx])

        if remainder > 0:
            candidates = [
                (z, y, x - (max_radius + 1)),
                (z, y, x + (max_radius + 1)),
                (z, y - (max_radius + 1), x),
                (z, y + (max_radius + 1), x),
                (z - (max_radius + 1), y, x),
                (z + (max_radius + 1), y, x)
            ]
            for cz, cy, cx in candidates:
                if remainder == 0:
                    break
                if in_bounds(cz, cy, cx):
                    n.append(arr[cz, cy, cx])
                    remainder -= 1
        return n

    for kz in z:
        for jy in y:
            for ix in x:
                neighbours = get_neighbours(kz, jy, ix)
                mean.append(arr[kz, jy, ix])
                if len(neighbours) > 0:
                   t=np.std(np.array(neighbours))
                   if(t==0):
                     t=1e-5
                   std.append(t)
                else:
                   std.append(1e-5)
    return np.array(mean), np.array(std)

# creating new image_data
new_image_data = vtk.vtkImageData()
new_image_data.DeepCopy(data)
point_data = new_image_data.GetPointData()

if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"
print(device)

for name in array_names:
    curr_arr = data.GetPointData().GetArray(name)
    arr = vtk_to_numpy(curr_arr).reshape(dim[::-1])  # (z, y, x)

    mean, std = compute_mean_std(arr, np.array(x), np.array(y), np.array(z), mode=12)

    mean_vtk = numpy_to_vtk(num_array=mean, deep=True)
    std_vtk = numpy_to_vtk(num_array=std, deep=True)
    mean_vtk.SetName("Mean")
    std_vtk.SetName("Std")
    point_data.AddArray(mean_vtk)
    point_data.AddArray(std_vtk)

point_data.RemoveArray("teardrop")
output_file="GT_teardrop_Gaussian.vti"
output_file_path=f'data/{output_file}'
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_file_path)
writer.SetInputData(new_image_data)
writer.Write()
