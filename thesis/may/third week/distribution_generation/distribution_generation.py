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
from vtkmodules.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import numpy as np

#load the data
reader=vtk.vtkXMLImageDataReader()
reader.SetFileName("Isabel_3D.vti")

reader.update()
data=reader.GetOutput()

dim=data.GetDimensions()
print(dim)

spacing=data.GetSpacing()
origin=data.GetOrigin()
print(spacing)
print(origin)

#getting the data
Number_of_array=data.GetPointData().GetNumberOfArrays()
print("no of arrays",Number_of_array)
total_data=[]
array_names=[]
for i in range(Number_of_array):
   curr_arr=data.GetPointData().GetArray(i)
   arr=vtk_to_numpy(curr_arr)
   array_names.append(data.GetPointData().GetArrayName(i))

print(array_names)

x = []
for i in range(dim[0]):
    x.append(int(i * spacing[0] + origin[0]))

y = []
for j in range(dim[1]):
    y.append(int(j * spacing[1] + origin[1]))

z = []
for k in range(dim[2]):
    z.append(int(k * spacing[2] + origin[2]))

def get_neighbours(arr, index, mode=6):
    z, y, x = index
    n = []
    max_radius = mode // 6
    remainder = mode % 6

    shape = arr.shape

    def in_bounds(z_, y_, x_):
        return 0 <= z_ < shape[0] and 0 <= y_ < shape[1] and 0 <= x_ < shape[2]

    # For each full radius shell
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

#creating new image_data
new_image_data=vtk.vtkImageData()
new_image_data.DeepCopy(data)
point_data=new_image_data.GetPointData()

if(torch.cuda.is_available()):
  device="cuda"
else:
  device="cpu"
print(device)

for name in array_names:
  curr_arr=data.GetPointData().GetArray(name)
  arr=vtk_to_numpy(curr_arr).reshape(dim[::-1])
  mean=[]
  std=[]
  for k in z:
    for j in y:
      for i in x:
        neighbours=get_neighbours(arr,(k,j,i))
        #n=torch.Tensor(neighbours)
        #n.to(device)
        mean.append(arr[k,j,i])
        t=np.std(neighbours)
        #t=torch.std(n)
        #t.cpu().numpy()
        std.append(t)
  mean_vtk = numpy_to_vtk(num_array=np.array(mean), deep=True)
  std_vtk = numpy_to_vtk(num_array=np.array(std), deep=True)
  mean_vtk.SetName(name+"_Mean")
  std_vtk.SetName(name+"_Std")
  point_data.AddArray(mean_vtk)
  point_data.AddArray(std_vtk)

point_data.RemoveArray("Pressure")
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("isabel_gaussian.vti")
writer.SetInputData(new_image_data)
writer.Write()

