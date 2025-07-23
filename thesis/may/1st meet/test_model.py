import vtk
from vtk.util import numpy_support
import pandas as pd

# Load the VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Isabel_3D.vti")
reader.Update()
image_data = reader.GetOutput()

# Get dimensions and spacing
dims = image_data.GetDimensions()
spacing = image_data.GetSpacing()
origin = image_data.GetOrigin()

# Get the scalar data
scalars = image_data.GetPointData().GetScalars()
scalar_array = numpy_support.vtk_to_numpy(scalars)

# Generate (x, y, z) coordinates based on image geometry
x_coords = [origin[0] + i * spacing[0] for i in range(dims[0])]
y_coords = [origin[1] + j * spacing[1] for j in range(dims[1])]
z_coords = [origin[2] + k * spacing[2] for k in range(dims[2])]

# Create a full grid of coordinates
data = []
index = 0
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            value = scalar_array[index]
            data.append([x, y, z, value])
            index += 1

# Create a DataFrame
df = pd.DataFrame(data, columns=["x", "y", "z", "value"])

# Save to Excel
df.to_excel("Isabel_3D_points.xlsx", index=False)
