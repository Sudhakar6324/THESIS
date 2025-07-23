import vtk

# Step 1: Read the .vtk file
reader = vtk.vtkDataSetReader()
reader.SetFileName(r"datasets\generated\probabilistic_marching_cubes.vti")
reader.Update()

dataset = reader.GetOutput()
print(dataset)
# Step 2: Access point data
point_data = dataset.GetPointData()
num_points = dataset.GetNumberOfPoints()

# Choose the array you want to filter on (you can loop through all if needed)
scalar_array = point_data.GetScalars()
array_name = scalar_array.GetName()

# Step 3: Create a new points list and scalar array
new_points = vtk.vtkPoints()
new_scalars = vtk.vtkFloatArray()
new_scalars.SetName(array_name)

# Mapping from old point ID to new point ID
point_id_map = {}

for i in range(num_points):
    value = scalar_array.GetTuple(i)

    # Check for zero in any component
    if all(v != 0 for v in value):
        point = dataset.GetPoint(i)
        new_id = new_points.InsertNextPoint(point)
        new_scalars.InsertNextTuple(value)
        point_id_map[i] = new_id

# Step 4: Create a new PolyData (or UnstructuredGrid depending on original)
new_data = vtk.vtkPolyData()
new_data.SetPoints(new_points)
new_data.GetPointData().SetScalars(new_scalars)

# NOTE: Geometry/topology (cells) is lost here unless you remap cells

# Step 5: Write the new dataset
writer = vtk.vtkPolyDataWriter()
writer.SetFileName("filtered_output.vtk")
writer.SetInputData(new_data)
writer.Write()

print("Filtered .vtk file saved as 'filtered_output.vtk'")
