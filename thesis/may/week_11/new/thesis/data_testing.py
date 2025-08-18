import vtk

# Path to your .vti file
vti_file_path = "data/new_GT_teardrop_GMM_week_11.vti"

# Read the VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(vti_file_path)
reader.Update()

# Get the image data
image_data = reader.GetOutput()

# Get point data (scalar or vector fields)
point_data = image_data.GetPointData()

# List available array (variable) names
num_arrays = point_data.GetNumberOfArrays()
print("Available variables:")
for i in range(num_arrays):
    print(f"{i}: {point_data.GetArrayName(i)}")

# Choose the variable you want to check (example: 0)
array_index = 0
array = point_data.GetArray(array_index)
array_name = array.GetName()
print(f"\nChecking variable: {array_name}")

# Count how many values are less than zero
num_points = image_data.GetNumberOfPoints()
count_less_than_zero = 0

for i in range(num_points):
    value = array.GetTuple1(i)  # For scalar arrays
    if value < 0:
        count_less_than_zero += 1

print(f"\nTotal grid points: {num_points}")
print(f"Values < 0 in '{array_name}': {count_less_than_zero}")
