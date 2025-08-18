import vtk

# Path to your .vti file
vti_file_path = r"data\isabel_GMM_week_11.vti"

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
    array = point_data.GetArray(i)
    array_name = array.GetName()
    print(f"\n{i}: {array_name}")

    num_points = image_data.GetNumberOfPoints()
    count_less_than_zero = 0

    for j in range(num_points):
        # For scalar arrays (1 component)
        value = array.GetTuple1(j)
        if value < 0:
            count_less_than_zero += 1

    print(f"Total points: {num_points}")
    print(f"Values < 0 in '{array_name}': {count_less_than_zero}")
