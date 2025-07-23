import vtk
import numpy as np
from vtk.util import numpy_support

# Input and output paths
input_path = "predicted_datasets\datapredicted_teardrop_hist.vti"
output_path = "predicted_datasetstear_drop_predicted_floor_hist.vti"

# Read the original VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_path)
reader.Update()
original_image = reader.GetOutput()

# Create a deep copy of the image data
deep_copy = vtk.vtkImageData()
deep_copy.DeepCopy(original_image)

# Get point data from the copy
point_data = deep_copy.GetPointData()

# Collect array names before modifying
array_names = [point_data.GetArray(i).GetName() for i in range(point_data.GetNumberOfArrays())]

# Clear all arrays before adding floored versions
point_data.Initialize()  # Optional: use only if you want to *remove* all original arrays

# Process and re-add each array
for name in array_names:
    print(f"Processing array: {name}")

    array = original_image.GetPointData().GetArray(name)
    vtk_data = numpy_support.vtk_to_numpy(array)
    floored_data = np.floor(vtk_data).astype(np.int32)

    vtk_floored_array = numpy_support.numpy_to_vtk(floored_data)
    vtk_floored_array.SetName(name)

    point_data.AddArray(vtk_floored_array)

# Save the modified data
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_path)
writer.SetInputData(deep_copy)
writer.Write()

print(f"✅ Saved floored histogram data to: {output_path}")
