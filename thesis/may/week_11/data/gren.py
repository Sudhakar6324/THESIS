import vtk

# Load original VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("data\GT_teardrop_128x128x128.vti")
reader.Update()

# Downsample using Shrink filter
shrink = vtk.vtkImageShrink3D()
shrink.SetInputConnection(reader.GetOutputPort())
shrink.SetShrinkFactors(16, 16, 16)  # 64 / 4 = 16
shrink.AveragingOn()
shrink.Update()

# Write the downsampled file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("downsampled_4x4x4.vti")
writer.SetInputConnection(shrink.GetOutputPort())
writer.Write()
