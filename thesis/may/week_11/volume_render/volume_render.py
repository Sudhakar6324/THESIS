import vtk
import os
from PIL import Image, ImageChops

# === Utility Functions ===
def OpacityTransferFunction(values):
    p_f = vtk.vtkPiecewiseFunction()
    for i in values:
        p_f.AddPoint(i[0], i[1])
    return p_f

def ColorTransferFunction(values):
    c_t = vtk.vtkColorTransferFunction()
    for i in values:
        c_t.AddRGBPoint(i[0], i[1], i[2], i[3])
    return c_t

def save_rendered_image(renderer_window, filename):
    images_dir="images/"
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderer_window)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(images_dir+filename)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"✅ Saved image: {filename}")

def crop_white_borders(input_file, output_file, bg_color=(255, 255, 255)):
    img = Image.open(input_file).convert("RGB")
    bg = Image.new("RGB", img.size, bg_color)
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()

    if bbox:
        cropped = img.crop(bbox)
        cropped.save(output_file)
        print(f"Cropped image saved as: {output_file}")
    else:
        print("No non-white content found.")

# === Data Loading ===
INPUT_FILE_NAME="GT_teardrop_128x128x128.vti"
input_path="data/"
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_path+INPUT_FILE_NAME)
reader.Update()
data = reader.GetOutput()

# === Shading Option ===
phong = "no"

# === Transfer Functions ===
# c_t = ColorTransferFunction([
#     [-4391.54, 0, 1, 1],
#     [-2508.95, 0, 0, 1],
#     [-1873.9,  0, 0, 0.5],
#     [-1027.16, 1, 0, 0],
#     [-298.031, 1, 0.4, 0],
#     [2594.97,  1, 1, 0]
# ])
c_t = ColorTransferFunction([
    [0.0,    0.0, 0.0, 1.0],    # Blue
    [20.0,   0.4, 0.7, 1.0],    # Light Blue
    [60.0,   1.0, 1.0, 1.0],    # White
    [100.0,  1.0, 0.5, 0.0],    # Orange
    [140.0,  1.0, 0.0, 0.0],    # Red
    [162.0,  0.5, 0.0, 0.0]     # Dark Red
])


# Opacity_function = OpacityTransferFunction([
#     [-4931.54, 1.0],
#     [101.815,  0.002],
#     [2594.97,  0.0]
# ])
Opacity_function = OpacityTransferFunction([
    [0.0,    0.0],
    [20.0,   0.05],
    [60.0,   0.3],
    [100.0,  0.7],
    [140.0,  1.0],
    [162.0,  0.5]
])

# === Volume Property ===
volume_property = vtk.vtkVolumeProperty()
if phong.lower() == "yes":
    volume_property.ShadeOn()
    volume_property.SetDiffuse(0.5)
    volume_property.SetAmbient(0.5)
    volume_property.SetSpecular(0.5)
    volume_property.SetSpecularPower(10)
volume_property.SetColor(c_t)
volume_property.SetScalarOpacity(Opacity_function)
volume_property.SetInterpolationTypeToLinear()

# === Mapper and Volume Actor ===
v_Mapper = vtk.vtkSmartVolumeMapper()
v_Mapper.SetInputData(data)

volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(v_Mapper)
volume_actor.SetProperty(volume_property)

# === Renderer and Render Window ===
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume_actor)
renderer.SetBackground(1, 1, 1)

renderer_window = vtk.vtkRenderWindow()
renderer_window.SetOffScreenRendering(1)  # ⬅️ Off-screen rendering
renderer_window.SetSize(1024,1024)
renderer_window.AddRenderer(renderer)

# === FRONT VIEW ===
renderer.ResetCamera()
renderer_window.Render()
save_rendered_image(renderer_window, f"{INPUT_FILE_NAME}_front.png")
#crop_white_borders("images_vr/front.png", "front_cropped.png")

# === BACK VIEW ===
camera = renderer.GetActiveCamera()
camera.Azimuth(180)
renderer.ResetCameraClippingRange()
renderer_window.Render()
save_rendered_image(renderer_window, f"{INPUT_FILE_NAME}_back.png")
#crop_white_borders("images_vr/back.png", "back_cropped.png")
