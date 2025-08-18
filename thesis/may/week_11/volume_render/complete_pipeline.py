import vtk
import numpy as np
import os
from PIL import Image, ImageChops
# === Load the original VTI file ===
input_file="new\isabel_test\data\predicted_gaussian_isabel.vti"
imgae_name="predicted"
num_samples=100
# === Output directory ===
output_dir = "samples_vti"
os.makedirs(output_dir, exist_ok=True)
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
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderer_window)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"Saved image: {filename}")

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
# === Load the original VTI file ===
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)  
reader.Update()

image_data = reader.GetOutput()
dims = image_data.GetDimensions()
n_points = image_data.GetNumberOfPoints()

# === Extract 'mean' and 'std' arrays from point data ===
point_data = image_data.GetPointData()
mean_array = point_data.GetArray("Mean") 
std_array = point_data.GetArray("Std") 

if mean_array is None or std_array is None:
    raise ValueError("Missing 'mean' or 'std' arrays in the input VTI file.")

# === Convert VTK arrays to NumPy ===
mean_np = np.array([mean_array.GetTuple1(i) for i in range(n_points)])
std_np  = np.array([std_array.GetTuple1(i) for i in range(n_points)])

# === Output directory ===
output_dir = "samples_vti"
os.makedirs(output_dir, exist_ok=True)
# === Generate 100 samples ===
for i in range(1, num_samples+1):
    sampled_np = np.random.normal(loc=mean_np, scale=std_np)

    # Create a new VTK float array to hold the sample
    sampled_array = vtk.vtkFloatArray()
    sampled_array.SetName("sample")
    sampled_array.SetNumberOfTuples(n_points)
    for j in range(n_points):
        sampled_array.SetValue(j, sampled_np[j])

    # Create a new vtkImageData to hold the sampled volume
    sample_data = vtk.vtkImageData()
    sample_data.DeepCopy(image_data)  # Copy geometry (extent, spacing, origin)
    sample_data.GetPointData().SetScalars(sampled_array)

    # Write to VTI
    # === Shading Option ===
    data=sample_data
    INPUT_FILE_NAME=f"sample_{i:03d}"
    phong = "no"

    # === Transfer Functions ===
    c_t = ColorTransferFunction([
        [-4391.54, 0, 1, 1],
        [-2508.95, 0, 0, 1],
        [-1873.9,  0, 0, 0.5],
        [-1027.16, 1, 0, 0],
        [-298.031, 1, 0.4, 0],
        [2594.97,  1, 1, 0]
    ])
    # c_t = ColorTransferFunction([
    #     [0.0,    0.0, 0.0, 1.0],    # Blue
    #     [20.0,   0.4, 0.7, 1.0],    # Light Blue
    #     [60.0,   1.0, 1.0, 1.0],    # White
    #     [100.0,  1.0, 0.5, 0.0],    # Orange
    #     [140.0,  1.0, 0.0, 0.0],    # Red
    #     [162.0,  0.5, 0.0, 0.0]     # Dark Red
    # ])


    Opacity_function = OpacityTransferFunction([
        [-4931.54, 1.0],
        [101.815,  0.002],
        [2594.97,  0.0]
    ])
    # Opacity_function = OpacityTransferFunction([
    #     [0.0,    0.0],
    #     [20.0,   0.05],
    #     [60.0,   0.3],
    #     [100.0,  0.7],
    #     [140.0,  1.0],
    #     [162.0,  0.5]
    # ])

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
    # === Scalar Bar (Color Legend) ===
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(c_t)
    scalar_bar.SetTitle("sample")
    scalar_bar.SetNumberOfLabels(6)
    scalar_bar.SetOrientationToVertical()
    scalar_bar.SetPosition(0.85, 0.1)
    scalar_bar.SetWidth(0.1)
    scalar_bar.SetHeight(0.8)

    scalar_bar.GetLabelTextProperty().SetFontSize(18)
    scalar_bar.GetTitleTextProperty().SetFontSize(20)
    scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)   # Black labels
    scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)   # Black title

    renderer.AddActor2D(scalar_bar)
    renderer.SetBackground(1, 1, 1)

    renderer_window = vtk.vtkRenderWindow()
    renderer_window.SetOffScreenRendering(1)  # ⬅️ Off-screen rendering
    renderer_window.SetSize(1024,1024)
    renderer_window.AddRenderer(renderer)

    # === FRONT VIEW ===
    renderer.ResetCamera()
    renderer_window.Render()
    images_dir="images"
    os.makedirs(images_dir,exist_ok=True)
    front=os.path.join(images_dir,f"{imgae_name}_front")
    os.makedirs(front,exist_ok=True)
    file_path=os.path.join(front,f"{imgae_name}_{INPUT_FILE_NAME}_front.png")
    save_rendered_image(renderer_window,file_path)
    #crop_white_borders("images_vr/front.png", "front_cropped.png")

    # === BACK VIEW ===
    back=os.path.join(images_dir,f"{imgae_name}_back")
    os.makedirs(back,exist_ok=True)
    file_path=os.path.join(back,f"{imgae_name}_{INPUT_FILE_NAME}_back.png")
    camera = renderer.GetActiveCamera()
    camera.Azimuth(180)
    renderer.ResetCameraClippingRange()
    renderer_window.Render()
    save_rendered_image(renderer_window,file_path)
import os
import numpy as np
from PIL import Image

def compute_mean_image(folder_path, output_path):
    image_files = sorted([
        f for f in os.listdir(folder_path) if f.endswith(".png")
    ])
    if not image_files:
        print(f"No PNG images found in {folder_path}")
        return

    # Load first image to get dimensions
    first_img = Image.open(os.path.join(folder_path, image_files[0])).convert("RGB")
    img_array = np.zeros_like(np.array(first_img), dtype=np.float64)

    num_images = len(image_files)
    print(f"Found {num_images} images in {folder_path}...")

    for filename in image_files:
        img = Image.open(os.path.join(folder_path, filename)).convert("RGB")
        img_array += np.array(img, dtype=np.float64)

    # Compute mean
    mean_array = (img_array / num_images).astype(np.uint8)

    # Save final mean image
    mean_image = Image.fromarray(mean_array)
    mean_image.save(output_path)
    print(f"Saved mean image → {output_path}")

# === Usage ===
compute_mean_image(f"images/{imgae_name}_front", f"images/mean_{imgae_name}_front.png")
compute_mean_image(f"images/{imgae_name}_back", f"images/mean_{imgae_name}_back.png")
