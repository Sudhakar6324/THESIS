import vtk
import numpy as np

# ---------------------------
filename = "papers\Isabel_3D.vti"   # Replace with your actual file path
n_isovalues = 20
eps = 1e-8
# ---------------------------

# 1. Read VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(filename)
reader.Update()
image = reader.GetOutput()

# 2. Auto-detect scalar field name
point_data = image.GetPointData()
n_arrays = point_data.GetNumberOfArrays()

if n_arrays == 0:
    raise RuntimeError("No scalar fields found in the .vti file!")

scalar_field = point_data.GetArrayName(0)
print(f"Using scalar field: {scalar_field}")

# 3. Compute gradient
grad_filter = vtk.vtkGradientFilter()
grad_filter.SetInputData(image)
grad_filter.SetInputScalars(vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_field)
grad_filter.SetResultArrayName("Gradient")
grad_filter.Update()
grad_image = grad_filter.GetOutput()

# 4. Compute gradient magnitude
grad_array = grad_image.GetPointData().GetArray("Gradient")
n_points = grad_image.GetNumberOfPoints()

grad_mag_array = vtk.vtkDoubleArray()
grad_mag_array.SetName("GradientMagnitude")
grad_mag_array.SetNumberOfTuples(n_points)

for i in range(n_points):
    g = grad_array.GetTuple3(i)
    mag = np.sqrt(g[0]**2 + g[1]**2 + g[2]**2)
    grad_mag_array.SetValue(i, mag)

grad_image.GetPointData().AddArray(grad_mag_array)

# 5. Compute condition number: κ = 1 / ||∇y||
cond_array = vtk.vtkDoubleArray()
cond_array.SetName("ConditionNumber")
cond_array.SetNumberOfTuples(n_points)

for i in range(n_points):
    gm = grad_mag_array.GetValue(i)
    cond = 1.0 / (gm + eps)
    cond_array.SetValue(i, cond)

grad_image.GetPointData().AddArray(cond_array)

# 6. Define isovalues
scalar_data = grad_image.GetPointData().GetArray(scalar_field)
scalar_range = scalar_data.GetRange()
isovalues = np.linspace(scalar_range[0], scalar_range[1], n_isovalues)

# 7. Compute average condition number for each isosurface
results = []
for val in isovalues:
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grad_image)
    contour.SetInputArrayToProcess(0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_field)
    contour.SetValue(0, val)
    contour.Update()

    iso_output = contour.GetOutput()
    if iso_output.GetNumberOfPoints() == 0:
        continue

    cond_on_iso = iso_output.GetPointData().GetArray("ConditionNumber")
    if cond_on_iso is None:
        continue

    cond_values = [cond_on_iso.GetValue(i) for i in range(cond_on_iso.GetNumberOfTuples())]
    avg_cond = np.mean(cond_values)
    results.append((val, avg_cond))

# 8. Sort and print results
results_sorted = sorted(results, key=lambda x: x[1])
print("\n--- Isovalue Condition Summary ---")
for val, cond in results_sorted:
    print(f"Isovalue: {val:.2f}, Avg Condition: {cond:.4f}")

best_val, best_cond = results_sorted[0]
worst_val, worst_cond = results_sorted[-1]

print(f"\nBest Isovalue:  {best_val:.2f}, Avg Condition: {best_cond:.4f}")
print(f"Worst Isovalue: {worst_val:.2f}, Avg Condition: {worst_cond:.4f}")

# 9. Extract isosurfaces
def extract_isosurface(val):
    contour = vtk.vtkContourFilter()
    contour.SetInputData(grad_image)
    contour.SetInputArrayToProcess(0, 0, 0,
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, scalar_field)
    contour.SetValue(0, val)
    contour.Update()
    return contour.GetOutput()

best_iso = extract_isosurface(best_val)
worst_iso = extract_isosurface(worst_val)

# 10. Create actor from isosurface
def create_actor(iso):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(iso)
    mapper.SetScalarRange(iso.GetPointData().GetArray("ConditionNumber").GetRange())
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("ConditionNumber")

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor, mapper

best_actor, best_mapper = create_actor(best_iso)
worst_actor, worst_mapper = create_actor(worst_iso)

# 11. Set up renderers
renderer1 = vtk.vtkRenderer()
renderer2 = vtk.vtkRenderer()

renderer1.SetViewport(0.0, 0.0, 0.5, 1.0)
renderer2.SetViewport(0.5, 0.0, 1.0, 1.0)

renderer1.AddActor(best_actor)
renderer2.AddActor(worst_actor)

renderer1.SetBackground(0.1, 0.1, 0.2)
renderer2.SetBackground(0.1, 0.1, 0.2)

# Scalar bars
scalar_bar1 = vtk.vtkScalarBarActor()
scalar_bar1.SetLookupTable(best_mapper.GetLookupTable())
scalar_bar1.SetTitle("Condition Number")
scalar_bar1.SetNumberOfLabels(4)
renderer1.AddActor2D(scalar_bar1)

scalar_bar2 = vtk.vtkScalarBarActor()
scalar_bar2.SetLookupTable(worst_mapper.GetLookupTable())
scalar_bar2.SetTitle("Condition Number")
scalar_bar2.SetNumberOfLabels(4)
renderer2.AddActor2D(scalar_bar2)

# 12. Render window and interactor
render_window = vtk.vtkRenderWindow()
render_window.SetSize(1200, 600)
render_window.AddRenderer(renderer1)
# render_window.AddRenderer(renderer2)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Sync camera
camera = vtk.vtkCamera()
renderer1.SetActiveCamera(camera)
# renderer2.SetActiveCamera(camera)

render_window.Render()
interactor.Start()
