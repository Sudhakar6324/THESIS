import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import random
random.seed(42)
np.random.seed(42)
no_of_samples=500
scale_factor=2
input_file="isabel_gaussian.vti"
output_file="sklearn_isabel_gmm.vti"
#getting data or loading data
reader=vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.update()
image_data=reader.GetOutput()

point_data=image_data.GetPointData()

origin=image_data.GetOrigin()
spacing=image_data.GetSpacing()
dims=image_data.GetDimensions()
num_points = dims[0] * dims[1] * dims[2]

print(num_points,dims,flush=True)
no_of_arrays=point_data.GetNumberOfArrays()

variables=[]

#getting variable names
for i in range(no_of_arrays):
  array=point_data.GetArrayName(i)
  variables.append(array)
print(variables,flush=True)
#extracting mean and variance
vtk_array=point_data.GetArray(0)
mean=vtk_to_numpy(vtk_array)
vtk_array=point_data.GetArray(1)
std=vtk_to_numpy(vtk_array)
print(std.shape,flush=True)
#create the array 
def create_vtk_array(name):
    arr = vtk.vtkFloatArray()
    arr.SetName(name)
    arr.SetNumberOfComponents(1)
    arr.SetNumberOfTuples(num_points)
    return arr

gmm_mean0_vtk = create_vtk_array("GMM_Mean0")
gmm_mean1_vtk = create_vtk_array("GMM_Mean1")
gmm_mean2_vtk = create_vtk_array("GMM_Mean2")
gmm_std0_vtk  = create_vtk_array("GMM_Std0")
gmm_std1_vtk  = create_vtk_array("GMM_Std1")
gmm_std2_vtk  = create_vtk_array("GMM_Std2")
gmm_w0_vtk    = create_vtk_array("GMM_Weight0")
gmm_w1_vtk    = create_vtk_array("GMM_Weight1")
gmm_w2_vtk    = create_vtk_array("GMM_Weight2")


#fitting gmm
for pt_id in range(num_points):
  mu=mean[pt_id]
  sigma=std[pt_id]
  scaled_sigma=sigma*scale_factor
  samples=np.random.normal(loc=mu,scale=scaled_sigma,size=no_of_samples).reshape(-1,1)#reshaping because it accepts the inbulit em accepts 2d data only
  gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
  gmm.fit(samples)
  weights=gmm.weights_.flatten()
  means=gmm.means_.flatten()
  stds=np.sqrt(gmm.covariances_.flatten())
  order = np.argsort(means)
  weights = weights[order]
  means = means[order]
  stds = stds[order]
  gmm_mean0_vtk.SetValue(pt_id, means[0])
  gmm_mean1_vtk.SetValue(pt_id, means[1])
  gmm_mean2_vtk.SetValue(pt_id, means[2])

  gmm_std0_vtk.SetValue(pt_id, stds[0])
  gmm_std1_vtk.SetValue(pt_id, stds[1])
  gmm_std2_vtk.SetValue(pt_id, stds[2])

  gmm_w0_vtk.SetValue(pt_id, weights[0])
  gmm_w1_vtk.SetValue(pt_id, weights[1])
  gmm_w2_vtk.SetValue(pt_id, weights[2])
  if i % 10000 == 0:
        tqdm.write(f"Progress: {i}")
  
  
  
point_data.AddArray(gmm_mean0_vtk)
point_data.AddArray(gmm_mean1_vtk)
point_data.AddArray(gmm_mean2_vtk)
point_data.AddArray(gmm_std0_vtk)
point_data.AddArray(gmm_std1_vtk)
point_data.AddArray(gmm_std2_vtk)
point_data.AddArray(gmm_w0_vtk)
point_data.AddArray(gmm_w1_vtk)
point_data.AddArray(gmm_w2_vtk)

point_data.RemoveArray(variables[0])
point_data.RemoveArray(variables[1])
print(point_data,flush=True)
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(output_file)
writer.SetInputData(image_data)
writer.Write()

print(f"Done. Output written ",output_file,flush=True)