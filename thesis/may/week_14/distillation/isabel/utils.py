## Util Functions
##########################################################

import numpy as np
import vtk
import os
from vtkmodules.util import numpy_support
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import lr_scheduler
##########################################################


def load_trained_model(mfile, single_model, device):
    checkpoint = torch.load(mfile, map_location=torch.device(device))
    # Adjust state_dict keys if they were saved using DataParallel
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
      # Remove the 'module.' prefix if it exists
      new_key = key.replace("module.", "") if key.startswith("module.") else key
      new_state_dict[new_key] = value

    single_model.load_state_dict(new_state_dict)
    return single_model

def recon_data(model, np_arr_coord, np_arr_vals, group_size, device):
    model.eval()
    torch_coords = torch.from_numpy(np_arr_coord)
    torch_vals = torch.from_numpy(np_arr_vals)
    predicted_vals = torch.zeros_like(torch_vals).squeeze().to(device)
    with torch.no_grad():
        for i in range(0, torch_coords.shape[0], group_size):
            coords = torch_coords[i:i + group_size]
            coords = coords.type(torch.float32).to(device)
            vals = model(coords)
            predicted_vals[i:i+group_size] = vals.squeeze()

    return predicted_vals.cpu().numpy().squeeze()

def save_volume(data, varname, extracted_vals1, outdata_path, dataset_name):
    # Now scale back to original range and then store
    min_data = data.GetPointData().GetArray(varname).GetRange()[0]
    max_data = data.GetPointData().GetArray(varname).GetRange()[1]
    extracted_vals1 = ((extracted_vals1 + 1) / 2.0) * (max_data - min_data) + min_data
    vtk_arr1 = numpy_support.numpy_to_vtk(extracted_vals1.squeeze())
    vtk_arr1.SetName('recon_' + varname)
    ## create an empty vtkImageData
    outdata = createVtkImageData(data.GetOrigin(), data.GetDimensions(), data.GetSpacing())
    outdata.GetPointData().AddArray(vtk_arr1)
    # Write reconstructed data out
    outfname = os.path.join(outdata_path, 
                            'recon_' + dataset_name + 
                            '_' + varname + '.vti')
    write_vti(outdata, outfname)

def data_setup(data, arrname):

    ## Load data
    num_pts = data.GetNumberOfPoints()
    dims = data.GetDimensions()

    data_arr = data.GetPointData().GetArray(arrname)

    np_arr_coord = np.zeros((num_pts,3))
    np_arr_vals = np.zeros((num_pts,1))

    for i in range(num_pts):
      pt = data.GetPoint(i)
      val1 = data_arr.GetTuple1(i)
      np_arr_vals[i,:] = val1
      np_arr_coord[i,:] = pt

    min_data = np.min(np_arr_vals[:,0])
    max_data = np.max(np_arr_vals[:,0])
    np_arr_vals[:,0] = 2.0*((np_arr_vals[:,0]-min_data)/(max_data-min_data)-0.5)

    ### Normalize between 0 to 1
    np_arr_coord[:,0] = np_arr_coord[:,0]/dims[0]
    np_arr_coord[:,1] = np_arr_coord[:,1]/dims[1]
    np_arr_coord[:,2] = np_arr_coord[:,2]/dims[2]

    return np_arr_coord, np_arr_vals

def coord_setup(data):

    ## Load data
    num_pts = data.GetNumberOfPoints()
    dims = data.GetDimensions()
    np_arr_coord = np.zeros((num_pts,3))
    for i in range(num_pts):
      pt = data.GetPoint(i)
      np_arr_coord[i,:] = pt

    ### Normalize between 0 to 1
    np_arr_coord[:,0] = np_arr_coord[:,0]/dims[0]
    np_arr_coord[:,1] = np_arr_coord[:,1]/dims[1]
    np_arr_coord[:,2] = np_arr_coord[:,2]/dims[2]

    return np_arr_coord

def random_sampling(dims,samp_percentage_to_use,coords, vals):
    total_pts = dims[0]*dims[1]*dims[2]
    num_samples = int(total_pts*samp_percentage_to_use)
    random_indices = torch.randint(0, total_pts, (num_samples,))
    ## now select corresponding points based on random indices
    return coords[random_indices], vals[random_indices]

# Function to read VTI files and extract data
def read_vti_file(file_path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader.GetOutput()

def write_vti(data, fname):
    writer = vtk.vtkXMLImageDataWriter()
    #fname = os.path.join(directory, 'recon_' + dataset_name + '.vti')
    writer.SetInputData(data)
    writer.SetFileName(fname)
    writer.Write()

## compute SNR
def compute_PSNR(arrgt, arr_recon):
    diff = arrgt - arr_recon
    sqd_max_diff = (np.max(arrgt) - np.min(arrgt))**2
    snr = 10 * np.log10(sqd_max_diff / np.mean(diff**2))
    return snr

## compute RMSE
def compute_rmse(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    return np.sqrt(mse)

## return an empty vtkimagedata
def createVtkImageData(origin, dimensions, spacing):
    localDataset = vtk.vtkImageData()
    localDataset.SetOrigin(origin)
    localDataset.SetDimensions(dimensions)
    localDataset.SetSpacing(spacing)
    return localDataset

def compute_gradient(data):
    ## compute gradient mag.
    gradient_filter = vtk.vtkGradientFilter()
    gradient_filter.SetInputData(data)
    gradient_filter.SetResultArrayName('Gradients')  # Name of the output gradient field
    gradient_filter.Update()
    # Compute Gradient Magnitude
    calculator = vtk.vtkArrayCalculator()
    calculator.SetInputConnection(gradient_filter.GetOutputPort())
    calculator.AddVectorArrayName('Gradients')
    calculator.SetResultArrayName('GradientMagnitude')
    calculator.SetFunction('mag(Gradients)')  # Compute the magnitude of the gradient vector
    calculator.Update()
    grad_mag_field = calculator.GetOutput()
    return grad_mag_field.GetPointData().GetArray('GradientMagnitude')




