

import vtk
from vtk.util import numpy_support
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

# Load the VTI file
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName("Isabel_3D.vti")
reader.Update()
image_data = reader.GetOutput()

# Get dimensions and spacing
dims = image_data.GetDimensions()
spacing = image_data.GetSpacing()
origin = image_data.GetOrigin()

# Get the scalar data
scalars = image_data.GetPointData().GetScalars()
scalar_array = numpy_support.vtk_to_numpy(scalars)

# Generate (x, y, z) coordinates based on image geometry
x_coords = [origin[0] + i * spacing[0] for i in range(dims[0])]
y_coords = [origin[1] + j * spacing[1] for j in range(dims[1])]
z_coords = [origin[2] + k * spacing[2] for k in range(dims[2])]

# Create a full grid of coordinates
data = []
index = 0
for z in z_coords:
    for y in y_coords:
        for x in x_coords:
            value = scalar_array[index]
            data.append([x, y, z, value])
            index += 1

# Create a DataFrame
df = pd.DataFrame(data, columns=["x", "y", "z", "value"])
# Min-max normalization for inputs (x, y, z) to [0, 1]
X = df[["x", "y", "z"]].values
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# Min-max normalization for output (value) to [-1, 1]
y = df["value"].values.reshape(-1, 1)
y_min = y.min()
y_max = y.max()
y_norm = 2 * (y - y_min) / (y_max - y_min) - 1

# Convert to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_norm, dtype=torch.float32).to(device)

# Dataset and Dataloader
batch_size = 1024
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.model(x)

# Train the model
model = NeuralNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 500
for epoch in tqdm(range(epochs), desc="Training"):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    avg_loss = epoch_loss / len(dataset)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}",flush=True)

#Prediction
model.eval()
with torch.no_grad():
    preds = model(X_tensor).cpu().numpy()

# Denormalize predictions back to original scale
preds_denorm = 0.5 * (preds + 1) * (y_max - y_min) + y_min
# Prepare VTK array
vtk_array = numpy_support.numpy_to_vtk(preds_denorm.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
vtk_array.SetName("PredictedScalar")
# Create new vtkImageData object
output_image = vtk.vtkImageData()
output_image.SetDimensions(dims)
output_image.SetSpacing(spacing)
output_image.SetOrigin(origin)
output_image.GetPointData().SetScalars(vtk_array)
# Write to file
writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName("new_Isabel_3D_Predicted.vti")
writer.SetInputData(output_image)
writer.Write()

print(" Predicted VTI file saved as 'Isabel_3D_Predicted.vti'")
