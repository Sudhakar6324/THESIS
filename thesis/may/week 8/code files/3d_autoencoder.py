# ?? Install required packages

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
# ?? Load and normalize each of the 9 channels independently to [-1, 1]
def load_vti_9channels(path):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()
    arr = dsa.WrapDataObject(data)

    dims = data.GetDimensions()
    fields = [data.GetPointData().GetArrayName(i) for i in range(9)]
    raw_channels = [arr.PointData[name].reshape(dims[::-1]) for name in fields]

    normed_channels = []
    min_vals = []
    max_vals = []

    for ch in raw_channels:
        ch = ch.astype(np.float32)
        min_v = ch.min()
        max_v = ch.max()
        normed = 2 * (ch - min_v) / (max_v - min_v + 1e-8) - 1
        normed_channels.append(normed)
        min_vals.append(min_v)
        max_vals.append(max_v)

    vol = np.stack(normed_channels, axis=-1)
    return vol, np.array(min_vals), np.array(max_vals), fields

# ?? Pad volume to make dimensions divisible by 8
def pad_to_multiple(vol, multiple=8):
    d, h, w, c = vol.shape
    pad_d = (multiple - d % multiple) % multiple
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_dims = ((0, pad_d), (0, pad_h), (0, pad_w), (0, 0))
    vol_padded = np.pad(vol, pad_dims, mode='constant')
    return vol_padded, pad_dims

def unpad_volume(vol, pad_dims):
    if pad_dims[0][1] > 0: vol = vol[:-pad_dims[0][1], :, :, :]
    if pad_dims[1][1] > 0: vol = vol[:, :-pad_dims[1][1], :, :]
    if pad_dims[2][1] > 0: vol = vol[:, :, :-pad_dims[2][1], :]
    return vol

# ?? 3D Autoencoder Model
class Conv3DAutoEncoder(nn.Module):
    def __init__(self, in_channels=9, base_channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(base_channels, in_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ??? Training
def train_autoencoder(model, volume, epochs=200, lr=1e-4, model_path="gmm_3d_autoencoder.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    vol = torch.from_numpy(volume).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    loader = DataLoader(TensorDataset(vol), batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        for xb, in loader:
            out = model(xb.to(device))
            loss = F.mse_loss(out, xb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"? Saved model to: {model_path}")
    return model

# ?? Denormalization helper
def denormalize(pred, min_vals, max_vals):
    return 0.5 * (pred + 1) * (max_vals - min_vals) + min_vals

# ?? Evaluate with PSNR & RMSE per channel
def evaluate_model(model, volume_normed, min_vals, max_vals, pad_dims, field_names):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(volume_normed).permute(3, 0, 1, 2).unsqueeze(0).float().to(next(model.parameters()).device)
        pred = model(x).squeeze(0).permute(1, 2, 3, 0).cpu().numpy()

    pred = unpad_volume(pred, pad_dims)
    volume_normed = unpad_volume(volume_normed, pad_dims)

    psnr_list, rmse_list = [], []
    for i in range(9):
        pred_i = denormalize(pred[..., i], min_vals[i], max_vals[i])
        true_i = denormalize(volume_normed[..., i], min_vals[i], max_vals[i])
        mse = np.mean((pred_i - true_i) ** 2)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10((max_vals[i] - min_vals[i]) / (rmse + 1e-8))
        psnr_list.append(psnr)
        rmse_list.append(rmse)

    print("\n?? PSNR / RMSE per channel:")
    for name, psnr, rmse in zip(field_names, psnr_list, rmse_list):
        print(f"{name:>5s}: PSNR = {psnr:.2f} dB | RMSE = {rmse:.6f}")

# ?? Run the full pipeline
vti_path = "new_gmm_isabel_week_6.vti"  # ?? Update this after uploading file
epochs = 20000
lr = 1e-4
model_path = "gmm_autoencoder.pth"

print("?? Loading volume...")
volume, min_vals, max_vals, field_names = load_vti_9channels(vti_path)
volume, pad_dims = pad_to_multiple(volume)

print(f"Loaded shape: {volume.shape}, fields: {field_names}")

print("?? Initializing model...")
model = Conv3DAutoEncoder()

print("?? Training...")
model = train_autoencoder(model, volume, epochs=epochs, lr=lr, model_path=model_path)

print("?? Evaluating...")
evaluate_model(model, volume, min_vals, max_vals, pad_dims, field_names)
