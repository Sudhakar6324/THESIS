from PIL import Image
import numpy as np

image = Image.open("images_vr/back.png")
pixels = np.array(image)
print(image)
print("Image shape:", pixels.shape)
print("Center pixel:", pixels[pixels.shape[0] // 2, pixels.shape[1] // 2])
image = Image.open("back_cropped.png")
pixels = np.array(image)
print(image)
print("Image shape:", pixels.shape)
print("Center pixel:", pixels[pixels.shape[0] // 2, pixels.shape[1] // 2])