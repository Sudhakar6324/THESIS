import cv2
import numpy as np

def compute_channel_ssim(x, y, C1=6.5025, C2=58.5225):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    
    return numerator / denominator

def ssim_rgb(img1, img2):
    # Ensure images are float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Split channels
    ssim_r = compute_channel_ssim(img1[:,:,2], img2[:,:,2])  # R
    ssim_g = compute_channel_ssim(img1[:,:,1], img2[:,:,1])  # G
    ssim_b = compute_channel_ssim(img1[:,:,0], img2[:,:,0])  # B

    return (ssim_r + ssim_g + ssim_b) / 3.0

# ---- Example Usage ----
img1 = cv2.imread("images\mean_original_back.png")
img2 = cv2.imread("images\mean_predicted_back.png")
score = ssim_rgb(img1, img2)
print("SSIM (RGB):", score)
