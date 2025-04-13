"""
dataset.py

This script load the dataset
"""
import torch
import numpy as np
import cv2

def load_image(img_path: str, transform = None) -> torch.Tensor:
    """
    Load an image from a file path and apply optional transformations.
    params:
        img_path: path to the image file
        transform: optional transformation to apply to the image
    returns:
        img_tensor: torch.Tensor, shape (3, H, W), normalized to [0, 1]
    """
    # (H, W, 3) - BGR
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # (3, H, W)
    img = np.float32(img/ 255.0) # normalize [0,1]

    img_tensor = torch.tensor(img, dtype=torch.float32)
    if transform:
        img_tensor = transform(img_tensor)
    return img_tensor

def addnoise(img_train: torch.Tensor, noiseL: float, device: torch.device) -> torch.Tensor:
    """
    Add random level of gaussian noise.
    params:
        img_train: image to add noise
        noiseL: the noise level
        device: the device to use (CPU or GPU)
    returns:
        image after adding noise
    """
    B, C, H, W = img_train.shape
    noise = torch.zeros((B, C, H, W), dtype=torch.float32)

    for i in range(B):
        noise[i] = torch.randn((C, H, W)) * (noiseL / 255.0)  # scale std to [0,1] domain

    noisy_imgs = img_train + noise.to(device)
    noisy_imgs = torch.clamp(noisy_imgs, 0.0, 1.0)  # Keep in valid image range

    return noisy_imgs