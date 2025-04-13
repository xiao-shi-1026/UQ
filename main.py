import cov_estim.grad as grad
import data.dataset as dataset
import torch
from models.UNet import UNet
import models.utils as utils
import cov_estim.grad as gd
import matplotlib.pyplot as plt
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = dataset.load_image("data/image/00302.png")
import data.mask as m
measurement_number = 1
n_ev = 1
img = img.to(device)  # shape [3, H, W]

mask = m.load_freeform_masks("data/masks", "freeform1020", measurement_number)
# H_list = m.mask_to_matrix(mask)
mask = mask.to(device)

for i in tqdm(range(measurement_number)):
    eigvecs = grad.init_eigvecs(n_ev, (3, 256, 256), device)
    mask_i = mask[i]  # shape [1, H, W]
    masked_img = m.apply_mask(img, mask_i)  # [3, H, W]
    model = utils.load_model(UNet, "models/pretrained/inpainting_unet.pth")
    model = model.to(device)
    mmse = model(masked_img.unsqueeze(0))  # [1, 3, H, W]
    mmse = mmse.squeeze(0)  # [3, H, W]
    eigvecs, eigvals = grad.power_iteration(model, n_ev, img,mask_i, mmse,eigvecs, num_iter=10, tol=1e-6, verbose=False) # [3, H, W]
    




