import cov_estim.grad as grad
import data.dataset as dataset
import torch
from models.UNet import UNet
import models.utils as utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import data.mask as m


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = dataset.load_image("data/image/00302.png")

measurement_number = 100
n_v = 1
shape = tuple(img.shape)
img = img.to(device)  # shape [3, H, W]

mask = m.generate_mask(measurement_number,(shape[1], shape[2]))

mask = mask.to(device)
lambda_accum = torch.zeros(n_v, device=device)

for i in tqdm(range(measurement_number)):
    rd = grad.random_direction(n_v, shape, device)
    mask_i = mask[i]  # shape [3, H, W]
    masked_img = m.apply_mask(img, mask_i)  # [3, H, W]
    # masked_img = dataset.addnoise(masked_img, 5, device)  # [3, H, W]
    model = utils.load_model(UNet, "models/pretrained/random_inpainting_unet.pth")
    model = model.to(device)
    mmse = model(masked_img.unsqueeze(0))  # [1, 3, H, W]
    mmse = mmse.squeeze(0)  # [3, H, W]
    lambda_proj = grad.directional_cov(model, masked_img, mmse, mask_i, rd, amount=1000)
    lambda_accum += lambda_proj

lambda_mean = lambda_accum / measurement_number

def show_img(tensor_img, title=''):
    img = tensor_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

for i in range(rd.shape[0]):
    v_i = rd[i]                             # direction i: [3, H, W]
    scale = lambda_mean[i].sqrt()          # sqrt(λ_i)
    v_scaled = v_i * scale                 # scaled perturbation

    x_plus = (mmse + v_scaled).clamp(0, 1)
    x_minus = (mmse - v_scaled).clamp(0, 1)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    show_img(mmse, title='MMSE')

    plt.subplot(1, 3, 2)
    show_img(x_plus, title=f'+ √λ ⋅ v_{i}')

    plt.subplot(1, 3, 3)
    show_img(x_minus, title=f'- √λ ⋅ v_{i}')

    plt.tight_layout()
    plt.show()

def show_noi(img_tensor, title='Image'):
    # img_tensor: [C, H, W] in [0, 1]
    img_np = img_tensor.detach().cpu().numpy()
    img_np = img_np.transpose(1, 2, 0)  # To HWC

    plt.imshow(img_np)
    plt.title(title)
    plt.axis("off")

v_scaled_enhanced = v_scaled.clamp(0, 1)

gamma = 0.3
v_scaled_enhanced = v_scaled_enhanced ** gamma

plt.figure(figsize=(10, 3))
show_noi(v_scaled_enhanced, title='v_scaled (gamma enhanced)')
plt.show()