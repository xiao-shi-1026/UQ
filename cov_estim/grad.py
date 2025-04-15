"""
This script calculate the gradient of the image with respect to the mask.
"""
import torch
import data.dataset as dataset
import data.mask as m
import numpy as np
from scipy.sparse import coo_matrix, issparse
from scipy.sparse.linalg import lsqr
from numpy.linalg import lstsq
def forward_direction(model: torch.nn.Module,
                      masked_image: torch.Tensor,
                      H: torch.Tensor,
                      vecs: torch.Tensor, 
                      amount: float = 0.01) -> torch.Tensor:
    """
    Forward u(y + ε * perturbation), where perturbation is masked eigvecs.

    params:
        model: torch.nn.Module, inpainting model
        masked_image: [3, H, W], corrupted image
        H: [1, H, W] or [3, H, W], binary mask (1 = masked)
        eigvecs: [n_ev, 3, H, W] or [3, H, W], directional perturbation
        amount: float, perturbation scale

    returns:
        output: [n_ev, 3, H, W]
    """
    if vecs.dim() == 3:
        vecs = vecs.unsqueeze(0)  # → [1, 3, H, W]

    n_ev = vecs.shape[0]

    perturbed_inputs = []
    for i in range(n_ev):
        # Use your masking expression
        perturbed = masked_image + amount * m.apply_mask(vecs[i], H)  # [3, H, W]
        perturbed_inputs.append(perturbed)

    input_batch = torch.stack(perturbed_inputs)  # [n_ev, 3, H, W]

    with torch.no_grad():
        output = model(input_batch)  # [n_ev, 3, H, W]

    return output

def random_direction(
    n_v: int,
    shape: torch.Size,
    device: torch.device = 'cpu',
    dtype=torch.float32) -> torch.Tensor:
    """
    Initialize n_v orthonormal random direction vectors with the given shape.

    params:
        n_v (int): Number of orthonormal directions to generate.
        shape (torch.Size): Shape of each direction, e.g., (3, 256, 256).
        device (torch.device): Device to place the directions on.
        dtype (torch.dtype): Data type of the directions.

    returns:
        torch.Tensor: Tensor of shape [n_v, *shape], containing orthonormal direction vectors.
    """
    D = torch.tensor(shape).prod().item()  # total number of elements
    mat = torch.randn(D, n_v, device=device, dtype=dtype)  # [D, n_v]

    # QR decomposition to orthonormalize
    Q, _ = torch.linalg.qr(mat, mode='reduced')  # Q: [D, n_v]

    # Reshape each column back to shape
    directions = Q.T.reshape(n_v, *shape)  # [n_v, 3, H, W]

    # Normalize each direction to unit norm
    directions = directions / directions.view(n_v, -1).norm(dim=1).view(n_v, 1, 1, 1)

    return directions

def directional_cov(
    model: torch.nn.Module,
    masked_image: torch.Tensor,
    mmse: torch.Tensor,
    H: torch.Tensor,
    directions: torch.Tensor, 
    amount: float = 1) -> torch.Tensor:
    """
    Estimate covariance projections along multiple directions using forward_direction batch evaluation.

    params:
        model (torch.nn.Module): Inpainting/denoising model.
        masked_image (torch.Tensor): Masked input image (i.e. y), shape [3, H, W].
        mmse (torch.Tensor): Model output at original masked image, shape [3, H, W].
        H (torch.Tensor): Binary mask, shape [1, H, W] or [3, H, W], where 1 = masked.
        directions (torch.Tensor): Direction vectors [n_v, 3, H, W], assumed normalized.
        forward_direction_fn: A callable like `forward_direction(model, y, H, v, eps)` → model output
        amount (float): Perturbation amount ε.

    returns:
        torch.Tensor: Projected variances λ_j for each direction, shape [n_v]
    """
    # Get perturbed predictions
    perturbed_outputs = forward_direction(model, masked_image, H, directions, amount)  # [n_v, 3, H, W]

    # Compute squared distances per direction
    deltas = perturbed_outputs - mmse.unsqueeze(0)  # [n_v, 3, H, W]
    lambda_projections = deltas.pow(2).flatten(1).sum(dim=1) / (amount ** 2)  # [n_v]

    return lambda_projections.detach()
