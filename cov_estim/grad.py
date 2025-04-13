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
                      eigvecs: torch.Tensor, 
                      amount: float = 0.01) -> torch.Tensor:
    """
    Forward u(y + ε * perturbation), where perturbation is masked eigvecs.

    Args:
        model: torch.nn.Module, inpainting model
        masked_image: [3, H, W], corrupted image
        H: [1, H, W] or [3, H, W], binary mask (1 = masked)
        eigvecs: [n_ev, 3, H, W] or [3, H, W], directional perturbation
        amount: float, perturbation scale

    Returns:
        output: [n_ev, 3, H, W]
    """
    if eigvecs.dim() == 3:
        eigvecs = eigvecs.unsqueeze(0)  # → [1, 3, H, W]

    n_ev = eigvecs.shape[0]

    perturbed_inputs = []
    for i in range(n_ev):
        # Use your masking expression
        perturbed = masked_image + m.apply_mask(amount * eigvecs[i], H)  # [3, H, W]
        perturbed_inputs.append(perturbed)

    input_batch = torch.stack(perturbed_inputs)  # [n_ev, 3, H, W]

    with torch.no_grad():
        output = model(input_batch)  # [n_ev, 3, H, W]

    return output

    
def power_iteration(model: torch.nn.Module,
                    n_ev: int,
                    image: torch.Tensor,
                    mask: torch.Tensor,
                    mmse: torch.Tensor,
                    eigvecs: torch.Tensor,
                    num_iter: int = 10,
                    tol: float = 1e-6,
                    verbose: bool = False):
    """
    Perform full power iteration to estimate dominant eigenvectors of the Jacobian covariance.

    Args:
        model: torch.nn.Module, the model used for inference
        n_ev: int, number of eigenvectors to estimate
        image: torch.Tensor, shape (3, H, W), masked input image
        mask: torch.Tensor, shape (1, H, W) or (3, H, W), binary mask (1 = masked, 0 = observed)
        mmse: torch.Tensor, model output at original masked image, shape (3, H, W)
        eigvecs: torch.Tensor, shape (n_ev, 3, H, W), initial eigenvector directions
        num_iter: int, number of iterations
        tol: float, stopping tolerance based on cosine similarity
        verbose: bool, whether to print convergence info

    Returns:
        eigvecs: torch.Tensor, estimated eigenvectors
        eigvals: torch.Tensor, estimated eigenvalues
    """
    prev_eigvecs = eigvecs.clone()

    for it in range(num_iter):
        with torch.no_grad():
            increment = forward_direction(model, image, mask, eigvecs, 0)
            Ab = increment - mmse.unsqueeze(0)  # match shape [n_ev, 3, H, W]

            # Normalize each direction
            norm_of_Ab = Ab.view(n_ev, -1).norm(dim=1)
            eigvecs = Ab / norm_of_Ab.view(n_ev, 1, 1, 1)

            # Orthonormalize
            Q, _ = torch.linalg.qr(eigvecs.permute(1,2,3,0).reshape(-1, n_ev), mode='reduced')
            Q = Q / Q.norm(dim=0)
            eigvecs = Q.T.reshape(eigvecs.shape)

        # Convergence check (cosine similarity)
        cos_sim = torch.sum(prev_eigvecs * eigvecs) / (prev_eigvecs.norm() * eigvecs.norm())
        if verbose:
            print(f"Iter {it+1}: cosine similarity = {cos_sim.item():.6f}")

        if abs(1.0 - cos_sim.item()) < tol:
            if verbose:
                print("Converged.")
            break

        prev_eigvecs = eigvecs.clone()

    eigvals = norm_of_Ab**2  # approximate eigenvalues = ||Jv||^2
    return eigvecs, eigvals



def compute_pinv_list(H_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Compute the pseudo-inverse of each H_b^T in a list of dense observation matrices.

    Args:
        H_list: list of torch.Tensor, each of shape [3K_b, 3HW]

    Returns:
        Ht_pinv_list: list of torch.Tensor, each of shape [3HW, 3K_b]
    """
    Ht_pinv_list = [torch.linalg.pinv(H.T) for H in H_list]
    return Ht_pinv_list

def init_eigvecs(n_ev: int, shape: tuple, device: torch.device, scale: float = 1e-3) -> torch.Tensor:
    """
    Initialize n_ev random eigenvectors.

    Args:
        n_ev: number of directions
        shape: image shape (C, H, W)
        device: torch.device
        scale: small factor to keep linear approximation

    Returns:
        eigvecs: torch.Tensor of shape [n_ev, C, H, W]
    """
    eigvecs = torch.randn((n_ev, *shape), device=device)
    eigvecs = eigvecs / eigvecs.flatten(1).norm(dim=1, keepdim=True).view(n_ev, *[1]*len(shape))
    return eigvecs * scale