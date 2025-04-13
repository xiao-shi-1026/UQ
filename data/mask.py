import torch
import numpy as np
from scipy.sparse import coo_matrix
from pathlib import Path

def generate_mask(batch_size: int, image_shape: tuple, drop_ratio: float = 0.15, RGB: bool = True) -> torch.Tensor:
    """
    Generate a batch of binary masks for images with a given shape.

    Args:
        batch_size: int, number of masks to generate
        image_shape: tuple, shape of the image (H, W)
        drop_ratio: float, percentage of pixels to be masked
        RGB: bool, if True, output shape is (B, 3, H, W), otherwise (B, 1, H, W)

    Returns:
        mask_batch: torch.Tensor, binary mask (1 = masked, 0 = keep)
    """
    H, W = image_shape
    total_pixels = H * W
    num_masked = int(drop_ratio * total_pixels)

    num_channels = 3 if RGB else 1
    mask_batch = torch.zeros((batch_size, num_channels, H, W), dtype=torch.uint8)

    for b in range(batch_size):
        flat_mask = torch.zeros(total_pixels, dtype=torch.uint8)
        mask_indices = torch.randperm(total_pixels)[:num_masked]
        flat_mask[mask_indices] = 1
        base_mask = flat_mask.view(1, H, W)  # shape: [1, H, W]

        # repeat across channels
        mask_batch[b] = base_mask.expand(num_channels, H, W)

    return mask_batch

def mask_to_matrix(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a single binary mask into a dense observation matrix H.

    Args:
        mask: torch.Tensor, shape [1, H, W] or [H, W], binary mask (1 = masked, 0 = observed)

    Returns:
        H_dense: torch.Tensor of shape [3K, 3HW], dtype=float32
    """
    if mask.dim() == 3:
        mask = mask[0]  # convert from [1, H, W] → [H, W]

    H, W = mask.shape
    HW = H * W

    H_idx, W_idx = torch.nonzero(mask == 0, as_tuple=True)  # observed pixels only
    flat_indices = H_idx * W + W_idx
    K = flat_indices.numel()

    row_idx = []
    col_idx = []

    for i, idx in enumerate(flat_indices):
        for c in range(3):
            row_idx.append(3 * i + c)
            col_idx.append(3 * idx + c)

    H = torch.zeros((3 * K, 3 * HW), dtype=torch.float32)
    H[row_idx, col_idx] = 1.0

    return H


def apply_mask(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Apply a binary mask to a single image. Masked regions will be white (1.0).

    Args:
        image: torch.Tensor, shape (3, H, W), input image in [0,1]
        mask: torch.Tensor, shape (1, H, W) or (3, H, W), binary mask (1 = masked)

    Returns:
        masked_img: torch.Tensor, shape (3, H, W)
    """
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.expand(3, -1, -1)  # [3, H, W]

    # Apply mask: 1 → white, 0 → original pixel
    masked_img = image * (1.0 - mask) + mask

    return masked_img


def load_freeform_masks(path: str, op_type: str, number: int, RGB: bool = True) -> torch.Tensor:
    """
    Load freeform masks from .npz and optionally expand to RGB channels.

    Args:
        path: str, directory path containing the npz file
        op_type: str, mask type (e.g., "freeform1020")
        number: int, number of masks to load
        RGB: bool, if True, return shape [N, 3, H, W]; else [N, 1, H, W]

    Returns:
        masks: torch.Tensor of shape [N, C, H, W] with 1 = masked, 0 = keep
    """
    path = Path(path)
    mask_fn = path / f"imagenet_{op_type}_masks.npz"

    # Load and slice masks: [10000, 256, 256] → [N, 1, 256, 256]
    masks_np = np.load(mask_fn)["mask"][:number, None]
    masks = torch.tensor(masks_np, dtype=torch.float32)  # [N, 1, H, W]

    if RGB:
        masks = masks.expand(-1, 3, -1, -1)  # → [N, 3, H, W]

    return masks
