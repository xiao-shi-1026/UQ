"""
dataset.py

This script load the dataset, generate masks, and load masks.
"""
import torch
import numpy as np
def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, height, width)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, 1).
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask

def get_center_mask(image_size):
    h, w = image_size
    mask = bbox2mask(image_size, (h//4, w//4, h//2, w//2))
    return torch.from_numpy(mask).permute(2,0,1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 示例图像尺寸（假设是 RGB 图像）
    img_shape = (256, 256, 3)

    # 定义一个边界框：(top, left, height, width)
    bbox = (100, 50, 60, 80)  # 表示从 (100, 50) 开始，高 60 宽 80 的区域

    # 生成 mask
    mask = bbox2mask(img_shape, bbox)

    # 可视化 mask
    plt.imshow(mask.squeeze(), cmap='gray')  # squeeze 去掉最后一个维度
    plt.title("Mask from bbox")
    plt.show()