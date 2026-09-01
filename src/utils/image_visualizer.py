from PIL import Image, ImageDraw
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_tensor_image(image_tensor=None, label="Unknown", mean=None, std=None):
    """
    Visualizes a given tensor as an image
    """
    if not isinstance(image_tensor, torch.Tensor):
        print("Not a tensor image type")
        return
    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3:
        raise ValueError(f"Expected tensor of shape [3, H, W] but got {image_tensor.shape}")
    
    img = image_tensor.detach().cpu().clone()
    # Reverse normalize(): pixel = normalized * std + mean
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean).view(3, 1, 1)
        std_t = torch.tensor(std).view(3, 1, 1)
        img = img * std_t + mean_t
    
    # Convert (C, H, W) -> (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    # Scale to [0, 255] uint8 for PIL
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0, 1)

    plt.figure()
    plt.imshow(img)
    plt.title(label)
    plt.axis("off")
    plt.show()



