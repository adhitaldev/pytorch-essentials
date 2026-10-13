"""
Helpful utility functions for vision tasks.
"""
import math
import torch
import torchvision.utils as vutils
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def get_mean_std(dataset):
    """
    Calculates the mean and standard deviation of a PyTorch dataset.
    Args:
        dataset (torch.utils.data.Dataset): The dataset for which to
                                            calculate the stats. It should
                                            return image tensors.
    Returns:
        (torch.Tensor, torch.Tensor): A tuple containing the mean and
                                      standard deviation tensors, each of
                                      shape (C,).
    """
    # Create a DataLoader to iterate through the dataset in batches for efficiency.
    # shuffle=False because the order of images doesn't matter for this calculation.
    loader = data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize tensors to store the sum of pixel values for each (RGB) channel.
    channel_sum = torch.zeros(3)
    # Initialize tensors to store the sum of squared pixel values for each channel.
    channel_sum_sq = torch.zeros(3)
    # Initialize a counter for the total number of pixels.
    num_pixels = 0

    # Go through the image batches in loader
    for images in loader:
        # Add the total number of pixels in this batch to the running total.
        num_pixels += images.size(0) * images.size(2) * images.size(3)
        
        # Sum the pixel values across the batch, height, and width dimensions,
        # leaving only the channel dimension. Add this to the running total.
        channel_sum += images.sum(dim=[0, 2, 3])
        
        # Square each pixel value, then sum them up similarly to the step above.
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    # Calculate the mean for each channel.
    mean = channel_sum / num_pixels
    # Calculate the standard deviation using the formula: sqrt(E[X^2] - E[X]^2)
    std = (channel_sum_sq / num_pixels - mean ** 2) ** 0.5

    # Return the calculated mean and standard deviation.
    return mean, std


def filter_object_detections(predictions, classes_to_detect,  min_confidence):
    """
    Filters out the object detection predictions based on what classes to
    detect and what is the minimum confidence to include.

    Args:
    predictions = predictions from the model
    classes_to_detect = {index: classname ...}
    """
    boxes = []
    labels = []
    scores = []

    for idx, klass in classes_to_detect.items():
        class_mask = (predictions['labels'] == idx) & (predictions['scores'] >= min_confidence)

        boxes_for_class = predictions['boxes'][class_mask]
        scores_for_class = predictions['scores'][class_mask]
        labels_for_class = predictions['labels'][class_mask]

        # Add to the result
        boxes.extend(boxes_for_class.tolist())
        labels.extend(labels_for_class.tolist())
        scores.extend(scores_for_class.tolist())
        
    return boxes, labels, scores


def get_annotated_image_tensor(img_tensor, boxes, labels, scores, classes_to_detect, bbox_colors):
    """
    Annotates the given image tensor with bounding boxes, class labels and scores.
    Converts scores to percentage and contatenates labels with scores for readability.
    """

    scores = [round(score * 100., 2) for score in scores]
    print(f"Scores : {scores}")

    # Concatenate labels and scores
    labels_scores = [f"{str(classes_to_detect[val])} {scores[idx]}%" for idx, val in enumerate(labels)]

    # Generate the box colors
    box_colors = [bbox_colors[classes_to_detect[label]] for label in labels]

    #print(f"Box colors: {box_colors}")

    # Draw the bounding boxes
    annotated_img_tensor = vutils.draw_bounding_boxes(
        img_tensor,
        torch.tensor(boxes),
        labels=labels_scores,
        colors=box_colors,
        label_colors=box_colors,
        width=2
    )
    return annotated_img_tensor


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


def visualize_multiple_tensor_images(images, cols=None, cell_size=None, padding=4, bg=(255, 255, 255)):
    """
    Takes in a list of image tensors and draws them in a grid evenly.
    """
    """
    Arrange a list of image tensors into an evenly spaced grid.

    images:    list of tensors shaped (C, H, W) or (H, W). Either uint8 in
               [0, 255] or float in [0, 1]. Sizes may differ.
    cols:      number of columns (default: roughly square grid)
    cell_size: (width, height) of every cell (default: size of the largest image)
    padding:   pixels between cells and around the border
    bg:        background color
    Returns a PIL.Image.
    """
    if len(images) == 0:
        raise ValueError("images is empty")

    n = len(images)
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Tensor -> PIL (handles uint8 and float [0,1]); force RGB so mixed
    # grayscale/color inputs paste cleanly
    pil_imgs = [to_pil_image(img.detach().cpu()).convert("RGB") for img in images]

    if cell_size is None:
        cell_w = max(im.width for im in pil_imgs)
        cell_h = max(im.height for im in pil_imgs)
    else:
        cell_w, cell_h = cell_size

    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), bg)

    for i, im in enumerate(pil_imgs):
        r, c = divmod(i, cols)
        # Scale to fit the cell while keeping aspect ratio
        im = ImageOps.contain(im, (cell_w, cell_h))
        # Center inside the cell
        x = padding + c * (cell_w + padding) + (cell_w - im.width) // 2
        y = padding + r * (cell_h + padding) + (cell_h - im.height) // 2
        grid.paste(im, (x, y))

    grid.show()





