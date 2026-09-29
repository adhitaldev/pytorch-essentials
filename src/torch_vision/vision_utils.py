"""
Helpful utility functions for vision tasks.
"""
import torch
import torchvision.utils as vutils
from PIL import Image, ImageDraw
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





