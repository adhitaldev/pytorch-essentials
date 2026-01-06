"""
Example of the 'superpoint' keypoint detector as described here:
https://huggingface.co/stevenbucaille/superpoint
Original Paper: SuperPoint: Self-Suprvised Interest Point Detection and Description https://arxiv.org/abs/1712.07629 (DeTone et.al., CVPREW 2018)
The model is able to detect interest points underhomographic transformations and provide a descriptor for each point.
"""
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import os
import matplotlib.pyplot as plt


def run_point_detector_example():
    print(' Running SuperPoint keypoint detector example... ')
    img1 = os.getenv("TEST_IMAGE_1", "pytorch-essentials/src/vision/data/test/img1.jpg")
    img2 = os.getenv("TEST_IMAGE_2", "pytorch-essentials/src/vision/data/test/img2.jpg")
    if not os.path.exists(img1) or not os.path.exists(img2):
        print('Test Image 1 or 2 does not exist')
        return

    img1 = Image.open(img1)
    img2 = Image.open(img2)
    images = [img1, img2]

    # Detecting interesting points on an image
    processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
    model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    inputs = processor(images, return_tensors="pt")
    img_sizes = [(image.size[1], image.size[0]) for image in images]

    outputs = model(**inputs) # contains the list of keypoint coordinates with their respective score and description (256 long vector)
    outputs = processor.post_process_keypoint_detection(outputs, img_sizes)
    keypoints = scores = descriptors = None
    for output in outputs:
        keypoints = output["keypoints"].detach().numpy()
        scores = output["scores"].detach().numpy()
        descriptors = output["descriptors"].detach().numpy()

    total_pts = scores.size
    # Show the keypoints
   # print(f"The outputs are \n {outputs}")
    plt.axis('off')
    print(f"KeyPoints Type:  {keypoints} Size: {keypoints.size}")
    print(f"Scores Type:  {scores} Size: {scores.size}")
    plt.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        c=scores * 100, # Marker color
        s=scores * 50, # Marker size
        alpha=0.8
    )

    for idx in range(total_pts):
        plt.text(keypoints[idx, 0], keypoints[idx, 1], scores[idx], ha='center', va='bottom')


    plt.imshow(img1)
    plt.show()


if __name__ == "__main__":
    run_point_detector_example()