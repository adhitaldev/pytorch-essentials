"""
 Lighglue is a DNN for matching features across images. It builds upon the SuperPoint
 model for detecting interest points and computing their descriptors, and   
"""
from transformers import AutoImageProcessor, AutoModel
import torch
from PIL import Image
import os

def run_lightglue_example():
    print(" --- Running Lighglue example --- ")
    img1 = os.getenv("TEST_IMAGE_1", "/Users/adhital/source/pytorch-essentials/src/vision/data/test/img1.jpg")
    img2 = os.getenv("TEST_IMAGE_2", "/Users/adhital/source/pytorch-essentials/src/vision/data/test/img2.jpg")
    if not os.path.exists(img1) or not os.path.exists(img2):
        print('Test Image 1 or 2 does not exist')
        return 
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    images = [img1, img2]
    processor = AutoImageProcessor.from_pretrained("ETH-CVG/lightglue_superpoint")
    model = AutoModel.from_pretrained("ETH-CVG/lightglue_superpoint")
    inputs = processor(images, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    image_sizes = [[(image.height, image.width) for image in images]]
    outputs = processor.post_process_keypoint_matching(outputs, image_sizes, threshold=0.2)
    for i, output in enumerate(outputs):
        print("For the image pair", i)
        for keypoint0, keypoint1, matching_score in zip(output["keypoints0"], output["keypoints1"], output["matching_scores"]):
            print(f"Keypoint at coordinate {keypoint0.numpy()} in the first image matches with keypoint at coordinate {keypoint1.numpy()} in the second image with a score of {matching_score}.")

    processor.plot_keypoint_matching(images, outputs)



if __name__ == "__main__":
    run_lightglue_example()