"""
Object detection using the Faster R-CNN model with a ResNet50 backbone
and Feature Pyramid Network (FPN) for feature extraction. 

ResNet-50 is a deep conbolutoinal neural network with 50 layers, well known for its
feature extraction abilities.

FPN or Feature Pyramid Network ehnances model's ability to detect objects at multiple
scales by constructing a pyramid of feature maps with different resolutions, thus aiding
to model's learning at differnet sizes.
"""
import torch
import torchvision.models as vmodels
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.io import decode_image
from PIL import Image
from utils.model_utils import get_model_classes_from_weights_meta
from torch_vision.vision_utils import filter_object_detections, visualize_tensor_image, get_annotated_image_tensor


def load_model():
    print(f"Loading FasterCNN_RESTNET50_FPN object detector")
    weights = vmodels.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = vmodels.detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
    classes = get_model_classes_from_weights_meta(model, weights)
    print(f"Classes: {classes} \nTotal Classes: {len(classes)}")
    return model, classes


if __name__=="__main__":
    # Load the model and define the classes to detect
    model, classes = load_model()
    target_class_names = ["car", "traffic light"]
    bbox_colors = {"car": 'red', "traffic light": 'blue'}
    classes_to_detect = {classes.index(name): name for name in target_class_names}
    print(f"Classes to detect : {classes_to_detect}")

    # Test the model with a test image
    from dotenv import load_dotenv
    import os
    load_dotenv()
    random_images = os.getenv("RANDOM_IMAGES", ".")
    img_path = os.path.join(random_images, "cars_traffic_light.jpg")
    img = Image.open(img_path).convert("RGB")
    #img.show()

    # Define the transform
    img_transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = img_transform(img)
    print(img_tensor.shape)

    # Add the batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    print(f"Unsqueezing to add batch dimension = {img_tensor.shape}")
    
    # Infer
    predictions = None
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Filtering predictions by classes to detect and minimum confidence
    min_confidence = 0.5
    print(f"Filtering predictions")
    boxes, labels, scores = filter_object_detections(predictions, classes_to_detect, min_confidence)

    # Annotate and visualize the detected bounding box class and scores
    img_tensor = img_tensor.squeeze(0)
    annotated_img_tensor = get_annotated_image_tensor(img_tensor, boxes, labels, scores, classes_to_detect, bbox_colors)
    visualize_tensor_image(annotated_img_tensor)







    
 




