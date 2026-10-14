"""
Object detection using the Faster R-CNN model with a ResNet50 backbone
and Feature Pyramid Network (FPN) for feature extraction. 

ResNet-50 is a deep conbolutoinal neural network with 50 layers, well known for its
feature extraction abilities.

FPN or Feature Pyramid Network ehnances model's ability to detect objects at multiple
scales by constructing a pyramid of feature maps with different resolutions, thus aiding
to model's learning at differnet sizes.
"""
import os
import torch
import ast
import torchvision.models as vmodels
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision.io import decode_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from src.utils.model_utils import get_model_classes_from_weights_meta
from src.torch_vision.vision_utils import filter_object_detections, visualize_tensor_image, get_annotated_image_tensor, display_random_dataset_images
from .object_detection_dataset import create_voc_dataset
from dotenv import load_dotenv

# Use this environment file in the torch_vision package for environment variables
env_file = f".env.object_detection"

def load_model():
    print(f"Loading FasterCNN_RESTNET50_FPN object detector")
    weights = vmodels.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = vmodels.detection.fasterrcnn_resnet50_fpn(weights=weights).eval()
    classes = get_model_classes_from_weights_meta(model, weights)
    print(f"Classes: {classes} \nTotal Classes: {len(classes)}")
    return model, classes

def modify_model(model, num_classes: int):
    # Freeze all the params
    for params in model.parameters():
        params.requires_grad = False
    
    print(f"Before head replacement {model.roi_heads}")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print(f"After head replacement {model.roi_heads}")

    # Unfreeze the roi heads for training
    for params in model.roi_heads.parameters():
        params.requires_grad = True
    return model

def prepare_data(
                 train_images_path, 
                 train_annotations_path,
                 validation_images_path, 
                 validation_annotations_path,
                 class_labels: 'dict',
                 batch_size=64,):
  """
  Prepares data based on the batch size, and training and validation paths.
  Returns data loaders for training and validation.
  """
  train_transforms =  T.Compose([
      T.RandomHorizontalFlip(),
      T.RandomRotation(degrees=15),
      T.ColorJitter(brightness=(0.5, 1.4), contrast=(0.8, 1.5), saturation=0.1),
      T.Resize(256),
      T.CenterCrop(224)
  ])

  validate_transforms = T.Compose([
    T.ToTensor()
  ])

  train_dataset = create_voc_dataset(images_path=train_images_path, annotations_path=train_annotations_path, class_to_labels=class_labels, transforms=train_transforms)
  validate_dataset = create_voc_dataset(images_path=validation_images_path, annotations_path=validation_annotations_path, class_to_labels=class_labels, transforms=validate_transforms)

  # Data loader
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)

  return train_dataset, validate_dataset, train_loader, validate_loader


if __name__=="__main__":
    # Load the model and define the classes to detect
    model, classes = load_model()

    # Replace the box predictor for this model
    total_classes = 4
    modify_model(model, total_classes)

    if os.path.exists(env_file):
      load_dotenv(env_file)
    else:
      load_dotenv()

    print("Loading the environment parameters for the fastercnn_restnet50_fpn model training ...")
    train_images_path = os.getenv("OBJ_DETECTION_TRAIN_IMAGES_PATH")
    train_annotations_path = os.getenv("OBJ_DETECTION_TRAIN_ANNOTATIONS_PATH")
    validate_images_path = os.getenv("OBJ_DETECTION_VALIDATE_IMAGES_PATH")
    validate_annotations_path = os.getenv("OBJ_DETECTION_VALIDATE_ANNOTATIONS_PATH")
    class_labels = ast.literal_eval(os.getenv("OBJ_DETECTION_CLASS_LABELS"))
    num_samples= int(os.getenv("OBJ_DETECTION_NUM_SAMPLES", "10"))
    batch_size = int(os.getenv("OBJ_DETECTION_BATCH_SIZE", "64"))
   
    print(f"Train Images Path: {train_images_path}")
    print(f"Train Annotations Path: {train_annotations_path}")
    print(f"Validate Images Path: {validate_images_path}")
    print(f"Validate Annotations Path: {validate_annotations_path}")
    print(f"Class labels: {class_labels}")
    print(f"Batch size: {batch_size}")
    print(f"Total display samples: {num_samples}")

    train_dataset, validate_dataset, train_loader, validate_loader = prepare_data(
                                                                                  train_images_path=train_images_path, 
                                                                                  train_annotations_path=train_annotations_path,
                                                                                  validation_images_path=validate_images_path, 
                                                                                  validation_annotations_path=validate_annotations_path, 
                                                                                  class_labels=class_labels,
                                                                                  batch_size=batch_size
                                                                                  )
    print(f"Total training samples {len(train_dataset)}")
    print(f"Train dataloader created with {batch_size}")
    print(f"Total validation samples {len(validate_dataset)}")
    print(f"Validation dataloader created with {batch_size}")
    # Visualize the images
    id_to_names = {v:k for k, v in class_labels.items()}
    display_random_dataset_images(train_dataset, id_to_names, num_images=num_samples, title=f"renet50fpn: {num_samples} Training Samples")
    display_random_dataset_images(validate_dataset, id_to_names, num_images=num_samples, title=f"renet50fpn: {num_samples} Validation Samples")


    # Test the model with a test image
    # from dotenv import load_dotenv
    # import os
    # load_dotenv()
    # random_images = os.getenv("RANDOM_IMAGES", ".")
    # img_path = os.path.join(random_images, "cars_traffic_light.jpg")
    # img = Image.open(img_path).convert("RGB")
    # #img.show()

    # # Define the transform
    # img_transform = transforms.Compose([transforms.ToTensor()])
    # img_tensor = img_transform(img)
    # print(img_tensor.shape)

    # # Add the batch dimension
    # img_tensor = img_tensor.unsqueeze(0)
    # print(f"Unsqueezing to add batch dimension = {img_tensor.shape}")
    
    # # Infer
    # predictions = None
    # with torch.no_grad():
    #     predictions = model(img_tensor)[0]
    
    # # Filtering predictions by classes to detect and minimum confidence
    # min_confidence = 0.5
    # print(f"Filtering predictions")
    # boxes, labels, scores = filter_object_detections(predictions, classes_to_detect, min_confidence)

    # # Annotate and visualize the detected bounding box class and scores
    # img_tensor = img_tensor.squeeze(0)
    # annotated_img_tensor = get_annotated_image_tensor(img_tensor, boxes, labels, scores, classes_to_detect, bbox_colors)
    # visualize_tensor_image(annotated_img_tensor)

""""
Resnet50_fpn architecture:

FasterRCNN(
  (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
  )
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): FrozenBatchNorm2d(64, eps=0.0)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(256, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(512, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(1024, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(2048, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
    )
    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(
        (0): Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Conv2dNormActivation(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (3): Conv2dNormActivation(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (layer_blocks): ModuleList(
        (0-3): 4 x Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (extra_blocks): LastLevelMaxPool()
    )
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
        )
      )
      (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
    (box_head): TwoMLPHead(
      (fc6): Linear(in_features=12544, out_features=1024, bias=True)
      (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
    )
  )
)

"""



    
 




