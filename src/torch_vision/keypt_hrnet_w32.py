"""
Modifies the HRNetW32 image classifier while still using the backbone to train with semantic keypoints.
In HRNetW32, every modules downstream of 'incre_modules' make up the complete Image Classification Head
in the official HRNet documentation.
Original paper: "Deep High-Resolution Represetntaion Learning for Human Pose Estimation" (Sun, et. all)
"""
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image
from vision_utils import annotate_key_points_on_image, visualize_tensor_image, get_pil_image_from_torch_tensor, denormalize_image_tensor, gridify_multiple_tensor_images
from keypt_hrnet_dataset import HRnetKeypointDetectionDataset, create_keypt_hrnet_dataset, get_annotation_dict_from_dog_pose_images, get_idx_keypoints_dog_pose_dataset, get_idx_class_dog_pose_dataset
from keypt_utils import create_keypts_output_infrastructure

IN_H, IN_W = 256, 192 # Standard HRnet input dimension
hrnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
hrnet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class HRNetCustomPoseEstimator(torch.nn.Module):
    import torch
import torch.nn as nn

class HRNetCustomPoseEstimator(nn.Module):
  def __init__(self, backbone, num_keypoints=9):
    super().__init__()
    self.backbone = backbone

    # 1x1 Conv mapping 32 channels to num_keypoints
    self.keypoint_head = nn.Conv2d(
        in_channels=32,
        out_channels=num_keypoints,
        kernel_size=1,
        stride=1,
        padding=0,
    )

  def forward(self, x):
    # 1. Stem
    x = self.backbone.conv1(x)
    x = self.backbone.bn1(x)
    x = self.backbone.act1(x)
    x = self.backbone.conv2(x)
    x = self.backbone.bn2(x)
    x = self.backbone.act2(x)

    # 2. Stage 1 (layer1)
    x = self.backbone.layer1(x)

    # 3. Pass through Stage 2, 3, and 4 using timm's internal stage handlers
    # If using timm's HighResolutionNet, stages/transitions run cleanly sequentially:
    xl = [x]
    if hasattr(self.backbone, 'stage1'):
      xl = self.backbone.stage1(xl)

    # Transition 1 -> Stage 2
    x_list = []
    for i in range(2):
      if self.backbone.transition1[i] is not None:
        x_list.append(self.backbone.transition1[i](x))
      else:
        x_list.append(x)
    yl = self.backbone.stage2(x_list)

    # Transition 2 -> Stage 3
    x_list = []
    for i in range(len(self.backbone.transition2)):
      if self.backbone.transition2[i] is not None:
        # Check index mapping for downsampling branch
        if i < len(yl):
          x_list.append(self.backbone.transition2[i](yl[i]))
        else:
          x_list.append(self.backbone.transition2[i](yl[-1]))
      else:
        x_list.append(yl[i])
    yl = self.backbone.stage3(x_list)

    # Transition 3 -> Stage 4
    x_list = []
    for i in range(len(self.backbone.transition3)):
      if self.backbone.transition3[i] is not None:
        if i < len(yl):
          x_list.append(self.backbone.transition3[i](yl[i]))
        else:
          x_list.append(self.backbone.transition3[i](yl[-1]))
      else:
        x_list.append(yl[i])
    yl = self.backbone.stage4(x_list)

    # yl[0] holds Stage 4 Branch 0 features: shape (Batch, 32, 64, 48)
    high_res_features = yl[0]

    # Map to keypoint heatmaps: shape (Batch, 9, 64, 48)
    return self.keypoint_head(high_res_features)
 

def decode_heatmaps(heatmaps, img_size=(256, 192)):
    """ 
    Extracts the (x, y) coordinates from the heatmap tensors
    original_img_size in (H, W)
    """
    batch_size, num_keypoints, h_map, w_map = heatmaps.shape
    target_h, target_w  = img_size

    # Reshape heatmap to 2D matrix per keypoint: (Batch, Num_Keypoints, H_map * W_map)
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)

    # Find the pixel index with maximum probability response
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    # Convert 1D flattened index back into (x, y) grid coordinates
    preds = torch.zeros((batch_size, num_keypoints, 2), device=heatmaps.device)
    preds[:, :, 0] = (idx % w_map).float()  * (target_w / w_map)  # x coordinate on heatmap
    preds[:, :, 1] = (idx // w_map).float()  * (target_h / h_map) # y coordinate on heatmap

    # Squeeze batch dimension for single-image inference
    keypoints = preds[0].cpu().numpy()
    confidence = maxvals[0].cpu().numpy()
    return keypoints, confidence

def get_model_backbone(only_features=True):
    model = timm.create_model("hrnet_w32", pretrained=True, f_only=only_features)
    return model

def modify_model_get_backbone(model):
    """
    Modifies the HRNet classifier to output 'n' landmark points.
    """
    # Change the modules inside the classifier head to identity
    # so feature maps pass through it intact
    model.incre_modules = torch.nn.Identity()
    model.downsamp_modules = torch.nn.Identity()
    model.final_layer = torch.nn.Identity()
    model.global_pool = torch.nn.Identity()
    model.head_drop = torch.nn.Identity()
    model.classifier = torch.nn.Identity()
    return model

def predict_pose(model, image_path, device="cuda"):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    org_w, org_h = img.size

    # 2. Standard HRNet input dimensions (Height: 256, Width: 192)
    # HRNet expects a 4:3 aspect ratio rather than a square (256x256)
    transform = transforms.Compose([
      transforms.Resize((IN_H, IN_W), interpolation=transforms.InterpolationMode.BILINEAR),
      transforms.ToTensor(),  # Automatically scales to [0.0, 1.0] and reshapes to (3, 256, 192)
      transforms.Normalize(
          mean=hrnet_mean, std=hrnet_std
      ),
    ])

    # Apply transformation
    tensor = transform(img)

    # Add batch dimension
    print(f"2Input Tensor shape: {tensor.shape}")
    tensor = tensor.unsqueeze(0).to(device)
    print(f"Input Tensor shape: {tensor.shape}")

    with torch.no_grad():
        heatmaps = model(tensor)
        # Normaize raw logits to [0, 1.0]
        heatmaps = torch.sigmoid(heatmaps)
    
    keypoint_positions, confidence_scores = decode_heatmaps(
      heatmaps, (org_w, org_h)
    )
    return keypoint_positions, confidence_scores

def prepare_key_pts_data(
    train_images_path: 'str',
    train_annotations_dict: 'dict',
    val_images_path: 'str',
    val_annotations_dict: 'dict',
    idx_to_class: 'dict',
    idx_to_keypts: 'dict',
    batch_size=64,
    train_transforms=None,
    val_transforms=None):
    """
    Prepares data for key point detection based on the given parameters.
    Returns data loaders for training and validation.
    """
    print("\n--------------- PREPARING HRNET KEYPOINT DETECTOR DATA IN PROGRESS ----------------")
    # if not train_transforms:
    #     train_transforms =  T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomRotation(degrees=15),
    #         T.ColorJitter(brightness=(0.5, 1.4), contrast=(0.8, 1.5), saturation=0.1)
    #     ])

    # if not val_transforms:
    #     val_transforms = T.Compose([
    #     T.ToDtype(torch.float32, scale=True)
    # ])

    train_dataset = create_keypt_hrnet_dataset(train_images_path, 
                                        train_annotations_dict,
                                        train_transforms,
                                        idx_to_class,
                                        idx_to_keypts
                                        )

    val_dataset = create_keypt_hrnet_dataset(val_images_path, 
                                        val_annotations_dict,
                                        val_transforms,
                                        idx_to_class,
                                        idx_to_keypts) 
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, val_dataset, train_loader, val_loader

def start_hrnet_training_validation(
  model, 
  train_loader,
  val_loader,
  idx_to_class,
  idx_to_keypts,
  optimizer,
  num_epochs,
  devices,
  samples_to_display=0,
  batch_size=64,
  output_path=".",
  comparision_images=0,
  score_threshold=0.1,
  display=False):
  """
  Starts the training and validation loop with the given parameters
  """
  print("\n--------------- TRAINING & VALIDATION IN PROGRESS ----------------")
  model_name = model.__class__.__name__
  print(f"""
            Model: {model_name}
            Total Training Images: {len(train_loader.dataset)}
            Total Validation Images: {len(val_loader.dataset)}
            Id to Class: {idx_to_class}
            Id to Keypoints: {idx_to_keypts}
            Optimizer: {optimizer}
            Total Epochs: {num_epochs}
            Device: {device}
            Batch Size: {batch_size}
            Output path: {output_path}
            Comparision Images: {comparision_images}
            Score Threshold: {score_threshold}\n
            Display: {display}
              """)

  # Setting up the path for output and analytics and convert ids to names
  model_path, artifacts_path, analytics_path, epoch_map_path, epoch_pr_path = create_keypts_output_infrastructure(output_path, model_name)

  # Display samples of dataset images 
  train_imgs = create_random_keypts_dataset_images(train_loader.dataset, idx_to_keypts, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Training_Samples", artifacts_path=artifacts_path)
  val_imgs = create_random_keypts_dataset_images(val_loader.dataset, idx_to_keypts, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Validation_Samples", artifacts_path=artifacts_path)


def create_random_keypts_dataset_images(dataset, id_to_keypts, num_images=0, title="", padding=10, class_colors=["blue", "red", "green"], artifacts_path=None):
  """
  Creates and returns a grid of 'num_images' random images from the given 'dataset'
  and adds a title and padding if requested. Saves it if necessary.
  """
  if not dataset:
      print(f"{create_random_dataset_images.__name__} No dataset provided to draw samples")
      return

  if  num_images <= 0:
      print(f"{create_random_dataset_images.__name__} Number of images to draw <= 0")
      return

  indexes = [random.randint(0, len(dataset) - 1) for _ in range(num_images)]
  img_tensors = []

  for idx in indexes:
      image, heatmaps, target_mask, class_id = dataset[idx]
      #image.show()
      heatmaps = heatmaps.unsqueeze(0)
      keypts, confidences = decode_heatmaps(heatmaps)

      # Drawing the image back with keypoints
      adjusted_kpts = []
      confs = []
      names = []
      for idx, kp in enumerate(keypts):
        if confidences[idx] <= 0:
          continue
        h = kp[0]
        w = kp[1]
        adjusted_kpts.append((h, w))
        confs.append(confidences[idx])
        names.append(idx_to_keypts[idx])
      denorm_img = denormalize_image_tensor(image, hrnet_mean, hrnet_std)
      pil_img = get_pil_image_from_torch_tensor(denorm_img)
      pil_img = annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=False)
      img_tensors.append(TF.to_tensor(pil_img))
  img_grid = gridify_multiple_tensor_images(img_tensors, title=title)
  img_grid.show()
  # save if artifacts path is provided
  if os.path.exists(artifacts_path):
      img_name = f"{title}.jpg"
      output = os.path.join(artifacts_path, img_name)
      img_grid.save(output)

def test_dataset_creation(img_dir, anno_dict, idx_to_class, idx_to_keypts):
  """
  Simple test to create a dataset with heatmaps, get the first item from it,
  decode the heatmaps back to keypoints, and then draw them back on the image.
  """
  dog_dataset = create_keypt_hrnet_dataset(
        img_dir,
        annotations_dict=anno_dict,
        transforms=None,
        idx_to_class=idx_to_class,
        idx_to_keypts=idx_to_keypts,
        input_size=(IN_H, IN_W),
        sigma=2)

  print(f"Length of dataset {len(dog_dataset)}")
  print(f"Getting first img, heatmap item in dataset ....")
  img, target_heatmaps, target_mask, class_id = dog_dataset[0]
  print(f"Image Shape: {img.shape}")
  print(f"Target mask: {target_mask}")
  
  # Adding batch dim to heatmaps
  target_heatmaps = target_heatmaps.unsqueeze(0)
  print(f"Decoding heatmaps\ntarget_heapmaps: Shape: {target_heatmaps.shape} Value:{target_heatmaps}")
  keypts, confidences = decode_heatmaps(target_heatmaps)
  print(f"Total keypoints: {len(keypts)} Keypoint: {keypts} Confidences:\n{confidences}")
  
  # Drawing the image back with keypoints
  adjusted_kpts = []
  confs = []
  names = []
  for idx, kp in enumerate(keypts):
    if confidences[idx] <= 0:
      continue
    h = kp[0]
    w = kp[1]
    adjusted_kpts.append((h, w))
    confs.append(confidences[idx])
    names.append(idx_to_keypts[idx])
  denorm_img = denormalize_image_tensor(img, hrnet_mean, hrnet_std)
  pil_img = get_pil_image_from_torch_tensor(denorm_img)
  annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=True)

if __name__ == "__main__":
  # Get the modified model
  model = get_model_backbone(False)
  model = modify_model_get_backbone(model)
  model = HRNetCustomPoseEstimator(backbone=model, num_keypoints=17)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Load the dog pose dataset and train the model
  env_file = f".env.keypts_detection"
  from dotenv import load_dotenv
  load_dotenv(env_file)
  train_img_dir = os.getenv("KEYPT_DETECTION_TRAIN_IMAGES_PATH")
  train_anno_dir = os.getenv("KEYPT_DETECTION_TRAIN_ANNOTATIONS_PATH")
  val_img_dir = os.getenv("KEYPT_DETECTION_VALIDATE_IMAGES_PATH")
  val_anno_dir = os.getenv("KEYPT_DETECTION_VALIDATE_ANNOTATIONS_PATH")
  
  train_anno_dict = get_annotation_dict_from_dog_pose_images(train_anno_dir)
  val_anno_dict = get_annotation_dict_from_dog_pose_images(val_anno_dir)
  idx_to_class = get_idx_class_dog_pose_dataset()
  idx_to_keypts = get_idx_keypoints_dog_pose_dataset()

  # To create a test dataset, select an item and display it, use this function
  # test_dataset_creation(img_dir, annotations_dict, idx_to_class, idx_to_keypts)


  # Use this section to create train/val set and train the model  
  train_dataset, val_dataset, train_loader, val_loader = prepare_key_pts_data(
                                                                  train_img_dir,
                                                                  train_anno_dict,
                                                                  val_img_dir,
                                                                  val_anno_dict,
                                                                  idx_to_class,
                                                                  idx_to_keypts)

  num_epochs = int(os.getenv("KEYPT_DETECTION_NUM_EPOCHS", 2))
  learning_rate = 1e-5
  optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
  output_path = os.getenv("KEYPT_DETECTION_OUTPUT_PATH")
  samples = int(os.getenv("KEYPT_DETECTION_NUM_SAMPLES"))
  comparisions = int(os.getenv("KEYPT_DETECTION_COMPARISION_SAMPLES"))
  start_hrnet_training_validation(
                                  model, 
                                  train_loader,
                                  val_loader,
                                  idx_to_class,
                                  idx_to_keypts,
                                  optimizer,
                                  num_epochs,
                                  device,
                                  samples_to_display=samples,
                                  batch_size=64,
                                  output_path=output_path,
                                  comparision_images=comparisions,
                                  score_threshold=0.1,
                                  display=True
                                  )