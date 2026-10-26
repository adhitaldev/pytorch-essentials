"""
Modifies the HRNetW32 image classifier while still using the backbone to train with semantic keypoints.
In HRNetW32, every modules downstream of 'incre_modules' make up the complete Image Classification Head
in the official HRNet documentation.
Original paper: "Deep High-Resolution Represetntaion Learning for Human Pose Estimation" (Sun, et. all)
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
import timm
from PIL import Image
from keypt_det_utils import annotate_key_points_on_image

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

def decode_heatmaps(heatmaps, original_img_size=(256, 192)):
    """ 
    Extracts the (x, y) coordinates from the heatmap tensors
    original_img_size in (H, W)
    """
    batch_size, num_keypoints, h_map, w_map = heatmaps.shape
    org_w, org_h = original_img_size

    # Reshape heatmap to 2D matrix per keypoint: (Batch, Num_Keypoints, H_map * W_map)
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_keypoints, -1)

    # Find the pixel index with maximum probability response
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    # Convert 1D flattened index back into (x, y) grid coordinates
    preds = torch.zeros((batch_size, num_keypoints, 2), device=heatmaps.device)
    preds[:, :, 0] = (idx % w_map).float()  # x coordinate on heatmap
    preds[:, :, 1] = (idx // w_map).float()  # y coordinate on heatmap

    # Scale coordinates from heatmap resolution back to original image size
    preds[:, :, 0] *= org_w / w_map
    preds[:, :, 1] *= org_h / h_map

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


def get_keypoint_names_for_dog_pose_dataset():
    """
    Returns the keypoint names for dog_pose_dataset
    """
    dog_pose_kpts = {
        0: "front_left_paw",
        1: "front_left_knee",
        2: "front_left_elbow",
        3: "rear_left_paw",
        4: "rear_left_knee",
        5: "rear_left_elbow",
        6: "front_right_paw",
        7: "front_right_knee",
        8: "front_right_elbow",
        9: "rear_right_paw",
        10: "rear_right_knee",
        11: "rear_right_elbow",
        12: "tail_start",
        13: "tail_end",
        14: "left_ear_base",
        15: "right_ear_base",
        16: "nose",
        17: "chin",
        18: "left_ear_tip",
        19: "right_ear_tip",
        20: "left_eye",
        21: "right_eye",
        22: "withers",
        23: "throat"
    }
    return dog_pose_kpts

if __name__ == "__main__":
    model = get_model_backbone(False)
    model = modify_model_get_backbone(model)
    model = HRNetCustomPoseEstimator(backbone=model, num_keypoints=17)
    print(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_file = f".env.keypts_detection"
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    test_img = os.getenv("KEYPT_EXAMPLE_IMG")
    if not os.path.exists(test_img):
        print(f"Image does not exists")
    
    # Define your dataset's landmark name map
    KEYPOINT_NAMES = {
        0: "Nose",
        1: "Left Eye",
        2: "Right Eye",
        3: "Left Ear",
        4: "Right Ear",
        5: "Left Shoulder",
        6: "Right Shoulder",
        7: "Left Elbow",
        8: "Right Elbow",
        9: "Left Wrist",
        10: "Right Wrist",
        11: "Left Hip",
        12: "Right Hip",
        13: "Left Knee",
        14: "Right Knee",
        15: "Left Ankle",
        16: "Right Ankle"
        }
    keypoints, confidence = predict_pose(model, test_img, device)

    coordinates, names, confidences = [], [], []
    for idx, (x, y) in enumerate(keypoints):
        coordinates.append((x, y))
        names.append(KEYPOINT_NAMES[idx])
        confidences.append(confidence[idx])
    annotate_key_points_on_image(input_img_path=test_img, keypoints=coordinates, names=names,  confidences=confidences, output_path=None, show=True)
       
    
