"""
Modifies the HRNetW32 image classifier while still using the backbone to train with semantic keypoints.
In HRNetW32, every modules downstream of 'incre_modules' make up the complete Image Classification Head
in the official HRNet documentation.
Original paper: "Deep High-Resolution Represetntaion Learning for Human Pose Estimation" (Sun, et. all)
"""
import math
import numpy as np
import os
import random
import timeit
import torch
import tqdm
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from PIL import Image
from torch_vision.vision_utils import annotate_key_points_on_image, visualize_tensor_image, get_pil_image_from_torch_tensor, denormalize_image_tensor, gridify_multiple_tensor_images
from torch_vision.keypt_hrnet_dataset import HRnetKeypointDetectionDataset, create_keypt_hrnet_dataset, get_annotation_dict_from_dog_pose_images, get_idx_keypoints_dog_pose_dataset, get_idx_class_dog_pose_dataset
from torch_vision.keypt_utils import create_keypts_output_infrastructure, generate_final_analytics, load_keypts_env, EPOCH, STEP, LR, LOSS, TRAIN_LOSS, VALIDATION_LOSS, MEAN_PIXEL_ERROR, MEDIAN_PIXEL_ERROR,PRECISION, RECALL, F1_SCORE
from utils.model_utils import write_to_csv, read_from_csv

IN_H, IN_W = 256, 192 # Standard HRnet input dimension
hrnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
hrnet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DISTANCE_THRESHOLD=5.0
CONFIDENCE_THRESHOLD=0.2

# Keep track of global steps
steps = 0

class HRNetCustomPoseEstimator(nn.Module):
  def __init__(self, backbone, num_keypoints=9, freeze_backbone=True):
    super().__init__()
    self.backbone = backbone
    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

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
    #     train_transforms =  transforms.Compose([
    #         # transforms.RandomHorizontalFlip(),
    #         # transforms.RandomRotation(degrees=15),
    #         transforms.ColorJitter(brightness=(0.5, 1.4), contrast=(0.8, 1.5), saturation=0.1)
    #     ])

    # if not val_transforms:
    #     val_transforms = transforms.Compose([
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
  lr_scheduler,
  loss_func,
  num_epochs,
  device,
  samples_to_display=0,
  batch_size=64,
  output_path=".",
  comparison_images=0,
  score_threshold=0.0,
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
            Loss Func: {loss_func}
            Total Epochs: {num_epochs}
            Device: {device}
            Batch Size: {batch_size}
            Output path: {output_path}
            Comparison Images: {comparison_images}
            Score Threshold: {score_threshold}\n
            Display: {display}
              """)

  # Setting up the path for output and analytics and convert ids to names
  model_path, artifacts_path, analytics_path, epoch_results_path = create_keypts_output_infrastructure(output_path, model_name)

  # Display samples of dataset images 
  train_imgs = create_random_keypts_dataset_images(train_loader.dataset, idx_to_keypts, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Training_Samples", artifacts_path=artifacts_path)
  val_imgs = create_random_keypts_dataset_images(val_loader.dataset, idx_to_keypts, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Validation_Samples", artifacts_path=artifacts_path)

  if display:
    train_imgs.show()
    val_imgs.show()

  comparison_indexes = None
  if comparison_images > 0:
      comparison_indexes = [random.randint(0, len(val_loader.dataset) - 1) for _ in range(comparison_images)]

  # Training and validation through epochs
  model = model.to(device)
  global steps
  steps = 0
  best_median_px_error = math.inf 
  best_validation_loss = math.inf
  for epoch in range(num_epochs):
    avg_train_loss = train_one_epoch(model, train_loader, device, optimizer,loss_func, epoch, num_epochs, batch_size, analytics_path)
    eval_result = evaluate_one_epoch_with_pr(model, val_loader, device, loss_func, epoch, num_epochs, batch_size)

    # Update the learning rate scheduler after each epoch
    lr_scheduler.step()  

    # Append train loss to the eval_result  and change to create a dataframe for output
    res = {
      EPOCH: [epoch + 1],
      TRAIN_LOSS: [avg_train_loss],
      VALIDATION_LOSS: [eval_result[VALIDATION_LOSS]],
      MEAN_PIXEL_ERROR: [eval_result[MEAN_PIXEL_ERROR]],
      MEDIAN_PIXEL_ERROR: [eval_result[MEDIAN_PIXEL_ERROR]],
      PRECISION: [eval_result[PRECISION]],
      RECALL: [eval_result[RECALL]],
      F1_SCORE: [eval_result[F1_SCORE]],
      LR: [lr_scheduler.get_last_lr()[0]]
    }
    print(res)
    if analytics_path:
      write_to_csv(res, epoch_results_path)

    # Save the best model for least pixel error and least validation loss
    curr_median_pixel_error = eval_result[MEDIAN_PIXEL_ERROR]
    curr_validation_loss = eval_result[VALIDATION_LOSS]
    if curr_median_pixel_error < best_median_px_error and curr_validation_loss < best_validation_loss:
      best_median_px_error = curr_median_pixel_error
      best_validation_loss = curr_validation_loss
      print(f"Updating best median pixel error: {best_median_px_error} validation loss: {best_validation_loss} and saving model")
      torch.save(model.state_dict(), model_path)
    
    comp_img = create_groundtruth_prediction_comparison(model, device, val_loader.dataset, comparison_indexes, epoch + 1, idx_to_keypts, score_threshold=score_threshold, artifacts_path=artifacts_path)
    if display:
      comp_img.show()
    
  generate_final_analytics(analytics_path, epoch_results_path, artifacts_path)

  print("\n--------------- TRAINING & VALIDATION COMPLETE ----------------")

def train_one_epoch(model, train_loader, device, optimizer, loss_func, epoch, num_epochs, batch_size, analytics_path):
  """
  Trains a single epoch of the HRnet pose estimator
  """
  model.train()
  running_train_loss = 0
  images_so_far = 0
  pbar = pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
  for idx,(images, target_heatmaps, target_mask, class_id) in enumerate(pbar, start=1):
    images = images.to(device) # required shape (batch_size, 3, 256, 192)
    target_heatmaps = target_heatmaps.to(device) # required shape (batch_size, num_keypts, 64, 48)
    target_mask = target_mask.to(device) # required shape (batch_size, num_keypts, 1)
    optimizer.zero_grad()
    pred_heatmaps = model(images)# required shape (batch_size, num_keypts, 64, 48) i.e. same as target_heatmaps
    
    # Raw element wise squared differences [batch_size, num_keypts, 64, 48]
    raw_loss = loss_func(pred_heatmaps, target_heatmaps)

    # Average spatial dimension loss
    avg_kpt_loss = raw_loss.mean(dim=[2, 3])

    # Visibility mask 
    masked_loss = avg_kpt_loss * target_mask.squeeze(-1)
    valid_keypoints = torch.clamp(target_mask.sum(), min=1.0) # avoid div by zero if there are no valid key_pts
    loss = 0.5 * (masked_loss.sum() / valid_keypoints)

    # Backward pass & parameter optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Running loss 
    running_train_loss += loss.item()
    images_so_far += len(images)
    curr_avg_loss = running_train_loss / idx
    global steps
    steps += 1
    pbar.set_postfix(steps=f"{steps}", images=f"{images_so_far} / {len(train_loader.dataset)}", avg_loss=f"{curr_avg_loss:.4f}")

    # Save the analytics on a 'step' level
    result = {
      EPOCH: [epoch + 1],
      STEP: [steps],
      LR: optimizer.param_groups[0]['lr'],
      LOSS: [curr_avg_loss]
      }
    if analytics_path:
      write_to_csv(result, analytics_path)
  avg_training_loss = running_train_loss / len(train_loader)
  return avg_training_loss

@torch.no_grad()
def evaluate_one_epoch(model, val_loader, device, loss_func, epoch,num_epochs, batch_size):
  """
  Evaluates the scores of the validation dataset.
  """
  model.eval()
  pbar = tqdm.tqdm(
      val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validate]"
  )

  total_val_loss = 0.0
  total_valid_keypoints = 0.0
  total_pixel_distance_error = 0.0
  pixel_errors = []  # Stores individual keypoint pixel errors

  for idx, (images, target_heatmaps, target_mask, class_id) in enumerate(pbar, start=1):
    images = images.to(device, non_blocking=True)
    target_heatmaps = target_heatmaps.to(device, non_blocking=True)
    target_mask = target_mask.to(device, non_blocking=True)

    pred_heatmaps = model(images)

    # 1. Masked MSE Loss
    raw_loss = loss_func(pred_heatmaps, target_heatmaps)
    avg_kpt_loss = raw_loss.mean(dim=[2, 3])  # Spatial reduction -> [B, 24]
    masked_loss = avg_kpt_loss * target_mask.squeeze(-1)

    batch_valid_kpts = target_mask.sum().item()
    # Accumulate unnormalized loss sum and keypoint count
    total_val_loss += 0.5 * masked_loss.sum().item()
    total_valid_keypoints += batch_valid_kpts

    # 2. Pixel Distance Metrics (256x192 canvas)
    b, k, h_hm, w_hm = pred_heatmaps.shape
    pred_flat = pred_heatmaps.reshape(b, k, -1)
    targ_flat = target_heatmaps.reshape(b, k, -1)

    _, pred_idx = torch.max(pred_flat, dim=2)
    _, targ_idx = torch.max(targ_flat, dim=2)

    pred_x = (pred_idx % w_hm).float() * 4.0
    pred_y = (pred_idx // w_hm).float() * 4.0
    targ_x = (targ_idx % w_hm).float() * 4.0
    targ_y = (targ_idx // w_hm).float() * 4.0

    euclidean_dist = torch.sqrt(
        (pred_x - targ_x) ** 2 + (pred_y - targ_y) ** 2
    )

    valid_mask = target_mask.squeeze(-1) > 0
    if valid_mask.any():
      valid_errs = euclidean_dist[valid_mask].cpu().numpy()
      total_pixel_distance_error += valid_errs.sum()
      pixel_errors.extend(valid_errs.tolist())

    # Running display normalized by keypoints seen so far
    running_norm = max(total_valid_keypoints, 1.0)
    running_avg_loss = total_val_loss / running_norm
    running_pixel_err = total_pixel_distance_error / running_norm

    pbar.set_postfix(
        val_loss=f"{running_avg_loss:.5f}",
        mean_px_err=f"{running_pixel_err:.2f}px",
    )

  # Final epoch reductions
  final_norm = max(total_valid_keypoints, 1.0)
  avg_val_loss = total_val_loss / final_norm
  final_mean_pixel_err = total_pixel_distance_error / final_norm
  final_median_pixel_error = (
      float(np.median(pixel_errors)) if pixel_errors else 0.0
  )
  return final_mean_pixel_err, final_median_pixel_error, avg_val_loss

def evaluate_one_epoch_with_pr(
    model,
    val_loader,
    device,
    loss_func,
    epoch,
    num_epochs,
    conf_thresh=CONFIDENCE_THRESHOLD,
    dist_thresh=DISTANCE_THRESHOLD,
):
  """
  Evaluates validation loss, pixel errors, and Precision/Recall/F1 score.
  """

  model.eval()
  pbar = tqdm.tqdm(
      val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validate]"
  )

  total_val_loss = 0.0
  total_valid_keypoints = 0.0
  total_pixel_distance_error = 0.0
  pixel_errors = []

  total_tp = 0
  total_fp = 0
  total_fn = 0

  with torch.no_grad():
    for images, target_heatmaps, target_mask, class_id in pbar:
      images = images.to(device, non_blocking=True)
      target_heatmaps = target_heatmaps.to(device, non_blocking=True)
      target_mask = target_mask.to(device, non_blocking=True)

      # Sigmoid ensures heatmap values map to [0.0, 1.0] for confidence thresholding
      pred_heatmaps = torch.sigmoid(model(images))

      # 1. Masked MSE Loss
      raw_loss = loss_func(pred_heatmaps, target_heatmaps)
      avg_kpt_loss = raw_loss.mean(dim=[2, 3])
      masked_loss = avg_kpt_loss * target_mask.squeeze(-1)

      batch_valid_kpts = target_mask.sum().item()
      total_val_loss += 0.5 * masked_loss.sum().item()
      total_valid_keypoints += batch_valid_kpts

      # 2. Extract peak coordinates and confidences
      b, k, h_hm, w_hm = pred_heatmaps.shape
      pred_flat = pred_heatmaps.reshape(b, k, -1)
      targ_flat = target_heatmaps.reshape(b, k, -1)

      pred_confs, pred_idx = torch.max(pred_flat, dim=2)
      _, targ_idx = torch.max(targ_flat, dim=2)

      # Scale to 256x192 canvas
      pred_x = (pred_idx % w_hm).float() * 4.0
      pred_y = (pred_idx // w_hm).float() * 4.0
      targ_x = (targ_idx % w_hm).float() * 4.0
      targ_y = (targ_idx // w_hm).float() * 4.0

      euclidean_dist = torch.sqrt(
          (pred_x - targ_x) ** 2 + (pred_y - targ_y) ** 2
      )

      # 3. Ground Truth vs Prediction Masks
      is_gt_visible = target_mask.squeeze(-1) > 0  # [B, 24]
      is_pred_positive = pred_confs >= conf_thresh  # [B, 24]
      is_within_dist = euclidean_dist <= dist_thresh  # [B, 24]

      # TP: GT visible, predicted positive, and within pixel threshold
      tp = (is_gt_visible & is_pred_positive & is_within_dist).sum().item()

      # FP: Predicted positive, but either GT was absent OR location was wrong
      fp = (is_pred_positive & (~is_gt_visible | ~is_within_dist)).sum().item()

      # FN: GT was visible, but either predicted negative OR location was wrong
      fn = (is_gt_visible & (~is_pred_positive | ~is_within_dist)).sum().item()

      total_tp += tp
      total_fp += fp
      total_fn += fn

      # Pixel error tracking on ground-truth visible keypoints
      if is_gt_visible.any():
        valid_errs = euclidean_dist[is_gt_visible].cpu().numpy()
        total_pixel_distance_error += valid_errs.sum()
        pixel_errors.extend(valid_errs.tolist())

  # 4. Final Metric Computations
  final_norm = max(total_valid_keypoints, 1.0)
  final_val_loss = total_val_loss / final_norm
  final_mean_pixel_err = total_pixel_distance_error / final_norm
  final_median_pixel_err = (
      float(np.median(pixel_errors)) if pixel_errors else 0.0
  )

  precision = (
      total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
  )
  recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
  f1_score = (
      (2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0
      else 0.0
  )

  return {
      EPOCH: epoch + 1,
      VALIDATION_LOSS: final_val_loss,
      MEAN_PIXEL_ERROR: final_mean_pixel_err,
      MEDIAN_PIXEL_ERROR: final_median_pixel_err,
      PRECISION: precision,
      RECALL: recall,
      F1_SCORE: f1_score
  }

@torch.no_grad()
def create_groundtruth_prediction_comparison(model, device, val_dataset, image_indexes, epoch, id_to_keypts, score_threshold=0.0, artifacts_path=None):
    """
    Creates and returns a grid of side by side image comparion of groundtruth keypoints
    with the predicted keypoints. Saves it if necessary.
    """
    if not val_dataset or not image_indexes:
        print(f"{create_groundtruth_prediction_comparison.__name__}No dataset and/or images for groundtruth viewing")
        return
    img_tensors = []
    model.eval()
    print(f"Creating comparison visualization for epoch {epoch}... ")
    for idx in image_indexes:
      image, target_heatmaps, target_mask, class_id = val_dataset[idx]
      target_heatmaps =target_heatmaps.unsqueeze(0)

      adjusted_kpts = []
      confs = []
      names = []

      # Ground truth image and keypoints
      keypts, confidences = decode_heatmaps(target_heatmaps)
      for idx, kp in enumerate(keypts):
        h = kp[0]
        w = kp[1]
        adjusted_kpts.append((h, w))
        confs.append(confidences[idx])
        names.append(id_to_keypts[idx])
      denorm_img = denormalize_image_tensor(image, hrnet_mean, hrnet_std)
      pil_img = get_pil_image_from_torch_tensor(denorm_img)
      pil_img = annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=False)
      img_tensors.append(TF.to_tensor(pil_img))

      # Predicted keypoints 
      adjusted_kpts = []
      confs = []
      names = []
      image = image.unsqueeze(0).to(device)
      pred_heatmaps = torch.sigmoid(model(image))
      keypts, confidences = decode_heatmaps(pred_heatmaps)
      for idx, kp in enumerate(keypts):
        w = kp[0]
        h = kp[1]
        adjusted_kpts.append((h, w))
        confs.append(confidences[idx])
        names.append(id_to_keypts[idx])
      
      denorm_img = denormalize_image_tensor(image.squeeze(0), hrnet_mean, hrnet_std)
      pil_img = get_pil_image_from_torch_tensor(denorm_img)
      pil_img = annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=False)
      img_tensors.append(TF.to_tensor(pil_img))

    img_grid = gridify_multiple_tensor_images(img_tensors, title=f"After Epoch {epoch}", cols=2, cell_size=(IN_W * 2 , IN_H * 2))

    # save if artifacts path is provided
    if os.path.exists(artifacts_path):
        img_name = f"epoch_{epoch}.jpg"
        output = os.path.join(artifacts_path, img_name)
        img_grid.save(output)
    return img_grid

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
        names.append(id_to_keypts[idx])
      denorm_img = denormalize_image_tensor(image, hrnet_mean, hrnet_std)
      pil_img = get_pil_image_from_torch_tensor(denorm_img)
      pil_img = annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=False)
      img_tensors.append(TF.to_tensor(pil_img))
  img_grid = gridify_multiple_tensor_images(img_tensors, title=title)
  # save if artifacts path is provided
  if os.path.exists(artifacts_path):
      img_name = f"{title}.jpg"
      output = os.path.join(artifacts_path, img_name)
      img_grid.save(output)
  return img_grid

def infer_on_image(model, pil_img, id_to_keypoints):
  """
  Infers on an image
  """
  model.eval()
  pil_img = pil_img.convert('RGB')
  image_resized = pil_img.resize((IN_W, IN_H), Image.BILINEAR)
  transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=hrnet_mean, std= hrnet_std)
            ])
  image_tensor = transform(image_resized)
  image_tensor = image_tensor.unsqueeze(0)
  with torch.no_grad():
    begin = timeit.default_timer()
    pred_heatmaps = torch.sigmoid(model(image_tensor))
    end = timeit.default_timer()
    keypts, confidences = decode_heatmaps(pred_heatmaps)
  adjusted_kpts = []
  confs = []
  names = []
  for idx, kp in enumerate(keypts):
      w = kp[0]
      h = kp[1]
      adjusted_kpts.append((h, w))
      confs.append(confidences[idx])
      names.append(id_to_keypoints[idx])

  denorm_img = denormalize_image_tensor(image_tensor.squeeze(0), hrnet_mean, hrnet_std)
  pil_img = get_pil_image_from_torch_tensor(denorm_img)
  pil_img = annotate_key_points_on_image(pil_img, keypoints=adjusted_kpts, names=names,  confidences=confs, output_path=None, show=False)
  inf_time = end - begin
  return pil_img, inf_time


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

def get_trained_model(model_path, num_keypoints=9):
  model = get_model_backbone(False)
  model = modify_model_get_backbone(model)
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = HRNetCustomPoseEstimator(backbone=model, num_keypoints=num_keypoints)
  checkpoint = torch.load(model_path, map_location=device, weights_only=True)

  # 3. Handle checkpoints saved either as raw state_dict or as a wrapper dict
  if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
  elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
  else:
    state_dict = checkpoint  # The checkpoint is the raw OrderedDict

    # 4. Load weights into the model
    model.load_state_dict(state_dict)
  
  return model

def run_keydet_training():
   # Load the dog pose dataset and train the model
  load_keypts_env()
  train_img_dir = os.getenv("KEYPT_DETECTION_TRAIN_IMAGES_PATH")
  train_anno_dir = os.getenv("KEYPT_DETECTION_TRAIN_ANNOTATIONS_PATH")
  val_img_dir = os.getenv("KEYPT_DETECTION_VALIDATE_IMAGES_PATH")
  val_anno_dir = os.getenv("KEYPT_DETECTION_VALIDATE_ANNOTATIONS_PATH")
  
  train_anno_dict = get_annotation_dict_from_dog_pose_images(train_anno_dir)
  val_anno_dict = get_annotation_dict_from_dog_pose_images(val_anno_dir)
  idx_to_class = get_idx_class_dog_pose_dataset()
  idx_to_keypts = get_idx_keypoints_dog_pose_dataset()

    # Get the modified model
  model = get_model_backbone(False)
  model = modify_model_get_backbone(model)
  model = HRNetCustomPoseEstimator(backbone=model, num_keypoints=len(idx_to_keypts))
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # To create a test dataset, select an item and display it, uncomment the following line
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
  learning_rate = float(os.getenv("KEYPT_DETECTION_LEARNING_RATE", 0.0001))
  #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=learning_rate, betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=1e-4)

  lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
  
  loss_func = nn.MSELoss(reduction="none") # 'none' enables manual visibility masking
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
                                  lr_scheduler,
                                  loss_func,
                                  num_epochs,
                                  device,
                                  samples_to_display=samples,
                                  batch_size=64,
                                  output_path=output_path,
                                  comparison_images=comparisions,
                                  score_threshold=CONFIDENCE_THRESHOLD,
                                  display=True
                                  )


if __name__ == "__main__":
  run_keydet_training()