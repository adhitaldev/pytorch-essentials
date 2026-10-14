"""
General util function that are applicable to all object detection workflows
(training, mAP evaluation, etc)
"""
import torch
import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import v2 as T
from .obj_det_dataset import create_voc_dataset

def prepare_obj_det_data(
                 train_images_path, 
                 train_annotations_path,
                 validation_images_path, 
                 validation_annotations_path,
                 class_labels: 'dict',
                 batch_size=64,
                 train_transforms=None,
                 validate_transforms=None):
    """
    Prepares data based on the batch size, and training and validation paths.
    Returns data loaders for training and validation.
    """
    print(f"Train Images Path: {train_images_path}")
    print(f"Train Annotations Path: {train_annotations_path}")
    print(f"Validate Images Path: {validation_images_path}")
    print(f"Validate Annotations Path: {validation_annotations_path}")
    print(f"Class labels: {class_labels}")
    print(f"Batch size: {batch_size}")

    if not train_transforms:
        train_transforms =  T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=(0.5, 1.4), contrast=(0.8, 1.5), saturation=0.1),
            T.Resize((224, 224)),
            # Random rotation and centercrop can shrink boxes to zero width. Use sanitization of bounding boxes.
            T.SanitizeBoundingBoxes()
        ])

    if not validate_transforms:
        validate_transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float32, scale=True)
        ])

    train_dataset = create_voc_dataset(images_path=train_images_path, annotations_path=train_annotations_path, class_to_labels=class_labels, transforms=train_transforms)
    validate_dataset = create_voc_dataset(images_path=validation_images_path, annotations_path=validation_annotations_path, class_to_labels=class_labels, transforms=validate_transforms)

    print(f"Total training samples {len(train_dataset)} Batch_size: {batch_size}")
    print(f"Total validation samples {len(validate_dataset)} Batch_size: {batch_size}")

    # Data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    return train_dataset, validate_dataset, train_loader, validate_loader


def start_obj_det_training_validation(
                                  model,
                                  train_loader, 
                                  val_loader,
                                  optimizer,
                                  num_epochs,
                                  device, 
                                  batch_size=64,
                                  model_out_path="obj_det_model.pth"):
  """
  Starts the training and validation loop with the given parameters
  """
  model = model.to(device)
  best_map = 0.0
  for epoch in range(num_epochs): 
    train_loss = train_one_obj_det_epoch(model, train_loader, optimizer, device, epoch, num_epochs, batch_size)
    # Evaluation
    val_loss = get_validation_loss(model, val_loader, device)
    metrics = evaluate_mAP(model, val_loader, device)

    print(f"Epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f} | "
          f"mAP {metrics['map']:.3f} | mAP@0.5 {metrics['map_50']:.3f}")
    
    # Saves the best model
    if metrics["map"] > best_map:
        best_map = metrics["map"]
        print(f"Best mAP so far: {best_map}")
        torch.save(model.state_dict(), model_out_path)

def train_one_obj_det_epoch(model, train_loader, optimizer, device, epoch, num_epochs, batch_size):
    """
    Trains a single epoch of a object detection model
    """
  # Training
    model.train()
    running_train_loss = 0
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
    for idx, (images, targets) in enumerate(pbar, start=1):
      images = [img.to(device) for img in images]
      # print(f"Targets need to be tuple : {type(targets)}")
      # print(f"Each target a dictionary: {type(targets[0])}")
      targets = [
        {k:(v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets
      ]
      optimizer.zero_grad()
      loss_dict = model(images, targets)
      loss = sum(loss_dict.values())
      loss.backward()
      optimizer.step()
      running_train_loss += loss.item()
      pbar.set_postfix(images=f"{batch_size * (idx + 1)}/{len(train_loader) * batch_size}", avg_loss=f"{running_train_loss/ idx:.4f}")
    return running_train_loss / len(train_loader)

@torch.no_grad()
def evaluate_mAP(model, val_loader, device):
  """
  Evaluates the mAP scores of the dataset provided.
  """
  model.eval()
  metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
  for images, targets in tqdm.tqdm(val_loader, desc="[Validate]"):
      images = [img.to(device) for img in images]
      outputs = model(images)   # list of dicts: boxes, labels, scores
      preds = [
          {
              "boxes": o["boxes"].cpu(),
              "scores": o["scores"].cpu(),
              "labels": o["labels"].cpu(),
          }
          for o in outputs
      ]
      gts = [
          {
              "boxes": t["boxes"].as_subclass(torch.Tensor).cpu(),
              "labels": t["labels"].cpu(),
          }
          for t in targets
      ]
      metric.update(preds, gts)
  results = metric.compute()
  return {k: (v.tolist() if v.ndim else v.item()) for k, v in results.items()}


def get_validation_loss(model, val_loader, device):
    """
    Get's the validation loss from the provided model
    and the validation loader
    """
    model.train() # needed so that model returns the loss_dict
    total = 0.0
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        total += sum(loss_dict.values()).item()
    return total/len(val_loader)

def _collate(batch):
  """
  Returns a tuple of dictionary instead of a single dictionary
  """
  return tuple(zip(*batch))


