"""
General util function that are applicable to all object detection workflows
(training, mAP evaluation, etc)
"""
import os
import random
import torch
import tqdm
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import v2 as T
from torchvision.ops import box_iou
import  torchvision.utils as vutils
from .obj_det_dataset import create_voc_dataset
from .vision_utils import gridify_multiple_tensor_images
from ..utils.model_utils import write_to_csv, read_from_csv
from ..utils.data_plotter import plot_x_y, plot_multiX_y, plot_multiY_x

ANALYTICS_FILE="analytics.csv"
EPOCH_MAP_FILE="mAP.csv"
EPOCH_PR_FILE ="precision_recall.csv"
ARTIFACTS_FOLDER="artifacts"
EPOCH="Epoch"
STEP="Step"
LR="Lr"
LOSS="Loss"
TRAIN_LOSS="TrainLoss"
VALIDATE_LOSS="ValidateLoss"
MAP="mAP"
MAP_05="mAP_0.5"
STEP_VS_LOSS_FILE="step_vs_loss.jpg"
TRAIN_VS_VALIDATE_LOSS_FILE="train_vs_validate_loss.jpg"
MAPS_FILE = "mAPs.jpg"
PR_FILE = "precision_recall.jpg"
PRECISION_PER_CLASS = "precision_per_class"
RECALL_PER_CLASS = "recall_per_class"
PRECISION = "precision"
RECALL = "recall"

# Keep track of global steps
steps = 0


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
    print("\n--------------- PREPARING DATA IN PROGRESS ----------------")
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
                                  class_labels,
                                  optimizer,
                                  num_epochs,
                                  device,
                                  samples_to_display=0,
                                  batch_size=64,
                                  output_path=".",
                                  comparison_images=0,
                                  id_to_names=None,
                                  score_threshold=0.5,
                                  display=False):
    """
    Starts the training and validation loop with the given parameters
    """
    print("\n--------------- TRAINING & VALIDATION IN PROGRESS ----------------")
    model_name=model.__class__.__name__
    print(f"""
            Model: {model_name}
            Total Training Images: {len(train_loader.dataset)}
            Total Validation Images: {len(val_loader.dataset)}
            Class Labels: {class_labels}
            Optimizer: {optimizer}
            Total Epochs: {num_epochs}
            Device: {device}
            Batch Size: {batch_size}
            Output path: {output_path}
            Comparision Images: {comparison_images}
            Id to Names: {id_to_names}
            Score Threshold: {score_threshold}\n
            Display: {display}
            """)

    # Setting up the path for output and analytics and convert ids to names
    model_path, artifacts_path, analytics_path, epoch_map_path, epoch_pr_path = _create_output_infrastructure(output_path, model_name)
    id_to_names = {v:k for k, v in class_labels.items()}

    # Display samples of dataset images 
    train_imgs = create_random_dataset_images(train_loader.dataset, id_to_names, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Training_Samples", artifacts_path=artifacts_path)
    val_imgs = create_random_dataset_images(val_loader.dataset, id_to_names, num_images=samples_to_display, title=f"{model_name}_{samples_to_display}_Validation_Samples", artifacts_path=artifacts_path)

    if display:
        train_imgs.show()
        val_imgs.show()

    model = model.to(device)
    best_map = 0.0
    global steps
    steps = 0

    comparison_indexes = None
    if comparison_images > 0:
        comparison_indexes = [random.randint(0, len(val_loader.dataset) - 1) for _ in range(comparison_images)]

    for epoch in range(num_epochs): 
        train_loss = train_one_obj_det_epoch(model, train_loader, optimizer, device, epoch, num_epochs, batch_size, analytics_path)
        # Evaluation
        val_loss = get_validation_loss(model, val_loader, device)
        metrics = evaluate_mAP(model, val_loader, device, epoch, num_epochs, batch_size)
        precision_recall = detection_pr(model, val_loader, device, epoch, id_to_names, score_thresh=0.5, iou_thresh=0.5)
        print(f"Epoch {epoch + 1}: train {train_loss:.4f} | val {val_loss:.4f} | "
        f"mAP {metrics['map']:.3f} | mAP@0.5 {metrics['map_50']:.3f}\nPR: {precision_recall}")
        write_to_csv(precision_recall, epoch_pr_path)

        epoch_map_results = {
                                EPOCH: [epoch + 1],
                                TRAIN_LOSS: [train_loss],
                                VALIDATE_LOSS: [val_loss],
                                MAP: [metrics['map']],
                                MAP_05: [metrics['map_50']]}

        write_to_csv(epoch_map_results, epoch_map_path)

        # Saves the best model
        if metrics["map"] > best_map:
            best_map = metrics["map"]
            print(f"Best mAP so far: {best_map}")
            torch.save(model.state_dict(), model_path)

        # If comparison images is greater than zero, invokes the function to show them
        comp_img = create_groundtruth_prediction_comparision(model, device, val_loader.dataset, comparison_indexes, epoch, id_to_names, score_threshold, artifacts_path)
        if display:
            comp_img.show()

    # Generate the relevant analytics results
    _prepare_final_analytics(analytics_path, epoch_map_path, epoch_pr_path, artifacts_path)

    print("\n--------------- TRAINING & VALIDATION COMPLETE ----------------")

def train_one_obj_det_epoch(model, train_loader, optimizer, device, epoch, num_epochs, batch_size, analytics_path):
    """
    Trains a single epoch of a object detection model
    """
  # Training
    model.train()
    running_train_loss = 0
    images_so_far = 0
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
      images_so_far += len(images)
      global steps
      steps += 1
      pbar.set_postfix(steps=f"{steps}", images=f" {images_so_far}/{len(train_loader.dataset)}", avg_loss=f"{running_train_loss/ idx:.4f}")
    
      # Save the analytics
      if analytics_path:
        write_to_csv({EPOCH: [epoch + 1], STEP: [steps], LR: optimizer.param_groups[0]['lr'], LOSS: [loss.item()]}, analytics_path)
      #print(f"Epoch: {epoch + 1} Step: {idx + 1} Loss: {loss.item()} ")
    return running_train_loss / len(train_loader)

@torch.no_grad()
def evaluate_mAP(model, val_loader, device, epoch, num_epochs, batch_size):
  """
  Evaluates the mAP scores of the dataset provided.
  """
  model.eval()
  metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
  pbar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validate]")
  images_so_far = 0
  for images, targets in pbar:
      images = [img.to(device) for img in images]
      images_so_far += len(images)
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
      pbar.set_postfix(images=f" {images_so_far}/{len(val_loader.dataset)}")
  results = metric.compute()
  return {k: (v.tolist() if v.ndim else v.item()) for k, v in results.items()}

@torch.no_grad()
def detection_pr(model, val_loader, device, epoch, id_to_names, score_thresh=0.5, iou_thresh=0.5):
    """
    Calculates the precision and recall of the model with the given score and iou thresholds.
    Precision - How often is the model's prediction correct? P = TP/(TP + FP)
    Recall - How many of the positive cases did the model correctly identify ? R = TP / (TP + FN)
    """
    model.eval()
    num_classes = len(id_to_names) + 1          # index 0 = background
    # True positives, false positives, and false negatives tensors
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            mask = out["scores"] >= score_thresh
            p_boxes = out["boxes"][mask].cpu()
            p_labels = out["labels"][mask].cpu()
            p_scores = out["scores"][mask].cpu()

            g_boxes = tgt["boxes"].as_subclass(torch.Tensor).cpu()
            g_labels = tgt["labels"].cpu()

            matched_gt = set()
            order = p_scores.argsort(descending=True)

            if len(p_boxes) and len(g_boxes):
                ious = box_iou(p_boxes, g_boxes)
            else:
                ious = torch.zeros(len(p_boxes), len(g_boxes))

            for i in order.tolist():
                c = int(p_labels[i])
                best_iou, best_j = 0.0, -1
                for j in range(len(g_boxes)):
                    if j in matched_gt or int(g_labels[j]) != c:
                        continue
                    if ious[i, j] > best_iou:
                        best_iou, best_j = float(ious[i, j]), j
                if best_iou >= iou_thresh:
                    matched_gt.add(best_j)
                    tp[c] += 1
                else:
                    fp[c] += 1

            for j in range(len(g_boxes)):
                if j not in matched_gt:
                    fn[int(g_labels[j])] += 1

    # Computed once, after all batches
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    overall_p = tp[1:].sum() / (tp[1:].sum() + fp[1:].sum() + eps)
    overall_r = tp[1:].sum() / (tp[1:].sum() + fn[1:].sum() + eps)

    res = { 
           EPOCH: [epoch],
           PRECISION: [overall_p.item()],
           RECALL : [overall_r.item()]
           }

    for label_id in range(1, num_classes):
        name = id_to_names[label_id]
        res[f"{PRECISION}_{name}"] = [precision[label_id].item()]
        res[f"{RECALL}_{name}"] = [recall[label_id].item()]
    return res

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


def create_random_dataset_images(dataset, id_to_names, num_images=0, title="", padding=10, class_colors=["blue", "red", "green"], artifacts_path=None):
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
        image, target = dataset[idx]
        # print(f"Info {image.dtype}, {image.min().item()}, {image.max().item()}")
        labels = ["Unknown" for _ in range(len(targets["labels"]))] if not id_to_names else [id_to_names[int(l)] for l in target["labels"]]
        annotated_img = vutils.draw_bounding_boxes(
            image=image, 
            boxes=target["boxes"],
            colors=class_colors,
            labels=labels,
            width=3
        )
        img_tensors.append(annotated_img)
    img_grid = gridify_multiple_tensor_images(img_tensors, padding=padding, title=title)

    # save if artifacts path is provided
    if os.path.exists(artifacts_path):
        if not title:
            title="data"
        img_name = f"{title}.jpg"
        output = os.path.join(artifacts_path, img_name)
        img_grid.save(output)
    return img_grid

def create_groundtruth_prediction_comparision(model, device, val_dataset, image_indexes, epoch, id_to_names, score_threshold=0.0, artifacts_path=None):
    """
    Creates and returns a grid of side by side image comparion of groundtruth bounding boxes
    with the predicted bounding boxes. Saves it if necessary.
    """
    if not val_dataset or not image_indexes:
        print(f"{create_groundtruth_prediction_comparision.__name__}No dataset and/or images for groundtruth viewing")
        return
    img_tensors = []
    model.eval()
    print(f"Creating comparison visualization for epoch {epoch}... ")
    for idx in image_indexes:
        image, target = val_dataset[idx]
        output = model([image.to(device)])[0]

        # draw_bounding_boxes needs uint8 in 0-255
        img_u8 = (image * 255).clamp(0, 255).to(torch.uint8).cpu()
        # Ground truth
        gt_boxes = target["boxes"].as_subclass(torch.Tensor).cpu()
        gt_names = [id_to_names[int(l)] for l in target["labels"]]
        gt_img = vutils.draw_bounding_boxes(img_u8, gt_boxes, labels=gt_names,
                                    colors="lime", width=3)

        # Filter out the scores that are below the threshold
        valid = output["scores"] >= score_threshold
        pred_boxes = output["boxes"][valid].cpu()
        pred_names = [f"{id_to_names[int(l)]} {s:.2f}"
               for l, s in zip(output["labels"][valid].cpu(), output["scores"][valid].cpu())]
        pred_img = vutils.draw_bounding_boxes(img_u8, pred_boxes, labels=pred_names, colors="red", width=3)

        # Add both to image tensors
        img_tensors.append(gt_img)
        img_tensors.append(pred_img)

    img_grid = gridify_multiple_tensor_images(img_tensors, title=f"After Epoch {epoch + 1}")

    # save if artifacts path is provided
    if os.path.exists(artifacts_path):
        img_name = f"epoch_{epoch + 1}.jpg"
        output = os.path.join(artifacts_path, img_name)
        img_grid.save(output)
    return img_grid

def _prepare_final_analytics(analytics_path, epoch_map_path, epoch_pr_path, artifacts_path):
    """
    Reads from the analytics files and then generates the artifacts
    """
    print(f"Preparing analytics ...")
    # Getting and plotting data from the analytics file
    data = read_from_csv(analytics_path)
    print(f"Reading from analytics file: {analytics_path}")
    steps = data[STEP].tolist()
    losses = data[LOSS].tolist()
    save_path=os.path.join(artifacts_path, STEP_VS_LOSS_FILE)
    plot_x_y(steps, losses, x_label="steps", y_label="loss", title="Step vs Loss", save_path=save_path)

    # Getting and plotting data from the mAP file
    print(f"Reading from mAP file: {epoch_map_path}")
    mdata = read_from_csv(epoch_map_path)
    epochs = mdata[EPOCH].tolist()
    train_loss = mdata[TRAIN_LOSS].tolist()
    val_loss = mdata[VALIDATE_LOSS].tolist()
    mAP = mdata[MAP].tolist()
    mAP_05 = mdata[MAP_05].tolist()

    # Training vs Validation loss on epochs
    y_values = [train_loss, val_loss]
    y_names = ["TrainLoss", "ValidateLoss"]
    y_colors = ["blue", "red"]
    save_path = os.path.join(artifacts_path, TRAIN_VS_VALIDATE_LOSS_FILE)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="Training vs Validation Loss", save_path=save_path)
    
    # mAPs on epochs
    y_values = [mAP, mAP_05]
    y_names = ["mAP", "mAP_0.5"]
    y_colors = ["green", "orange"]
    save_path = os.path.join(artifacts_path, MAPS_FILE)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="mAPs", save_path=save_path)
    
    # Precision and recall
    print(f"Reading from precision recall file: {epoch_pr_path}")
    prdata = read_from_csv(epoch_pr_path)
    epochs = prdata[EPOCH].tolist()
    precision = prdata[PRECISION].tolist()
    recall = prdata[RECALL].tolist()

    y_values = [precision, recall]
    y_names = ["precision", "recall"]
    y_colors = ["gray", "blue"]
    save_path = os.path.join(artifacts_path, PR_FILE)
    plot_multiY_x(y_values=y_values, y_names=y_names, y_colors=y_colors, x_values=epochs, x_title="Epochs", title="Overall Precision vs Recall", save_path=save_path)
    

def _create_output_infrastructure(output_path, model_name):
    """
    Creates the necessary folder strucutures to output the model, artifacts etc.
    """
    folder_name = datetime.now().strftime(f"%Y%m%d_%H%M") + "_" + model_name
    final_path = os.path.join(output_path, folder_name)
    model_path = os.path.join(final_path, model_name + ".pt")
    analytics_path = os.path.join(final_path, ANALYTICS_FILE)
    epoch_map_path = os.path.join(final_path, EPOCH_MAP_FILE)
    epoch_pr_path = os.path.join(final_path, EPOCH_PR_FILE)
    artifacts_path = os.path.join(final_path, ARTIFACTS_FOLDER)
    if not os.path.exists(artifacts_path):
        os.makedirs(artifacts_path)
    return model_path, artifacts_path, analytics_path, epoch_map_path, epoch_pr_path

def _collate(batch):
  """
  Returns a tuple of dictionary instead of a single dictionary
  """
  return tuple(zip(*batch))


