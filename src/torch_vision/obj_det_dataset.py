import os
import random
import torch
from torchvision.transforms import transforms
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision import tv_tensors
from PIL import Image
from torchvision.datasets import VOCDetection
from xml.etree import ElementTree as ET
from pathlib import Path
from torchvision.transforms import v2 as T

img_extensions =('.jpg', 'jpeg', '.png', 'jpeg')
anno_extensions = '.xml'

class ObjectDetectionDataset(Dataset):
    """
    Generic dataset class for object detection tasks. Takes in an images and annotations folders (VOC format), transformations
    , class to labels, and then creates a dataset out of it.
    """
    def __init__(self, images_dir, annotations_dir, transforms=None, class_to_labels=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.class_to_labels = class_to_labels

        print(f"\n--- Creating object detection dataset ---")
        print(f"Image dir: {self.images_dir}")
        print(f"Annotation dir: {self.annotations_dir}")
        print(f"Transforms: {self.transforms}")
        print(f"Class labels: {class_to_labels}")

        # sort out the files
        self.images = set()
        self.annotations = set()
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(img_extensions):
                self.images.add(file)
        
        for file in os.listdir(self.annotations_dir):
            if file.lower().endswith(anno_extensions):
                self.annotations.add(file)

        # Check if all annotations are available
        missing_annotations = []
        for img in self.images:
            img_with_xml = img + anno_extensions
            if img_with_xml not in self.annotations:
                missing_annotations.append(img_with_xml)
        
        print(f"Total Images: {len(self.images)}")
        print(f"Total Annotations: {len(self.annotations)}")
        print (f"Missing annotations for: {missing_annotations}")
      
        self.images = sorted(list(self.images))
        self.annotations = sorted(list(self.annotations))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = os.path.join(self.images_dir, self.images[idx])
        anno = os.path.join(self.annotations_dir, self.annotations[idx])

        # Open the image and convert to tensor
        image = Image.open(img).convert('RGB')
        width, height = image.size
        image = tv_tensors.Image(image)

        # Parse the xml annotation
        tree = ET.parse(anno)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            label = obj.find('name').text
            label_id = self.class_to_labels[label]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        image_id = torch.tensor([idx])

        # To tv tensors
        boxes = tv_tensors.BoundingBoxes(
            torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            format="XYXY",
            canvas_size=(height, width)
        )
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Wrap it as Pytorch requires it
        target = {
            "boxes": boxes,
            "labels": labels, 
            "image_id": image_id,
            "iscrowd": torch.zeros((len(labels), ), dtype=torch.int64) 
        } 

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        # Ensure float32 in [0, 1] regardless of whether a transform did it
        if image.dtype == torch.uint8:
            image = T.functional.to_dtype(image, torch.float32, scale=True)
       
         # Area calculation after transforms- required by some architectures
        bxxs = target["boxes"]
        target["area"] = (bxxs[:, 2] - bxxs[:, 0]) * (bxxs[: , 3] - bxxs[:, 1])

        return image, target
        
def create_voc_dataset(images_path: 'str', annotations_path: 'str', class_to_labels: 'dict', transforms=None):
    dataset = ObjectDetectionDataset(
        images_dir=images_path,
        annotations_dir=annotations_path,
        transforms=transforms, 
        class_to_labels=class_to_labels
    )
    return dataset

if __name__=="__main__":
    # Uses the object detection environment variables to create a sample dataset
    from dotenv import load_dotenv
    import ast
    env_file = ".env.object_detection"
    if not os.path.exists(env_file):
        print(f"Object detection env file does not exists: {env_file}")

    load_dotenv(dotenv_path=env_file)
    imgs_path = os.getenv("OBJ_DETECTION_TRAIN_IMAGES_PATH")
    annos_path = os.getenv("OBJ_DETECTION_TRAIN_ANNOTATIONS_PATH")
    class_to_labels = ast.literal_eval(os.getenv("OBJ_DETECTION_CLASS_LABELS"))
    num_samples= int(os.getenv("OBJ_DETECTION_NUM_SAMPLES"))
   
    img_transforms = T.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=(0.8, 1.4), contrast=(0.8, 1.5), saturation=0.1),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    ds = create_voc_dataset(imgs_path, annos_path, class_to_labels, transforms=img_transforms)
    print(f"Total image data: {len(ds)}")


