"""
Defines a generic dataset that applies to key point detection
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import tv_tensors, transforms

img_extensions =('.jpg', 'jpeg', '.png', 'jpeg')
anno_extensions = '.json'
IN_H, IN_W = 256, 192 # Standard HRnet input dimension
hrnet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
hrnet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HRnetKeypointDetectionDataset(Dataset):
    """
    Generic dataset class for HRNet keypoint detection tasks. Takes in an image
    and keypoint annotations including class_label, keypoints labels, 
    transformations and then creates a dataset out of it.
    Args:
    images_dir: folder with images
    annotations_dir: folder with annotations
    transforms: optional
    class_to_idx: class to # labels to idx for class (example: {'dog': 0})
    keypts_to_idx: keypoints to idx of keypoints (example: {'nose': 0}), 
    input_size: expected input for HRnet (default is 256 x 192)
    sigma: Gaussian radius for groudth truth heatmap generation
    """
    def __init__(self, 
                images_dir, 
                annotations_dir,
                transforms=None,
                class_to_idx=None,
                keypts_to_labels=None, 
                input_size=(IN_H, IN_W),
                sigma=2):

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.class_to_idx = class_to_idx
        self.keypts_to_labels = keypts_to_labels
        self.target_height, self.target_width = input_size
        self.sigma = 2

        # HRnet stage 4 output spatial size is downsampled by a factor of 4
        self.heatmap_height = self.target_height // 4
        self.heatmap_width = self.target_width // 4

        # Transforms
        if transforms:
            self.transform = transforms
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=hrnet_mean, std= hrnet_std)
            ])

        print(f"\n--- Creating keypoints detection dataset ---")
        print(f"Image dir: {self.images_dir}")
        print(f"Annotation dir: {self.annotations_dir}")
        print(f"Transforms: {self.transforms}")
        print(f"Class labels: {class_to_labels}")
        print(f"Keypoint labels: {keypts_to_labels}")
        print(f"Target Height: {self.target_height} Target Width: {self.target_width}")
        print(f"Heatmap Sigma: {sigma} Height: {heatmap_height} Width: {heatmap_width}")
        print(f"Transforms:\n{self.transforms}")

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
            img_with_json = img + anno_extensions
            if img_with_json not in self.annotations:
                missing_annotations.append(img_with_json)
        
        print(f"Total Images: {len(self.images)}")
        print(f"Total Annotations: {len(self.annotations)}")
        print (f"Missing annotations for: {missing_annotations}")
      
        self.images = sorted(list(self.images))
        self.annotations = sorted(list(self.annotations))

    def __len__(self):
        return len(self.images)

    def __get__item(self, idx):
        img = os.path.join(self.images_dir, self.images[idx])
        anno = os.path.join(self.annotations_dir, self.annotations[idx])

        # Open the image and convert to tensor
        image = Image.open(img).convert('RGB')
        orig_w, orig_h = image.size
        image = tv_tensors.Image(image)
    

def create_keypt_dataset(images_path: 'str', 
                         annotations_path: 'str',
                         keypt_labels: 'dict',
                         class_labels: 'dict',
                         transforms=None):
    """
    Creates a keypoint dataset with the given params and returns them.
    """
    return None
