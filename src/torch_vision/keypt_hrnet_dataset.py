"""
Defines a generic dataset that applies to key point detection
"""
import os
import ast
import json
import torch
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from torchvision import tv_tensors, transforms
from vision_utils import visualize_tensor_image, annotate_key_points_on_image

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

    NOTE:
    This is a dataset for top-down pipeline i.e it assumes that there is a single object
    per image with keypoints. If the original image had multiple classes, they need to be
    cropped to single images and passed to this dataset class.

    Args:
    images_dir: folder with images
    annotations_dict: Dictionary that already has the mapping of image to class and keypoints. 
                      Each image_key needs to have these value items:  
                        a) 'keypoints' which is a list of single flatteened list keypoints in this order:
                                (x0, y0, visibility0,x1, y1, visibility1, ...)
                        b) 'class_id' the idx for the class of the bounding box that provides the keypoints
    transforms: optional
    idx_to_class:  Idx  to class names (example: {0: 'dog'})
    idx_to_keypts: Idx to keypoint names (example: {0: 'nose'}), 
    input_size: expected input for HRnet (default is 256 x 192)
    sigma: Gaussian radius for groudth truth heatmap generation
    """
    def __init__(self, 
                images_dir, 
                annotations_dict,
                transform=None,
                idx_to_class=None,
                idx_to_keypts=None, 
                input_size=(IN_H, IN_W),
                sigma=2):

        self.images_dir = images_dir
        self.annotations_dict = annotations_dict
        self.idx_to_class = idx_to_class
        self.idx_to_keypts = idx_to_keypts
        self.num_keypts = len(self.idx_to_keypts) # This is the total number of keypoints to learn about
        self.target_height, self.target_width = input_size
        self.sigma = 2

        # HRnet stage 4 output spatial size is downsampled by a factor of 4
        self.downsample_factor = 4
        self.heatmap_height = self.target_height // self.downsample_factor
        self.heatmap_width = self.target_width // self.downsample_factor

        # Transforms
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=hrnet_mean, std= hrnet_std)
            ])

        print(f"\n--- Creating {HRnetKeypointDetectionDataset.__name__} ---")
        print(f"Image dir: {self.images_dir}")
        print(f"Length annotation dict: {len(self.annotations_dict)}")
        print(f"Transforms: {self.transform}")
        print(f"Class labels: {self.idx_to_class}")
        print(f"Keypoint labels: {self.idx_to_keypts}")
        print(f"Total keypoints: {self.num_keypts}")
        print(f"Target Height: {self.target_height} Target Width: {self.target_width}")
        print(f"Heatmap Sigma: {self.sigma} Height: {self.heatmap_height} Width: {self.heatmap_width}")

        # sort out the files
        self.images = set()
        self.annotations = set()
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(img_extensions):
                self.images.add(file)

        print(f"Total Images: {len(self.images)}")
        print(f"Total Annotations: {len(self.annotations_dict)}")
      
        self.images = sorted(list(self.images))

    def __len__(self):
        return len(self.images)

    def _generate_heatmaps(self, scaled_keypoints):
        """
        Renders 2D gaussians for valid N keypoints
        Returns:
            Heatmaps:  (N, height_heatmap, width_heatmap) float32 tensor
            Mask: (N, 1) float32 tensor indicating which point to compute loss over
        """
        heatmaps = np.zeros((self.num_keypts, self.heatmap_height, self.heatmap_width), dtype=np.float32)

        # Loss weight mask (1.0 = compute loss, 0.0=ignore)
        mask = np.zeros((self.num_keypts, 1), dtype=np.float32)

        for k_idx, (x, y, vis) in enumerate(scaled_keypoints):
            # Visibility (0 = not labeled 2 = visible). Keep viz 1 to predict through occlusions
            if vis == 0  or x < 0 or y < 0:
                continue
            hm_x = x / self.downsample_factor
            hm_y = y / self.downsample_factor

            tmp_size = self.sigma * 3
            ul = [int(hm_x - tmp_size), int(hm_y - tmp_size)]
            br = [int(hm_x + tmp_size + 1), int(hm_y + tmp_size + 1)]

            # Check bounds on heatmap canvas
            if (
                ul[0] >= self.heatmap_width
                or ul[1] >= self.heatmap_height
                or br[0] < 0
                or br[1] < 0):
                continue

            # Create 2D Gaussian patch
            size = 2 * tmp_size + 1
            x_grid = np.arange(0, size, 1, np.float32)
            y_grid = x_grid[:, np.newaxis]
            x0 = y0 = size // 2
            gaussian = np.exp(
                -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * self.sigma**2)
            )

            # Valid region on heatmap canvas
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_height) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_width)
            img_y = max(0, ul[1]), min(br[1], self.heatmap_height)

            if g_x[1] > g_x[0] and g_y[1] > g_y[0]:
                heatmaps[k_idx, img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                    heatmaps[k_idx, img_y[0] : img_y[1], img_x[0] : img_x[1]],
                    gaussian[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                )
                mask[k_idx] = 1.0

        return torch.from_numpy(heatmaps), torch.from_numpy(mask)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # Get the annotation value from the image name key
        item = self.annotations_dict[img_name]

        # Read the image
        img = os.path.join(self.images_dir, self.images[idx])

        # Open the image and convert to tensor, resize to target sizez
        # and apply transformation
        image = Image.open(img).convert('RGB')
        orig_w, orig_h = image.size
        image_resized = image.resize((self.target_width, self.target_height), Image.BILINEAR)
        image_tensor = self.transform(image_resized)

        # Rescale the keypoints coordinates to match resized dimenstions
        scale_x = self.target_width / orig_w
        scale_y = self.target_height / orig_h

        raw_kpts = item["keypoints"]
        scaled_kpts = []
        for k_idx in range(self.num_keypts):
            # For each key point it comes in this order (x, y, visibility). It's already normmalized
            x_norm = raw_kpts[k_idx * 3]
            y_norm = raw_kpts[k_idx * 3 + 1]
            viz = raw_kpts[k_idx * 3 + 2]
            if viz > 0:
                x_pixel = x_norm * self.target_width
                y_pixel = y_norm * self.target_height
            else:
                x_pixel, y_pixel = 0.0, 0.0
            scaled_kpts.append([x_pixel, y_pixel, viz])
    
        # Generate the heatmaps and visibility mask
        target_heatmaps, target_mask = self._generate_heatmaps(scaled_kpts)

        # Extract class idx
        class_idx = item["class_id"]
        class_id = torch.tensor(class_idx, dtype=torch.long)
        return image_tensor, target_heatmaps, target_mask, class_id

def create_keypt_hrnet_dataset(images_dir, 
                annotations_dict: 'dict',
                transforms=None,
                idx_to_class=None,
                idx_to_keypts=None, 
                input_size=(IN_H, IN_W),
                sigma=2):
    """
    Creates a keypoint dataset with the given params and returns them.
    """
    if not idx_to_class:
        idx_to_class = {0: 'UnknownClass'}
    if not idx_to_keypts:
        idx_to_keypts = {0: 'UnknownKeypt'}
    return HRnetKeypointDetectionDataset(images_dir, annotations_dict, transforms, idx_to_class, idx_to_keypts, input_size, sigma)

def get_annotation_dict_from_dog_pose_images(annotation_dir):
    """
    Reads through the annotation jsons in the annotation folder
    and returns a annotation dictionary
    {'img1.jpg': {'keypoints':[kp0_x1, kp0_y1, kp0_viz, kp1_x1, kp2_y1, kp2_viz], 'class_id': 0}, 'img2.jpg': ...}
    """
    annotations_dict = {}
    for filename in sorted(os.listdir(annotation_dir)):
        if not filename.lower().endswith('.json'):
            continue

        json_path = Path(os.path.join(annotation_dir, filename))
        base_name = json_path.stem
        print(f"filename: {filename} basename {base_name}")
        with open(json_path, 'r') as f:
            payload = f.read()
            payload = ast.literal_eval(payload)

        if isinstance(payload, dict) and 'pose' in payload:
            pose = payload['pose']
        elif isinstance(payload, list):
            pose = payload
        else:
            pose = [payload]

        for keypoints in pose:
            kpts = keypoints[5: ] # Skip the first 5 values which are classid and 4 for bounding box
            #print(f"Keypoints : {kpts} type: {type(kpts)} total: {len(kpts)}")
            assert len(kpts) == 72 # dog pose dataset has 24 keypoints and each has x, y, and visibility
            keypts_class_map = {'keypoints':  kpts, 'class_id': 0 }
            annotations_dict[base_name] = keypts_class_map
    return annotations_dict

def get_idx_keypoints_dog_pose_dataset():
    """
    Returns the idx to keypoints of the dog pose dataset
    """
    idx_to_keypts = {
        0: 'front_left_paw', 
        1: 'front_left_knee',
        2: 'front_left_elbow',
        3: 'rear_left_paw',
        4: 'rear_left_knee',
        5: 'rear_left_elbow',
        6: 'front_right_paw',
        7: 'front_right_knee',
        8: 'front_right_elbow',
        9: 'rear_right_paw',
        10: 'rear_right_knee',
        11: 'rear_right_elbow',
        12: 'tail_start',
        13: 'tail_end',
        14: 'left_ear_base',
        15: 'right_ear_base',
        16: 'nose',
        17: 'chin',
        18: 'left_ear_tip',
        19: 'right_ear_tip',
        20: 'left_eye',
        21: 'right_eye',
        22: 'withers',
        23: 'throat'
        }
    return idx_to_keypts

def get_idx_class_dog_pose_dataset():
    """
    Returns the idx to class of the dog pose dataset
    """
    idx_to_class = {
        0: 'dog'
    }
    return idx_to_class

if __name__ == "__main__":
    env_file = f".env.keypts_detection"
    from dotenv import load_dotenv
    load_dotenv(env_file)
    img_dir = os.getenv("KEYPT_DETECTION_TRAIN_IMAGES_PATH")
    anno_dir = os.getenv("KEYPT_DETECTION_TRAIN_ANNOTATIONS_PATH")

    # For the dog pose dataset from Ultralytics, just one class, and single class per image.
    idx_to_class = get_idx_class_dog_pose_dataset()
    idx_to_keypts = get_idx_keypoints_dog_pose_dataset()

    # Read the annotations from the annotations folder and create an annotation dictionary where each key, value
    # looks like:
    annotations_dict = get_annotation_dict_from_dog_pose_images(anno_dir)
    print(f"Annotation dictionary for creating dataset:\n{annotations_dict}")
    dog_dataset = create_keypt_hrnet_dataset(
        img_dir,
        annotations_dict=annotations_dict,
        transforms=None,
        idx_to_class=idx_to_class,
        idx_to_keypts=idx_to_keypts,
        input_size=(IN_H, IN_W),
        sigma=2,
    )

    print(f"Length of dataset {len(dog_dataset)}")
    print(f"First item in dataset ....")
    img, target_heatmaps, target_mask, class_id = dog_dataset[0]
    print(f"target mask: {target_mask}")
    print(f"target_heapmaps: Shape: {target_heatmaps.shape} Value:{target_heatmaps}")
    visualize_tensor_image(img, label="test keypoints")

