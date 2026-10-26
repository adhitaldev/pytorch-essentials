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
                transforms=None,
                idx_to_class=None,
                idx_to_keypts=None, 
                input_size=(IN_H, IN_W),
                sigma=2):

        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.idx_to_class = idx_to_class
        self.idx_to_keypts = idx_to_keypts
        self.num_keypts = len(self.idx_to_keypts) # This is the total number of keypoints to learn about
        self.target_height, self.target_width = input_size
        self.sigma = 2

        # HRnet stage 4 output spatial size is downsampled by a factor of 4
        self.downsample_factor = 4
        self.heatmap_height = self.target_height // downsample_factor
        self.heatmap_width = self.target_width // downsample_factor

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
        print(f"Class labels: {self.idx_to_class}")
        print(f"Keypoint labels: {self.idx_to_keypts}")
        print(f"Target Height: {self.target_height} Target Width: {self.target_width}")
        print(f"Heatmap Sigma: {self.sigma} Height: {self.heatmap_height} Width: {self.heatmap_width}")
        print(f"Transforms:\n{self.transforms}")

        # sort out the files
        self.images = set()
        self.annotations = set()
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(img_extensions):
                self.images.add(file)

        print(f"Total Images: {len(self.images)}")
        print(f"Total Annotations: {len(self.annotations_dict)}")
      
        self.images = sorted(list(self.images))
        self.annotations = sorted(list(self.annotations_dict))

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

        for k_idx, (x, y, viz) in enumerate(scaled_keypoints):
            # Visibility (0 = not labeled, 1 = labeled but occluded, 2 = visible)
            if vis == 0 or vis == 1 or x < 0 or y < 0:
                continue
            hm_x = x / self.downsample_factor
            hm_y = x / self.downsample_factor

            tmp_size = self.sigma * 3
            ul = [int(hm_x - tmp_size), int(hm_y - tmp_size)]
            br = [int(hm_x + tmp_size + 1), int(hm_y + tmp_size + 1)]

            # Check bounds on heatmap canvas
            if (
                ul[0] >= self.heatmap_width
                or ul[1] >= self.heatmap_height
                or br[0] < 0
                or br[1] < 0
            ):
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


    def __get__item(self, idx):
        img_name = self.images[idx]

        # Get the annotation value from the image name key
        item = self.annotations[img_name]

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

        raw_pts = item["keypoints"]
        scaled_kpts = []
        for k_idx in range(self.num_keypoints):
            # For each key point it comes in this order (x, y, visibility)
            x = raw_kpts[k_idx * 3]
            y = raw_kpts[k_idx * 3 + 1]
            viz = raw_ktps[k_idx * 3 + 2]
            scaled_kpts.append([x * scale_x, y * scale_y, viz])
    
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
