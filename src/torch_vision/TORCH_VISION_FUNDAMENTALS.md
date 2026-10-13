## TORCH VISION FUNDAMENTALS

# TorchVision Utilities 
__decode_image__ Converts image files to tensors in the format(channels, height, width)

__make_grid__ Arranges a batch of images into a clean, single grid for visualization

__save_image__  Saves a tensor image back to standard file format

# Transforms
Common image transformations using __transforms__ are:

__ToTensor()__  Converts image to tensor in the format [channel, height, width]

__ToPILImage()__ Converts Torch tensor to PILImage in original format

__ReSize(size)__ Resizes the image. If one number is provided, the other one is picked to preserve proportion

__CenterCrop(size)__ Used to focus on center of the image. It crops from center.

__RandomResizeCrop(size)__ Randomly crops a portion of the image and resizes to the given size.

___RandomHorizontalFlip(p)__ Randomly flips the image horizontally with the given probability.

__ColorJitter(brightness, contrast, saturation)__ Randomly changes the brightness, contrast, and saturation parameters of the image.

__Normalize()__ Normalizes pixel values (z-score) from 0-255 to 0-1. Mean and stdv needed for his are either pre-computed manually or provided with the dataset

# Preprocessing pipeline 
__Compose__ function allows creation of a transform pipleline in the specified order, which can be used with the dataset.
```
base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.46], std=[0.22, 0.22, 0.223])
  ] 
)
```
Image augmentation adds variations to exisiting images to increase the data size. 
```
data_aug_transform = transforms.Compose([
    transforms.Resized(),
    transforms.HorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor() 
    transforms.Normalize(std=.., mean=...)
])
```

# Custom Transformation
Custom transformations may be helpful for creating agumentation that's not defined, for example: simulating a particular type of camera noise.
Define the custom transformation with a class with a __call__method. This will take a PIL image as input and return the modified image.

```
class SaltAndPepperNoise:
    """
    A custom transform to add salt and pepper noise to a PIL image.

    Args:
        salt_vs_pepper (float): The ratio of salt to pepper noise.
                                (e.g., 0.5 is an equal amount of each).
        amount (float): The total proportion of pixels to be affected by noise.
    """
    def __init__(self, salt_vs_pepper=0.5, amount=0.04):
        self.s_vs_p = salt_vs_pepper
        self.amount = amount

    def __call__(self, image):
        # Make a copy of the image
        output = np.copy(np.array(image))

        # Add Salt Noise
        num_salt = np.ceil(self.amount * image.size[0] * image.size[1] * self.s_vs_p)
        # Generate random coordinates for salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.size]
        # Set pixels to white
        output[coords[1], coords[0]] = 255  

        # Add Pepper Noise
        num_pepper = np.ceil(self.amount * image.size[0] * image.size[1] * (1.0 - self.s_vs_p))
        # Generate random coordinates for pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.size]
        # Set pixels to black
        output[coords[1], coords[0]] = 0

        # Convert the NumPy array back to a PIL image
        return Image.fromarray(output)

    def __repr__(self):
        return self.__class__.__name__ + f'(salt_vs_pepper={self.s_vs_p}, amount={self.amount})'


#This can be used as:
custom_transform = transforms.Compose([
    transforms.Resize(256), 
    SaltAndPepperNoise(), 
    ...
])
```

# Fake datasets
TorchVision provides mechanism to create fake datasets that can be used to debug the pipeline, test the fundamentals, etc.
```
fake_dataset = datasets.FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=fake_data_transform)
```

# Available models
TorchVision provides model architectures that can be used for training from scratch or using pre-trained weights. Some of these are _resnet50_, etc.
__Image Classification__ ResNET, VGG, AlexNet, MobileNETV3
__Image Segmentation__ FCN, DeepLabV3
__Object Detection__ Faster R-CNN, RetinaNet, SSD
__Video Classification__ R(2+1)D 18, MC3 18, Video MViT
These models can be used for either direct inference or for transfer learning (fine-tuning).
```
from torchvision import models as tv_models
resnet50_model = tv_models.resnet50(pretrained=True).eval()
```

# Transfer Learning and Fine Tuning
Pre-trained models can be used for transfer learning. Early layers of a layer detect generic shapes or patterns, and the later layers can be trained or tuned to specific tasks. It requires to work with either a block of layers or just the final layer in the original model architecture.

# Visualization utilities
Visualiation utilities are essential to see what the model is doing in various stages of the training/testing processes. Two of the major utilities available in TorchVision are: __draw_bounding_boxes__ and __draw_segmentation_masks__.
```
import torchvision.utils as vutils
image = decode_image("imgpath")
boxes = torch.tensor([[140, 30, 375, 315]], dtype=torch.float)
labels=["dog"] 
result = vutils.draw_bounding_boxes(image=image, boxes=boxes, labels=labels, colors=["red", "blue"], width=3)
```










