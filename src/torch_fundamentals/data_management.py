"""
For ML training, data comes in different shapes, sizes, and formats.
It's essential to pre-process the data for effective learning. This
script explores some of the fundamental techniques for data processing.
The examples are based on the Oxford 102 flowers dataset.
"""

"""
Oxford 102 flowers dataset specifics:
- 102 classes of flowers, 40-258 images/class
- 8189 images named image_00001.jpg to image_08189.jpg
- labels stored in imagelabels.mat file
- Large scale, pose, and light variations

"""
import os
import random
import dotenv
import scipy
import tarfile
import urllib.request
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image

dotenv.load_dotenv()
output = os.getenv("DATA_PATH","./")
data_dir = os.path.join(output, 'OxfordFlowers')

# Because the dataset is nonstandard, implement it as Dataset class
# Dataset class has three major functions: __init__, __len__, and __getitem__
# that need to be defined.
class OxfordFlowersData(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Defines where to find images and labels
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'jpg')
        self.transform = transform

        # Load the labels from the matlab file using scipy
        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))
        self.labels = labels_mat['labels'][0] - 1 # Because labels start from 1
        print(f"Total labels {len(self.labels)}")
        print(f"First 10 labels: {self.labels[:10]}")

    def __len__(self):
        """
        Returns the total number of the samples
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Given an index, returns the appropriate data and label.
        Handles error to avoid corrupt images, etc.
        """
        # Build the image file name
        img_name = f'image_{idx + 1:05d}.jpg' # Pads the image name to 5 digit width
        img_path = os.path.join(self.img_dir, img_name)
        # Load the image
        image = Image.open(img_path)
        # Get the label
        label = self.labels[idx]
        return image, label

# Progress hook for large downloads
def progress_hook(block_num, block_size, total_size):
  downloaded = block_num * block_size
  percent = int(downloaded / total_size * 100) if total_size > 0 else 0
  print(f"\rDownloading... {percent}%", end="") 
    
def download_data():
    """
    Downloads the raw data and extracts the images from the tgz file.
    """
    print("Downloading Oxford flowers dataset ...")
    image_tgz_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_mat_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    
    image_tgz_path = os.path.join(data_dir, "102flowers.tgz")
    labels_mat_path = os.path.join(data_dir, "imagelabels.mat")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        urllib.request.urlretrieve(image_tgz_url, image_tgz_path, reporthook=progress_hook)
        urllib.request.urlretrieve(labels_mat_url, labels_mat_path)

        # Extract the tgz file
        print("Extracting images ... ")
        with tarfile.open(image_tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
    else:
        print("Image data has already been downdloaded")
    

def prepare_data(raw_data_dir: str):
    """
    Uses the location of the raw data to prepare the dataset.
    """  
    print("Preparing dataset ...")  
    flower_ds = OxfordFlowersData(raw_data_dir)
    dataloader = DataLoader(flower_ds, batch_size=4, shuffle=True)
    print(f"Total Samples: {len(flower_ds)}")
    # Load a sample data 
    print(f"Loading sample data from index 1...")
    img, label = flower_ds[10]
    print(f"Image {img} label: {label}")
    # img.show()

    # Because the image sizes are different, a dataloader
    # will not work directly on this data as the input
    # is expected to be of the same size. Plus, Torch
    # expects tensors and not PIL images.
    try:
        dataloader = DataLoader(flower_ds, batch_size=4, shuffle=True)
        # Try to get a batch
        for images, labels in dataloader:
            print(f"Batch shape: {images.shape}")
            break
    except Exception as e:
        print(f"Error creating data : {e}")
    
    # Resizing an image may distort it if it's rectangular.
    # A standard practice for different sized images is to 
    # scale the images in one direction, and then center crop it
    # Recommended strategy is to check the transformation before
    # committing to it.
    transform = transforms.Compose([
        transforms.Resize(256), # Resizes shorter edge to 256
        transforms.CenterCrop(224) # Extract 224 x 224 center square
        ])
    
    # A quick verification of what this may do to an image
    # This step is crucial during actual image data preparation 
    resized_img = transforms.Resize(256)(img)
    print("After resize")
    resized_img.show()
    center_cropped_img = transforms.Resize(224)(resized_img)
    print("After center cropped")
    center_cropped_img.show()

    # Another crucial transformation step is to convert the image to tensor
    # Tensor also scales the value dividing each by 255. So each value will
    # fall between 0 and 1
    final_img = transforms.ToTensor()(center_cropped_img)
    print(f"After tensor conversion: {final_img} ")

    # The obtained tensor is generally normalized using mean and std. deviation
    # of the data. Some transforms only work on tensors. So the sequence of 
    # applying transforms for images can be thought of as 'before the tensor' and 
    # 'after the tensor', or before and after the tensor bridge. 
    # A final, combined transform function may look like this:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # The tensor bridge
        transforms.Normalize((1.0, ), (2.0,))
    ])

    #--------------- COMPLETE SETUP FROM DATASET CREATION TO SPLITS ----------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def create_data_splits(raw_data_dir: str):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # The tensor bridge
        transforms.Normalize(mean=mean, std=std) 
    ])
    flower_ds = OxfordFlowersData(raw_data_dir, transform=transform)

    # Splitting data into training, validation, and test sets. Generally 70%-15%-15%
    train_size = int(0.70 * len(flower_ds))
    val_size = int(0.15 * len(flower_ds))
    test_size = len(flower_ds) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(flower_ds, [train_size, val_size, test_size])
    print(f"""After split:\n 
        Total Training Images: {len(train_ds)}\n
        Total Validation Images: {len(val_ds)}\n
        Total Test Images: {len(test_ds)}""")
    
    # Use DataLoader to load the split datasets as batches with necessary shuffle
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

def augment_data(data_dir):
    """
    Illustrates the data augmentation techniques and usage.
    During actual setup, implement pipelines to visualize
    augmented data as much as possible.
    """
    train_transform = transforms.Compose([
        # Random augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1),
        # Standard preprocessing
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # The tensor bridge
        transforms.Normalize((1.0, ), (2.0,))
    ])
    flower_ds = OxfordFlowersData(data_dir, transform=train_transform)
    # Test random sample
    rand_indices = random.sample(range(1, len(flower_ds)), 5)
    for rand in rand_indices:
        img, lbl = flower_ds[rand]
        img.show()

if __name__ == "__main__":
    download_data()
    # prepare_data(data_dir)
    train_loader, val_loader, test_loader = create_data_splits(data_dir)
    print(f"""Total Train batches: {len(train_loader)}\n
            Total Val Batches: {len(val_loader)}\n
            Total Test Batches: {len(test_loader)}""")
    print("Applying random augmentations to train data")
    augment_data(data_dir)