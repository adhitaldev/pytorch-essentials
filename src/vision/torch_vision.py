import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor    
from .simple_nn import SimpleNN, get_model

########## Torch Data handling #####################################################
# Has Two major primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset
# Dataset stores the samples/lavels, DataLoader wraps an iterable around Dataset
##############################################################################

# Basics of working with Pytorch vision - building a data and then using a
# a model to run training/test on it
def run_example():
    print("-- Starting Torch Vision Example ---")
    # Download training data FashionMNIST
    # Available PyTorch Datasets - https://docs.pytorch.org/vision/stable/datasets.html
    training_Data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    # Download test data
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    # Pass the dataset as an arg to dataloader which will wrap an iterable over out dataset
    # DataLoader supports automatchiing batching, sampling, shuffling, and multiprocess data loading
    batchsize = 64
    train_dataloader = DataLoader(training_Data, batch_size=batchsize)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")




if __name__ == "__main__":
    run_example()