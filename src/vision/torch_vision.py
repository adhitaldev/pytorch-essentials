import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor



## Basics of working with Pytorch vision
def run_example():
    print("-- Starting Torch Vision Example ---")
    # Download training data (food data in this case)
    training_Data = datasets.Food101(
        root="data",
        split="train",
        download=True,
        transform=ToTensor()
    )
    
    # Download test data
    test_data = datasets.Food101(
        root="data",
        split="test",
        download=True,
        transform=ToTensor
    )
    # Pass the dataset as an arg to dataloader which will wrap an iterable over out dataset
    batchsize = 64
    train_dataloader = DataLoader(training_Data, batch_size=batchsize)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")


if __name__ == "__main__":
    run_example()