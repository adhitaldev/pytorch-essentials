"""
A simple convolutional neural network (CNN) in PyTorch to 
classify nature images from the CIFAR-100 dataset or 32x32 
color images for 100 different classes.
"""
import os
import torch
import torchvision
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from utils.image_visualizer import visualize_tensor_image
from dnn.mnist.mnist_visualizer import display_image

load_dotenv()
data_dir = os.getenv("DATA_PATH")

 # Mean and Std for the CIFAR-100 dataset
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def prepare_data(batch_size=64):
    """
    Gets the CIFAR10 dataset and prepares the dataloaders
    for train and test.
    """

    # Transformation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    # Get the dataset
    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transform)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    print(f"Classes:\n{train_dataset.classes}")
    label_classes = {idx: val for idx, val in  enumerate(train_dataset.classes)}
    print(f"Label and classes:\n{label_classes}")

    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, label_classes


class CifarCNN(nn.Module):
    """
    Uses conv filter and maxpooling layers for feature extraction
    from images.
    """
    def __init__(self, num_classes):
        super(CifarCNN, self).__init__()
        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Conv layer 2 - after maxpool with kernel_size 3, img size now 16 x 16
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #Conv layer3 - ...img size is now 8 x 8
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # ... img size becomes 4 x 4

        # Flatten the layers
        self.flatten = nn.Flatten()

        # Fully connected dense layer
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        # Dropout is generally implemented before the output layer
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
    
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def training_loop(model, train_loader, val_loader, loss_func, optimizer, num_epochs, device):
    """
    Starts the training and evaluation loop
    """
    model = model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        epoch_loss = 0
        batch = 0
        for images, labels in train_loader:
            # Move inputs and outputs to the same device as the model
            images = images.to(device)
            labels = labels.to(device)
            # Clear accumulated gradients
            optimizer.zero_grad()
            # Get the predictions
            output = model(images)
            # Calculate the loss - output and labels in this order
            loss = loss_func(output, labels)
            # Backprop to compute graadients
            loss.backward()
            # Update the model params
            optimizer.step()
            # Accumulate the training loss for the batch
            running_train_loss += loss.item() * images.size(0)
            #print(f"Training Batch# {batch + 1}:{len(images)} images")
            batch += 1
        
        # Average loss over entire dataset
        epoch_loss += running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Run evaluation on the epoch
        model.eval()
        running_val_loss = 0
        correct = 0
        total = 0
        epoch_val_loss = 0
        batch = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # Validation loss for the batch
                val_loss = loss_func(outputs, labels)
                # Total validation loss 
                running_val_loss += val_loss.item() * images.size(0)
                # Get the predicted class labels
                _, predicted = torch.max(outputs, 1)
                # All the labels
                total += labels.size(0)
                # Correct ones are where the predicted matches the labels
                correct += (predicted == labels).sum().item()
                #print(f"Evaluation Batch# {batch + 1}: {len(images)} images")
                batch += 1
        
        # Avg validation loss for the epoch
        epoch_val_loss = running_val_loss / (len(val_loader.dataset))
        val_losses.append(epoch_loss)

        epoch_accuracy = (correct / total) * 100.
        val_accuracies.append(epoch_accuracy)
        print(f"Epoch {(epoch + 1)}/{num_epochs} Train Loss: {epoch_val_loss} Validation Loss: {epoch_val_loss} Validation Accuracy: {epoch_accuracy}")

    print("------ TRAINING COMPLETE -------")
    metrics = [train_losses, val_losses, val_accuracies]
    return model, metrics

if __name__== "__main__":
    print("------ CIFAR 100 Conv. Classifier ------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current Device: {device}")

    train_loader, val_loader, label_classes = prepare_data(64)
    print(f"Total Train Batches: {len(train_loader)} Total Test Batches: {len(val_loader)}")
    print(f"Label Classes: {label_classes} Total Classes: {len(label_classes)}")
    
    # Sample display of an image
    print(f"Displaying sample data from training set")
    for images, labels in train_loader:
        pick_idx = 15
        img = images[pick_idx]
        lbl = labels[pick_idx]
        obj_class = label_classes[lbl.item()]
        visualize_tensor_image(img, mean=cifar100_mean, std=cifar100_std, label=f"{pick_idx} - {obj_class}")
        break

    # Model and training
    cnn_model = CifarCNN(len(label_classes))
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    num_epochs = 10
    print(f"Training and Evaluation loop started for {num_epochs} epochs...Please wait for updates")
    training_loop(cnn_model, train_loader, val_loader, loss_func, optimizer, num_epochs, device)



