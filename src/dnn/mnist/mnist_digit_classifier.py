"""
Model for a image digit classifier based on the classic MNIST dataset
of grayscale digit images. It uses the Torch Vision library to get the
dataset, train a model and then run evaluation on it

MNIST Dataset in Torch:
https://www.tensorflow.org/datasets/catalog/mnist
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .mnist_visualizer import display_image, plot_metrics

# Mean and standard deviations for the entire MNIST dataset
mnist_mean = 0.1307
mnist_std = 0.3801

# Set up the transform
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mnist_mean,), (mnist_std,))
])

# Get MNIST dataset from Torch Vision
train_dataset = datasets.MNIST(
    root="../data", # Where data is stored
    train=True, # Get the training split of the data
    download=True
)

# Get a sample data from the dataset - for learning only. Not relevant to training
# and done before transformation.
image_pil, label = train_dataset[0]
print(f"Image type: {type(image_pil)}")
print(f"Image dims: {image_pil.size}")
print(f"Label type: {type(label)}")
print(f"Label value: {label}")
display_image(image_pil, label, "MNIST Digit", show_values=True)
# Apply the transformation
train_dataset = datasets.MNIST(
    root="../data", # Where data is stored
    train=True, # Get the training split of the data
    download=True,
    transform=mnist_transform
)

# Image has been transformed now
image_tensor, label = train_dataset[0]
print(f"Transformed Image type: {type(image_tensor)}")
print(f"Transformed Image dims: {image_tensor.shape}")
print(f"Transformed Label type: {type(label)}")
print(f"Transformed Label value: {label}")
display_image(image_tensor, label, "MNIST Digit (Tensor)", show_values=True)

test_dataset = datasets.MNIST(
    root="../data",
    train=False, # Get tehe evaluation split of the data
    download=True,
    transform=mnist_transform
)

# Batch sizes
train_batch = 64
test_batch = 1000

# Get the MNIST dataset from Torch. The test data batch size is larger
# than train batch size because there is no gradient calculation. Shuffle is enabled for
# training to avoid model converging on a bias too soon (for example if the 
# first 1000 images are of digit '0', it may unintentionally learn that the
# early batches are zeros). Shuffle enables variety. Order doesn't matter for testing and 
# hence shuffle is set to false.
train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=False)

# Model trainer for MNIST
class MNISTClassifier(nn.Module):
    """
    Each grayscale image in the MNIST dataset is that of a handwritten digit
    of size 28x28. So the total 28x28 = 784 pixels are flattened first into
    a single vector and fed into the network. 
    """
    def __init__(self):
        super().__init__()
        self.img_width = 28
        self.img_height = 28
        self.input_size = self.img_width * self.img_height
        self.output_size = 10 # for the 10 digits (0 - 9)
        self.layer_size = 128 # Number of neurons in the hidden layer

        # [1, 28, 28] tensor represent channel, width, and height for each image.
        # For batch size of 64, the incoming tensor size looks like [64, 1, 28, 28]
        # Since linear layers expect flat vectors, this tensor is flattened to a 
        # tensor of [64, 784].
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.layer_size),
            nn.ReLU(),
            nn.Linear(self.layer_size, self.output_size)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Setting up the device, loss func, optimizer, and the training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Adaptive optimizer
loss_func = nn.CrossEntropyLoss() # Great for classification task

print(f"Starting MNIST Classifier Training on {device}")

# Train Function
def train_epoch(model, train_loader, loss_function, optimizer, device):
    """
    Starts a training for an epoch with all the batches.
    """
    model.train() # Puts the model in training mode
    epoch_loss = 0 # Loss for the entire epoch when the entire dataset feeds in
    running_loss = 0.0 #Acccumulates the loss value
    correct_predictions = 0 # No of correct predictions
    total_predictions = 0 # Total number of predictions
    total = 0 # Samples seen so far
    total_batches = len(train_loader)
    progress_track_interval = 100

    # Loop over all the batches - each iteration is called a 'step'
    for batch_idx,(inputs, targets) in enumerate(train_loader):
        # Move the inputs and targets to the current decice
        inputs = inputs.to(device)
        targets = targets.to(device)
        # Clear the gradients
        optimizer.zero_grad()
        # Get prediction and calculate loss
        output = model(inputs)
        loss = loss_function(output, targets)
        # Backpropogate - evaluate the gradient
        loss.backward()
        # Update the weights
        optimizer.step()

        # Loss for tracking and reporting
        loss_value = loss.item()
        epoch_loss += loss_value
        running_loss += loss_value

        # Accuracy metrics for the current batch
        # output is tensor with shape(batch_size, num_classes). The following line
        # gets the index with the higest value.
        _, predicted_indices = output.max(1)
        batch_size = targets.size(0) # How many images are in this batch
        total_predictions += batch_size
        correct_predictions += predicted_indices.eq(targets).sum().item()
 
        # Print update every 100 batches
        if (batch_idx + 1) % progress_track_interval == 0 and batch_idx  > 0:
            avg_loss = running_loss / progress_track_interval
            accuracy = 100. * (correct_predictions / total_predictions)
            print(f"Step {batch_idx + 1}/{total_batches} Images: {(batch_idx + 1) * batch_size} Loss: {avg_loss:.3f} | Accuracy: {accuracy:.2f}%")
            running_loss = 0.0
            # Reset the trackers for next reporting interval
            correct_predictions = 0
            total_predictions = 0

    # Average loss for entire epoch
    avg_epoch_loss = epoch_loss / total_batches
    return model, avg_epoch_loss

# Evaluate
def evaluate (model, test_loader, device):
    """
    Evaluates the correctness of prediction on the given
    model, test loader running on the device.
    """
    model.eval() # Switch the model into evaluation mode
    correct = 0 # Number of correct predictions
    total = 0 # Number of total predictions
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # Index of the highest value in output tensor which represents the predicted class
            _, predicted = outputs.max(1) 
            # Adds the the batch size i.e. current number of images to total predictions
            total += targets.size(0) 
            # Compare the predicted indices with the target values
            correct_predictions = predicted.eq(targets)
            # Accumulate the correct predictions in the current batch
            correct += correct_predictions.sum().item()
    acc_percent = 100. * (correct / total)
    return acc_percent

# Training loop and evaluation
num_epoch = 10 # Number of passes through the entire dataset
train_loss = []
test_acc = []
for epoch in range(num_epoch):
    print(f"\nTraining/Testing Epoch {epoch + 1}")
    trained_model, loss = train_epoch(model, train_loader, loss_func, optimizer, device)
    train_loss.append(round(loss, 4))
    accuracy = evaluate(model, test_loader, device)
    test_acc.append(round(accuracy, 4))

# Results
print(f"----- TRAINING/EVALUATION SUMMARY")
print(f"Total Epochs: {num_epoch}")
print(f"Train Loss : {train_loss}")
print(f"Test Accuracy: {test_acc}")
plot_metrics(train_loss, test_acc)
