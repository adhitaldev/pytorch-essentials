"""
Example of script that replaces a modular block of a pre-trained model and use it
for transfer learning. In this case we use mobilenet_v3_small which was trained on the ImageNet dataset.
"""
import torch
import tqdm
import torch.nn as nn
import torchvision.datasets
import torchmetrics
import torchvision.models as vmodels
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import os
load_dotenv()
data_dir = os.getenv("DATA_PATH")

 # Mean and Std for the CIFAR-100 dataset
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

def get_model():
    model = vmodels.mobilenet_v3_small(weights='IMAGENET1K_V1')
    return model

def modify_model(model):
    # Freeze the params for the feature block
    # This is the central theme - the feature block is frozen
    # but the entire classifier block is re-trained for the new task.
    for params in model.features.parameters():
        params.requires_grad = False
    # Modify the last classifer head
    last_classifier_layer = model.classifier[-1]
    print(f"Last classifier tail: {last_classifier_layer}")

    # Modify the last classifier layer
    in_features = last_classifier_layer.in_features
    num_classes = 10
    new_classifier = nn.Linear(in_features = in_features, out_features = num_classes)
    model.classifier.pop(-1)
    model.classifier.append(new_classifier)
    print(f"New classifier tail: {model.classifier[-1]} ")
    return model

def fine_tune_head_top_layers(model):
    """
    Fine tunes the MobileNetV3model ever further by unfreezing the top layers of the 
    feature block backbone and retraining them, essentially this 'features' block: 
      (12): Conv2dNormActivation(
            (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): Hardswish()
            )
    """
    fine_tuned_model = model
    # Unfreeze the last block of the 'features' section
    print(f"Block to fine tune: {fine_tuned_model.features[12]}")
    for params in fine_tuned_model.features[12].parameters():
        params.requires_grad = True
    # Quick comparison test to make sure the params are frozen or not
    # This should be frozen
    print(f"Params in features[0] frozen: {not fine_tuned_model.features[0][0].weight.requires_grad}")
    # This should not be frozen
    print(f"Params in features[12] frozen: {not fine_tuned_model.features[12][0].weight.requires_grad}")
    # Make sure the classifier head is still not frozen and trainable
    print(f"Params in classifier frozen: {not fine_tuned_model.classifier[-1].weight.requires_grad}")
    
    return fine_tuned_model

def prepare_data(batch_size=64):
    """
    Prepares the data
    """
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

    # Datasets
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)

    label_classes = {idx:val for idx, val in enumerate(train_ds.classes)}
    print(label_classes)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_classes

def start_training_loop(model, train_loader, val_loader, loss_func, optimizer, num_epochs, device, num_classes):
    model = model.to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0
        train_tqdm = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for images,labels in train_tqdm:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            # Show current average loss
            train_tqdm.set_postfix({"Average loss": running_train_loss / (train_tqdm.n + 1)})
    
        model.eval()
        val_tqdm = tqdm.tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")
        with torch.no_grad():
            for images, labels in val_tqdm:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, 1)
                accuracy_metric.update(predicted, labels)

    
    print(" ---- Training Complete ---- ")
    final_accuracy = accuracy_metric.compute()
    print(f"Final Validation Accuracy: {final_accuracy: .4f}")
    return model

if __name__ == "__main__":
    model = get_model()
    print(f"Mobilenet architecture: \n{model}")
    # Modify the classifier head
    new_model = modify_model(model)

    # Fine tune the feature layer
    fine_tuned_model = fine_tune_head_top_layers(new_model)

    train_loader, test_loader, labels = prepare_data()
    # print(f"Data Preparation Complete")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current Device:{device}")
    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    # Only optimize the params that require gradients for mobilenet_model
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, fine_tuned_model.parameters()), lr=1e-5)
    num_epochs = 1
    num_classes = 10
    trained_model = start_training_loop (fine_tuned_model, train_loader, test_loader, loss_func, optimizer, num_epochs, device, num_classes)

    """
    Mobilenet's architecture looks like this:

    MobileNetV3(
        (features): Sequential(
            (0): Conv2dNormActivation(
            (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): Hardswish()
            )
            (1): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
                (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
                )
                (1): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(16, 8, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(8, 16, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (2): Conv2dNormActivation(
                (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (2): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
                (1): BatchNorm2d(72, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
                )
                (2): Conv2dNormActivation(
                (0): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (3): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
                (1): BatchNorm2d(88, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU(inplace=True)
                )
                (2): Conv2dNormActivation(
                (0): Conv2d(88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(24, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (4): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
                (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(24, 96, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (5): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
                (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (6): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
                (1): BatchNorm2d(240, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(64, 240, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(40, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (7): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
                (1): BatchNorm2d(120, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (8): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
                (1): BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(40, 144, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(48, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (9): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
                (1): BatchNorm2d(288, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(288, 72, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(72, 288, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (10): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
                (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (11): InvertedResidual(
            (block): Sequential(
                (0): Conv2dNormActivation(
                (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (1): Conv2dNormActivation(
                (0): Conv2d(576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
                (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                (2): Hardswish()
                )
                (2): SqueezeExcitation(
                (avgpool): AdaptiveAvgPool2d(output_size=1)
                (fc1): Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
                (fc2): Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))
                (activation): ReLU()
                (scale_activation): Hardsigmoid()
                )
                (3): Conv2dNormActivation(
                (0): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
                (1): BatchNorm2d(96, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
                )
            )
            )
            (12): Conv2dNormActivation(
            (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(576, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): Hardswish()
            )
        )
        (avgpool): AdaptiveAvgPool2d(output_size=1)
        (classifier): Sequential(
            (0): Linear(in_features=576, out_features=1024, bias=True)
            (1): Hardswish()
            (2): Dropout(p=0.2, inplace=True)
            (3): Linear(in_features=1024, out_features=1000, bias=True)
        )
        )
    """