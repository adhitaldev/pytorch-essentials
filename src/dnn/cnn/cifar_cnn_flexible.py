"""
Implementation of a flexible CNN on the CIFAR-10 dataset using PyTorch and Optuna for hyperparamter optimization.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from .cifar_cnn import prepare_data, training_loop, evaluate_with_torchmetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CiFarCNNFlex(nn.Module):
    """
    A flexible CNN model architecture for CIFAR-10 classification.
    """

    def __init__(self, num_layers, n_filters, start_in_channel, kernel_sizes, dropout_rate, fc_size, num_classes=10):
        super(CiFarCNNFlex, self).__init__()

        #Error checks here for the input params
        samesize =  len(n_filters) == num_layers == len(kernel_sizes)
        if not samesize:
            raise ValueError("The number of layers, filters, and kernel sizes must be the same.") 

        # Build the convolutional blocks
        in_chnl = start_in_channel
        out_chnl = n_filters[0]
        blocks = [] 

        for idx in range(num_layers):
            out_chnl = n_filters[idx] # Out channel for the current layer
            kernel = kernel_sizes[idx] # Kernel size for the current layer
            padding = (kernel - 1) // 2 # Same spatial dimensions after convolution

            # Construct the current block
            curr = nn.Sequential(
                nn.Conv2d(in_channels=in_chnl, out_channels=out_chnl, kernel_size=kernel, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # Append the block to the architecture
            blocks.append(curr)

            # Make the output channel the input for the next block
            in_chnl = out_chnl
        
        # Combine all the blocks into a single feature extractor module
        self.features = nn.Sequential(*blocks)
        self.dropout_rate = dropout_rate
        self.fc_size = fc_size
        self.num_classes = num_classes
        self.classifier = None # To allow for dynamic construction of the classifier 


    def _create_classifier(self, flattened_size, device):
        """
        Dynamically creates the classifier based on the input size and number of classes.
        """
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size, self.fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_size, self.num_classes)
        ).to(device)

    def forward(self, x):
        """
        Defines the data flow through the NN
        """
        device = x.device

        x = self.features(x)
        flattened = torch.flatten(x, 1) # Don't flatten the batch dimension i.e. 0
        flattened_size = flattened.size(1)
        if self.classifier is None:
            self._create_classifier(flattened_size, device)
        x = self.classifier(flattened)
        return x


def objective(trial, device):
    """
    Objective function for hyperparameter optimizatoin using Optuna.
    For each trial, the function samples a set of hyperparameters, 
    contructs a model, adn trains it for a fixed num of epochs. It 
    then evaluates the performance on the validation set and returns
    the accuracy.Optuna uses tis accuracy for best hyperparamter combo.

    Args:
       trial: An Optuna 'Trial' object
       device: 'cuda' or 'cpu'
    Returns:
        validation accuracy of the trained model
    """
    # Using Optuna to suggest a range of number of layers between 1 and 3
    n_layers = trial.suggest_int("n_layers", 1, 3)
    # Get a list of filter suggestions for the number of layers
    n_filters = [trial.suggest_int(f"n_filters_{i}", 16, 128) for i in range(n_layers)]
    # Get a list of kernel sizes for the number of layers
    kernel_sizes = [trial.suggest_int(f"n_kernel_size{i}", 3, 5) for i in range(n_layers)]

    # Suggestion for dropouts
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 256)

    start_channel = 3 # 3 channels in rgb image dataset in CiFAR
    model = CiFarCNNFlex(n_layers, n_filters, start_channel, kernel_sizes, dropout_rate, fc_size, num_classes=10).to(device)

    # Initialize the classifier layer by passing a a dummy input because the model
    # creates the classifier dynamically, and plain model initialization does not 
    # create these parameters. Passing data through the m odel forces it to calculate
    # the flattened feature size and build the classifier layers. DO IT BEFORE defining
    # the optimizer so that the model.parameters() includes the weights, else the optimizer
    # will only use the feature extractor and leave the classifier untrained.
    # like (1, 3, 32, 32) which represents the dims of 1 image in the CiFAR dataset
    dummy_input = torch.rand(1, 3, 32, 32).to(device)
    model(dummy_input)
    # Some fixed training params
    learning_rate = 0.001
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Fixed loading parameters
    batch_size = 128
    train_loader, val_loader, label_classes = prepare_data(batch_size)

    # Epochs and training
    num_epochs = 2
    training_loop(model, train_loader, val_loader, loss_func, optimizer, num_epochs, device)

    # Evaluation 
    accuracy, precision, recall, f1 = evaluate_with_torchmetrics(model, val_loader, device, num_classes=10)

    return accuracy

    
def run_optuna_study():
    """
    Creates and runs an Optuna study to manage the hyperparamters
    optimization process.
    """
    study = optuna.create_study(direction='maximize') # maximize accuracy
    n_trials = 2 # Use more trials in practice
    study.optimize(lambda trial: objective(trial, device), n_trials=n_trials)
    return study


if __name__=="__main__":
    study = run_optuna_study()

    # Extract the dataframes with the resutls
    df = study.trials_dataframe()

    # Extract the best trials
    best_trial = study.best_trial

    print(f"Best trial: {best_trial.number}")
    print(f"Best value (accuracy): {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

    # Plotting the optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("CiFARCNN Optuna Hyperparameter Optimization History")
    plt.show()

    # Hyperparameter importance
    optuna.visualization.matplotlib.plot_param_importances(study)

    ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
        study, params = ['n_layers', 'n_filters_0', 'kernel_size_0', 'dropout_rate', 'fc_size']
    )

    fig = ax.figure
    fig.set_size_inches(12, 6, forward=True) # Updates the canvas
    fig.tight_layout()













        
             
