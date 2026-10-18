"""
Simple plotter for visualization of data
WIP
"""
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import numpy as np

def plot_training_progress(epoch, loss, model, inputs_norm, outputs_norm):
    """
    Plots the training progress of a model on normalized data,
    showing the current fit at each epoch.

    Args:
        epoch: The current training epoch number.
        loss: The loss value at the current epoch.
        model: The model being trained.
        inputs_norm: The normalized input data.
        outputs_norm: The normalized target data.
    """
    # Clear the previous plot from the output cell
    clear_output(wait=True)

    # Make predictions using the current state of the model
    predicted_norm = model(inputs_norm)

    # Convert tensors to NumPy arrays for plotting
    x_plot = inputs_norm.numpy()
    y_plot = outputs_norm.numpy()
    
    # Detach predictions from the computation graph and convert to NumPy
    y_pred_plot = predicted_norm.detach().numpy()

    # Sort the data based on distance to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis=0).flatten()

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original normalized data points
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Actual Normalized Data')

    # Plot the model's predictions as a line
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color='green', label='Model Predictions')

    # Set the title of the plot, including the current epoch
    plt.title(f'Epoch: {epoch + 1} | Normalized Training Progress')
    # Set the x-axis label
    plt.xlabel('Normalized Distance')
    # Set the y-axis label
    plt.ylabel('Normalized Time')
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Pause briefly to allow the plot to be rendered
    time.sleep(0.05)

def plot_x_y(x_values: 'list', y_values: 'list', x_label="", y_label="", title="", save_path=""):
    """
    Plots the x and y values with  a plot title.
    """
    # Clear the previous plot from the output cell
    clear_output(wait=True)

    # Convert list to NumPy arrays for plotting
    x_plot = np.array(x_values)
    y_plot = np.array(y_values)
    
    # Sort the data based on distance to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis=0).flatten()

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original normalized data points
    plt.plot(x_plot, y_plot, color='blue', marker='o', linestyle='none')

    # Set the title of the plot, including the current epoch
    plt.title(title)

    # Set the x-axis label
    plt.xlabel(x_label)
    # Set the y-axis label
    plt.ylabel(y_label)
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_multiX_y(x_values: "list[list]", x_names: 'list', x_colors: 'list', y_values: 'list', y_title="", title="", save_path=""):
    """
    Plots multiple types of x values against a single list of y values. Helpful for comparision plots, for
    example plotting training loss vs validation loss against epochs.
    """
    if len(x_values) != len(x_names) != len(x_colors) :
        print(f"Provide equal number of names and colors for the x values")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, xs in enumerate(x_values):
        if len(xs) != len(y_values):
            raise ValueError(
                f"x_values[{idx}] has length {len(xs)}, "
                f"but y_values has length {len(y_values)}."
            )
        ax.plot(xs, y_values, color=x_colors[idx], label=x_names[idx])
   
    ax.set_ylabel(y_title)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

 