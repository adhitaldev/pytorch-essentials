"""
Simple plotter for visualization of data
WIP
"""
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

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