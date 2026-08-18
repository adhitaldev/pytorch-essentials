"""
Simple neural network with a non-linear activation function
that takes in a more complex dataset. Non-linear activations
like ReLU are useful when a simple linear model does not fit 
to predict the result of a query.
"""
import torch
import torch.nn as nn
import os
import dotenv
from utils.csv_to_tensor import get_tensor_from_csv
from utils.data_plotter import plot_training_progress

dotenv.load_dotenv()

# Data ingestion from a csv file with multiple columns
# Everything but the last tensor is the input values
# The last value is the output.
data_file = os.getenv("HOUSING_FILE_COMPLEX")

data_tensor = get_tensor_from_csv(data_file) # looks like [[1.0, 2.0, 3.0], ...]
inputs = torch.stack([t[:2] for t in data_tensor])
outputs = torch.stack([t[2:] for t in data_tensor])

# Inputs and outputs normalized
inputs_mean = inputs.mean()
inputs_std = inputs.std()
outputs_mean = outputs.mean()
outputs_std = outputs.std()
inputs = (inputs - inputs_mean)/ inputs_std
outputs = (outputs - outputs_mean) / outputs_std

# Layer input outputs
num_neurons = 3
num_features = inputs.shape[1]
num_outputs = outputs.shape[1]

# Ensure a consistent result for random number generators
torch.manual_seed(27)
# Note: nn.Linear(in, out) builds a (out, in) weight matrix
# This should be in order - (batch, in_features) X (in_features, out_features)
model = nn.Sequential(
    nn.Linear(num_features, num_neurons), # 3 neurons each receiving 2 outputs
    nn.ReLU(),
    nn.Linear(num_neurons, num_outputs ))

# Loss and optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Loop
iters = 1000
epoch_eval_step = 100
for epoch in range(iters):
    # Reset optimizer's gradients
    optimizer.zero_grad()
    # Results
    results = model(inputs)
    # Loss
    loss = loss_func(results, outputs)
    # Backward propogation
    loss.backward()
    # Update model's params
    optimizer.step()
    if (epoch + 1) % epoch_eval_step == 0:
        print(f"Epoch: {epoch + 1} Loss: {loss.item()}")
        plot_training_progress(
            epoch=epoch,
            loss=loss,
            model=model,
            inputs_norm=inputs,
            outputs_norm=outputs
        )

# Prediction
with torch.no_grad():
    input_to_predict_for = [1000, 1990]
    new_price = torch.tensor([input_to_predict_for], dtype=torch.float32)
    prediction = model(new_price)
    print(f"Prediction for {input_to_predict_for} is {prediction}")


