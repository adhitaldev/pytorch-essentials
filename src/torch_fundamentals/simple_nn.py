"""
 Simple Neural Network that takes in delivery miles as input
 and outputs the delivery time.
 
 Let's take in a simple house price prediction.
 SqFeet ---> (Neuron) ---> Price
"""
import torch
import torch.nn
import os
import dotenv
from   utils.csv_to_tensor import get_tensor_from_csv

dotenv.load_dotenv()

# Data Ingestion from a two-column csv file. First column is input
# and the second column is the output. Each row is a sample.
data_file = os.getenv("HOUSING_FILE_SIMPLE")

data_tensor = get_tensor_from_csv(data_file, delimiter=",", has_header=True) # sq_feets

# Input and output tensors
inputs = data_tensor[:, 0].view(-1, 1)  # Reshape to (N, 1)
outputs = data_tensor[:, 1].view(-1, 1)  # Reshape
print(f"Inputs: {inputs}")
print(f"Outputs: {outputs}")

# Sequencial model that models as (Time = W * Distance + B)
# takes in 1 input i.e.e distance and 1 output i.e. time
model = torch.nn.Sequential(torch.nn.Linear(1, 1))

# Functions
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training Loop
iters = 1000
epoch_eval_step = 100
for epoch in range(iters):
    # Reset the optimizer's gradients
    optimizer.zero_grad()
    # Make the predictions
    result = model(inputs)
    # Loss
    loss = loss_func(result, outputs)
    # Backward propogation that adjusts the weights
    loss.backward()
    # Update the model's params
    optimizer.step()
    if (epoch + 1) % epoch_eval_step == 0:
        print(f"Epoch: {epoch + 1} Loss:  {loss.item()}")

# Prediction - use torch.no_grad() context manager for efficiency
with torch.no_grad():
    input_to_predict_for = 1000
    new_distance = torch.tensor([[input_to_predict_for]], dtype=torch.float32)
    prediction = model(new_distance)
    print(f"Prediction for {input_to_predict_for} is {prediction.item():.1f}")


