## PyTorch/DL Fundamental Topics
The general ML pipleline is: 
Data Ingestion -> Data Preparation -> Modeling -> Training -> Evaluation -> Deployment
These topics explore the Torch various tools and practices as building blocks for such
a pipeline.

# 1. Tensors
Tensors are fundamental,math optimized data structures in PyTorch that 
a model can understand.
```
py_list = [1.1, 2.2, 3.4][Tensor Examples]
tensor = torch.tensor(py_list)
```
Refer to [Tensor Examples](./tensors.py) script for various operations on creation, manipulation, and usage of Torch tensors.

# 2. Loss Functions
Loss functions are used to determine how right/wrong the model is by computing the error between the predicted value and the original value. There are various loss functions available in PyTorch.

## 2.1. Common Loss functions
__Mean Squared Error Loss__ makes sure all mistakes count
and punishes bigger errors than it does smaller errors. It's 
great for predicting continouous values like distance, temp,
prices.
```
nn.MSELoss() 
```

__Cross Entropy Loss__ is generally used when the model
isn't just picking an answer but provides a list of confidence scores for all
possible answers (example: image classification). It punishes overconfident wrong answer. Use it to predict multi-class classification.
```
nn.CrossEntropyLoss()
```

##2.2 Less common loss functions
__L1Loss__ measures average absolute difference - regression that's less sensitive to outliers.
```
nn.L1Loss()
```

__BCEWithLogitsLoss__ combines of sigmoid activation layer and binary cross entropy into a single, and is useful for binary classification.
class.
```
nn.BCEWithLogitsLoss()
``` 

__Negative Log Likelihood__ function performance of model whose input consists
of log probabilities for each class i.e. when LogSoftMax activation is used.It's useful for multi-class classification.
```
nn.NLLLoss() 
```
__SmoothL1Loss__ is a robust regression function that balances MSE and L1Loss 
benefits. It's closely related to Huber loss function.
```
nn.SmoothL1Loss() 
```

__KLDivLoss__ measures the Kullback-Leibler divergence. It calculates how much a
predicted probability distribution differs from a true reference distribution.
nn.KLDivLoss() # Measures difference between two probability distribution

loss.backward() evaluates how much each weight contributes to the error. 

[More info on Torch loss functions](https://docs.pytorch.org/cppdocs/api/nn/loss.html)

# 3. Optimizers
The optimizers help with speed and stability of gradient descent while handling
noise and escaping saddle points. They help with adjusting and updating the weights with a goal of minimizing loss.

__Stochastic Gradient Descent (SGD)__ optimizers's simple strategy is to increase the weight gradient if it's negative and to decrease the weight gradient if it's positive. If the gradient is big, make a big change, and if it's small, make a
small change. It scales the gradient first using the learning rate. A good learning rate has a steady descent to the global minima.
```
sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

__Adam__ optimizer scales the learning rate adaptively for each individual parameter. It's popular reliable, flexible, and often faster than other optimizers.
```
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

CAUTION! Don't copy SGD's learning rate for Adam. The latter is tuned  differently, and may destabilize or break it. An ideal baseline learning rate for ADAM is typically 2 to 4 orders of magnitude smaller than the ideal rate 
for SGD.
```
Other optimizers in PyTorch include RMSProp, Adagrad, etc.
[More Info on PyTorch Optimizers](https://docs.pytorch.org/docs/2.13/optim.html)

## 4. Data Management and Loading
General pattern for data handling is to define the transform, prepare your dataset with transform, and then use the Torch's DataLoader to
load it for training and evaluation.
```
    # Transforms are operations that run on each data point as they are loaded. Compose just means to do the ops in the specific order.
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,)(std,))
    ])
```
```
    # Dataset - Torch has different pre-built datasets which have options for train/test, whether to download is, and apply transformation.
    
    dataset = SomeDataSet('./data', train=True, download=True, transform = transform)
    firstitem = dataset[0]
```

```
    # DataLoader - Allows to load the data and servein batches for training. batch_size tells the loader how many samples to serve with option to shuffle.
    
    dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training would look like this:
    for batch_idx, (data, labels) in enumerate(dataset_loader):
        output = model (data)
```

## 5. Device Management 
Every model or tensor lives on a device - whether GPU or CPU. These need to be
told where to live. Model or tensors on different devices can destablize or crash the training/model. Default device is CPU for PyTorch.

```
torch.cuda.is_available() # Check whether a GPU/accelerator is available
```
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu) # is a common pattern
```

```
# Moving model params and training data to selected device
model = MyModel().to(device) # puts model's params in the selected device
for inputs, targets in dataloader:
    inputs = inputs.to(device)
    targets = targets.to(device)

Always assign to a new variable. For example:
x.to(device) will not move.
x = x.to(device) will move
```

```
# Checking where does an object live
print(inputs.device) # For tensors
print(next(model.parameters().device))

```
Be careful of GPU RAM availability as well when moving objects to it. Batch size of 32-64 is a good starting point.


## 6. Usage Notes Summary
``` 
loss = nn.MSELoss(model.parameters(), lr=0.01) # Measures the error between prediction and groundtruth
```
```
loss.backward() # Diagnoses how each param contributes to the error. When called, PyTorch adds gradient to what's already there. If zero_grad() is not called on the optimizer, the gradients are acuumulated with last batches' incorrectly until the training breaks. PyTorch does this because it's useful for some use cases like gradient accumulation, or certain training schedules.
```
```
optimizer.step() # Use the diagnostic scores to update the weights
```
```
optimizer.zero_grad() # Clears the gradients over successive batches and is the general recommended step.
```
```
# A complete training loop. Follow this pattern for most scripts:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
optimizer = optim.Adam(model.parameters())
loss_function = nn.CrossEntropyLoss()

for inputs, targets in dataloader():
    inputs = inputs.to(device)
    targets = targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
```











