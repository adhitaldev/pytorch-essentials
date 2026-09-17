# PyTorch Techniques
This contains notes on metrics, optimzation, paramter tuning, efficient pipelines, etc.

## PyTorch Metrics
__torchmetrics__ provides the tools to calculate these common metrics:  
TP = True Positives  
TN = True Negatives   
FP = False Positives   
FN = False Negatives 

### Accuracy
Accuracy is the measure of the correct predictions. You want to maximize accuracy during training and validation.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN) .... (to be maximized)

```

### Precision
Precision is how often the model's positive predictions are correct. You want to have a value closer to 1.
This is super important if the false positives are costly to the use case.
```
Precision = TP / (TP + FP)
```

### Recall
Recall measures how many of the positive cases did the model correctly identify. You want to have a value closer to 1.
Recalls matter most when false negatives are costly.
```
Recall = TP / (TP + FN) 
```

### F1 Score
Precision and recalls need to be balanced, preferably. F1 scores combines both and give a harmonic mean.
You want to  use it when both precision and recall are equally important and want have a value closer to 1.
```
F1 Score = 2 * (Precision * Recall)/(Precision + Recall)
```

```
#Example in PyTorch:

import torchmetrics
def evaluate_metrics(model, val_dataloader, device, num_classes=10):
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average="macro").to(device)
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)

# macro-average treats all classes equally, and is important when class balance is important.
# micro-average computes metrics globally by aggregating all TP, FP, and FN and is useful when classes size vary significantly.
# weighted-average averages metrics across classes, weighted by the number of true instances per class. Use it when you want to reflect a true data distribution, ensuring that performance on common classes contributes proportionally to the final score than the rare classes.
```

## Optimization
The goals of optimization is to:
 * maximize the accuracy, recall, precision, or f1 score by tuning hyperparameters like the learning rate.
 * minimize training/inference time, or the loss functions

This can be done by externally by having cleaner, noise-free data. And it can be done internally by tuning hyperparameters
(like learning rate), introducing drop out layers, etc. 

## Learning Rate Schedulers
Learning rate schedulers in PyTorch start off with a higher learning rate in the beginning and then reduce the learning rate as the training progresses. This provides the best of both worlds of starting with higher or lower learning rates.
[PyTorch Learning Rates](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate)

__StepLR__ reduces learning rate after the given step with a given rate (Gamma Rate).
```
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2) # reduce the learning rate by 20% it's prior value
```
__ReduceLROnPlateau__ reduces learning rate by a factor after a patience period (num of epochs) during there is no improvement in training. The 'mode' argument defines whether the target metric should be maximized or minimized.
```
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
scheduler.step(val_acc) # Needs a metric passed to it
```
__CosineAnnealingLR__ gradually reduces the learning rate from initial value to a minimum  over the epochs period. It is beneficial for fine tuning and for long training sessions where gradual adjustments can enhance convergence.
```
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0.0002)
scheduler.step()
```
In a training loop, a scheduler can look somthing like:
```
for epoch in range(n_epochs):
    train_loss, train_acc = ...
    val_loss, val_acc = ...
    # Get the current learning rate before stepping the scheduler for record
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
```
# Tunable Hyperparameters 
The hyperparameters for any framework can be broadly categorized into 3 parts:
## Architectural 
These parameters define how the model is built.

__numOfNeurons__ More neurons leads to better results with complex datasets at the cost of more weights, more memory, and computational cost.

__numLayers__ Shallow networks are easier to interpret and computationally efficient but may struggle with complex data set, and require feature engineering. The deeper the network, the more patterns it can learn but will take longer and more memory. Deeper networks with limited data can lead to overfitting.

__ActivationFunctions__ These determine neuron output introduce non-linearlity for the network to learn complex pattern. The common ones are ReLU for hidden activations, Sigmoid (binary result) on final layers , and Softmax (classification)on final layers.

## Training 
__Optimizers__

__LearningRates & Schedulers__

__BatchSize__ Number of samples processeed before updating model's internal parameters. 34 or 64 is a good starting point for batch size. Smaller batch sizes take longer to train, but can escape local minima and take less memory.Larger batch sizes are quicker to train but take more memory and can get stuck in local minima.

## Regularization 
Regularization techniques help prevent over fitting.
__WeightDecay__ Penalizing large weights by adding the squares of the weights to the loss function.  

__DropoutLayers__ where a percentage of the neurons in a layer is disabled. Common drop out rates range from 0.1 - 0.5.

__BatchNomalization__ where the activations are normalized to a mean of zero and standard deviation of 1. Thisprevent training instability and to increase model performance.

__Early Stopping__ Halts training when model's performance on validation set stops improving.


__Start with a simple baseline model before staring to tune the parameters. Use the Torch's default parameters to start off with, or refer to literature for other similar works.__

## Hyperparamter Tuning with Hyperparamters
Traditionally, hyperparameter tuning was done with exhaustive run with all possible combinations.
Optuna is a tree-structured estimater tool that performs hyperparameter tuning efficiently. It uses results from previous trials, focuses on regions that are likely to yield better results, and it's powerful for large search spaces. 
It has these three major plots to visualize hyperparamters and their effect on training
and validation accuracies:

__plot_optimization_history__: This plot shows the optimization history of the objective function, allowing you to see how the performance of the model improved over time. It provides a visual representation of the objective values (in this case, accuracy) across different trials.

__plot_param_importances__: This plot shows the importance of each hyperparameter in the optimization process. It helps identify which hyperparameters had the most significant impact on the model's performance, allowing you to focus on the most influential hyperparameters in future experiments.

__plot_parallel_coordinate__: This plot visualizes the relationship between different hyperparameters and the objective function. It allows you to see how different hyperparameter configurations affected the model's performance, providing insights into the interactions between hyperparameters and their impact on the objective value.

[CiFarCNN with Optuna Implementation](../dnn/cnn/cifar_cnn_optuna.py) 


## Model Efficiency Evaluations
During deployment, model characterstics like model size (memory footprint), inference time, power consumption, latency and throughput requirements also become important than just accuracy.

```
# Example: Getting model's size in MB
def get_model_size_in_mb(model):
      params_size = 0
      # Trainable params
      for param in model.params():
          params_size += param.nelement() * param.element_size()
      # Non-trainable params like buffers, constants, etc
      buffer_size = 0
      for buffer in model.buffers():
          buffer_size += buffer.nelement() * buffer.element_size()
    
     size_in_mb = (params_size + buffer_size) / 1024 ** 2
     return size_in_mb
```

```
# Example: Comparing models based on weighted scores of different variables
def select_best_model_weighted(results, weights=None):
    # Convert the DataFrame to a dictionary for easier processing.
    results = results.to_dict(orient="index")
    
    # If no weights are provided, define a default set that prioritizes accuracy.
    if weights is None:
        weights = {"accuracy": 0.5, "model_size_mb": 0.2, "inference_time_ms": 0.3}

    # Get the list of metrics to be considered from the weights dictionary.
    metrics = list(weights.keys())
    # Initialize a dictionary to store the normalized metric values for each model.
    normalized = {name: {} for name in results}

    # Loop through each metric to normalize its values across all models.
    for metric in metrics:
        # Extract all values for the current metric to find the min and max.
        values = [res[metric] for res in results.values()]
        min_val, max_val = min(values), max(values)
        # Calculate the range of values, avoiding division by zero.
        range_val = max_val - min_val if max_val != min_val else 1.0

        # Iterate through each model's results to calculate its normalized score.
        for name, res in results.items():
            value = res[metric]
            # Check the metric type to determine the normalization direction.
            if metric == "accuracy":
                # For accuracy, higher values are better, so normalize directly.
                norm_value = (value - min_val) / range_val
            else:
                # For size and time, lower values are better, so invert the normalization.
                norm_value = 1 - (value - min_val) / range_val
            # Store the calculated normalized value.
            normalized[name][metric] = norm_value

    # Calculate the final weighted score for each model.
    scores = {
        name: sum(weights[metric] * normalized[name][metric] for metric in metrics)
        for name in results
    }

    # Find the model with the highest overall score.
    best_model = max(scores.items(), key=lambda x: x[1])
    # Return the name of the best model and the dictionary of all scores.
    return best_model[0], scores

```








 



