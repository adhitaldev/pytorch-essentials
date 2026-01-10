"""
To create a NN in pytorch, create a class that inherits from the 
nn.Module and define the layers in the __init__ function and 
define how the data will pass through the network in the forward function.
To accelerate ops, we move it to accelelration such as CUDA, MPS, MTIA or XPU.
If there is no accelerator, CPU is used.
"""
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def run_nn_example():
    nn = SimpleNN().to(device)

def get_model():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    using (f"Current Device: {device}")
    nn = SimpleNN().to(device)
    return nn

