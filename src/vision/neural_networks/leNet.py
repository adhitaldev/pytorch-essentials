"""
Implementation of the classic LeNet.
LeNet is a series of Conv Net created 
between 1988 to 1998 at the AT&T Bell 
Labs by Yann LeCun. It was designed to
read the small grayscale images.
"""

from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__(self)
