"""
This file contains different models that can be used for training.

The models are:
    - LinearModel
    - LSTM
    - GRU
    - RNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class OldModelDNN(nn.Module):
    """Old model for predicting the next stock price."""

    def __init__(self, input_size, output_size):
        """
        Parameters
        ----------
        input_size : int
            Size of the input.
        hidden_size : int
            Size of the hidden layer.
        output_size : int
            Size of the output.
        num_layers : int, optional
            Number of layers in the RNN. The default is 2.

        Returns
        -------
        None.

        """
        super(OldModelDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, int(np.floor(4*input_size/5)))
        self.fc2 = nn.Linear(int(np.floor(4*input_size/5)), int(np.floor(3*input_size/5)))
        self.fc3 = nn.Linear(int(np.floor(3*input_size/5)), int(np.floor(2*input_size/5)))
        self.fc4 = nn.Linear(int(np.floor(2*input_size/5)), int(np.floor(input_size/5)))
        self.fc5 = nn.Linear(int(np.floor(input_size/5)), output_size)
        self.activation = nn.Tanh()
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.

        """
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.fc5(out)
        out = self.out_activation(out)
        return out


class GradientBoostClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(GradientBoostClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, int(np.floor(4*input_size/5)))
        self.fc2 = nn.Linear(int(np.floor(4*input_size/5)), int(np.floor(3*input_size/5)))
        self.fc3 = nn.Linear(int(np.floor(3*input_size/5)), int(np.floor(2*input_size/5)))
        self.fc4 = nn.Linear(int(np.floor(2*input_size/5)), int(np.floor(input_size/5)))
        self.fc5 = nn.Linear(int(np.floor(input_size/5)), output_size)
        self.activation = nn.Tanh()
        self.out_activation = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.activation(out)
        out = self.fc4(out)
        out = self.activation(out)
        out = self.fc5(out)
        out = self.out_activation(out)
        return out