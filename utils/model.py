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

class LinearModel(nn.Module):
    """Linear model for predicting the next stock price."""

    def __init__(self, input_size:int, output_size:int) -> None:
        """
        Parameters
        ----------
        input_size : int
            Size of the input.
        output_size : int
            Size of the output.

        Returns
        -------
        None.

        """
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
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
        x = self.linear(x)
        return x

class LSTM(nn.Module):
    """LSTM model for predicting the next stock price."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
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
            Number of layers in the LSTM. The default is 2.

        Returns
        -------
        None.

        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class GRU(nn.Module):
    """GRU model for predicting the next stock price."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
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
            Number of layers in the GRU. The default is 2.

        Returns
        -------
        None.

        """
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class RNN(nn.Module):
    """RNN model for predicting the next stock price."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
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
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

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
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out