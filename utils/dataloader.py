""""
This file contains the DataLoader class that is used to load the data from the
csv files and prepare it for training. It loads the data for only one stock.
"""

import pandas as pd
import numpy as np
import os
import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class SPDataset(Dataset):
    """Data loader for the S&P 500 dataset."""

    def __init__(self, csv_path, sequence_length=50, train=True, normalize=False):
        """
        Parameters
        ----------
        csv_path : str
            Path to the csv file with the data.
        sequence_length : int, optional
            Length of the sequence used for training. The default is 50.
        train : bool, optional
            If True, the data is used for training. If False, the data is used
            for testing. The default is True.
        normalize : bool, optional
            If True, the data is normalized. The default is True.

        Returns
        -------
        None.
        """
        self.data = pd.read_csv(csv_path)
        self.sequence_length = sequence_length
        self.train = train
        self.normalize = normalize

        # Add hour and minute columns (int) to the data from the timestamp column and drop the timestamp column
        self.data['hour'] = self.data['timestamp'].apply(lambda x: int(x.split(' ')[1].split(':')[0]))
        self.data['minute'] = self.data['timestamp'].apply(lambda x: int(x.split(' ')[1].split(':')[1]))
        self.data.drop('timestamp', axis=1, inplace=True)

        # Forward fill the NaN values and drop the remaining NaN values
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna(inplace=True)

        # Convert the data to numpy array
        self.data = self.data.values

        self.features = []

        for row_multiple in range(len(self.data) // 5):
            row = row_multiple * 5
            current_window = self.data[row:row + 5, 4]
            average = np.mean(current_window, dtype=np.float64)
            standard_deviation = np.std(current_window, dtype=np.float64)
            a1_param,_ = np.polyfit(range(5), current_window.astype('float64'), 1)
            minute= self.data[row+5-1,6]
            hour= self.data[row+5-1,5]
            close= self.data[row+5-1,4]
            self.features.append([average, standard_deviation, a1_param, minute, hour, close])

        self.features = np.array(self.features, dtype=np.float64)

        logAVG = self.np_pseudo_log(self.features[:,0])
        X_time = np.concatenate([self.features[:,3], self.features[:,4], self.features[:,5], self.features[:,0]],axis=1)


        # Split the data into train and test sets
        train_size = int(len(self.data) * 0.8)
        self.train_set = self.data[:train_size]
        self.test_set = self.data[train_size - self.sequence_length:]

        # Normalize the data
        if normalize:
            self.scaler = MinMaxScaler()
            self.train_set = self.scaler.fit_transform(self.train_set)
            self.test_set = self.scaler.transform(self.test_set)

        # Select data to use
        if train:
            self.data = self.train_set
        else:
            self.data = self.test_set

    def np_pseudo_log(self, data):
        plog = []
        for i in range(len(data)):
            if(i>0):
                plog.append(np.log(data[i]/data[i-1])) 
        return np.array(plog, dtype = np.float64)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """Get a sequence at a given index."""
        # Get the sequence
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]

        # Convert the sequence to tensor
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).float()

        return x, y

    def get_train_data(self):
        """Get the training data."""
        return self.train_set

    def get_test_data(self):
        """Get the test data."""
        return self.test_set

    def get_scaler(self):
        """Get the scaler object."""
        return self.scaler

    def get_data(self):
        """Get the data."""
        return self.data

    def get_random_batch(self, batch_size=10):
        """Get a random batch of data."""
        # Get random starting points for the sequences in the batch
        start_points = np.random.randint(0, len(self.data) - self.sequence_length, batch_size)

        # Get the sequences
        x_batch = []
        y_batch = []
        for start in start_points:
            x, y = self.__getitem__(start)
            x_batch.append(x)
            y_batch.append(y)

        # Convert the sequences to tensors
        x_batch = torch.stack(x_batch)
        y_batch = torch.stack(y_batch)

        return x_batch, y_batch