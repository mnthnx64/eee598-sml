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

class StockDataset(Dataset):
    """Data loader for the S&P 500 dataset."""

    def __init__(self, csv_path, sequence_length=5, train=True, normalize=False, offset=0):
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

        for row_multiple in range(len(self.data) // self.sequence_length - 1):
            row = row_multiple * self.sequence_length + offset
            current_window = self.data[row:row + self.sequence_length, 3]
            average = np.mean(current_window, dtype=np.float64)
            standard_deviation = np.std(current_window, dtype=np.float64)
            # Slope of the line of best fit gives the trend of the stock (Going up or down)
            stock_trend,_ = np.polyfit(range(self.sequence_length), current_window.astype('float64'), 1)
            minute= self.data[row+self.sequence_length-1,6]
            hour= self.data[row+self.sequence_length-1,5]
            close= self.data[row+self.sequence_length-1,3]
            self.features.append([average, standard_deviation, stock_trend, minute, hour, close])

        self.features = np.array(self.features, dtype=np.float64)

        avg = self.features[:,0].reshape(-1,1)
        standard_devs = self.features[:,1].reshape(-1,1)
        trends = self.features[:,2].reshape(-1,1)
        minutes = self.features[:,3].reshape(-1,1)
        hours = self.features[:,4].reshape(-1,1)
        close = self.features[:,5].reshape(-1,1)

        log_avg = self.np_pseudo_log(self.features[:,0]).reshape(-1,1)
        x_time = np.concatenate([minutes, hours, close, avg],axis=1)
        proc_data = np.concatenate([log_avg, standard_devs, trends],axis=1)

        lookback = 5

        proc_x, proc_y = [],[]
        for i in range(len(proc_data) - lookback):
            proc_x.append(np.matrix(proc_data[i:i+lookback, :]))
            proc_y.append(proc_data[i+lookback, 0])
        proc_x = np.array(proc_x)
        proc_y = np.array(proc_y)

        X_train = np.reshape(proc_x, (proc_x.shape[0],proc_x.shape[1]*proc_x.shape[2]))
        X_time = x_time[lookback:]

        merged = np.concatenate([X_train,X_time],axis=1)
        merged = np.array(merged,dtype=float)
        proc_y = proc_y>0
        proc_y = proc_y.astype(int)

        # One hot encode the proc_y
        proc_y = np.eye(2)[proc_y]

        self.data = np.concatenate([merged,proc_y],axis=1)

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
        """Get the pseudo log of the data.

        Parameters
        ----------
        data : numpy array
            Data to get the pseudo log of.

        Returns
        -------
        numpy array
            Pseudo log of the data.
        """
        plog = [0]
        for i in range(len(data)):
            if(i>0):
                plog.append(np.log(data[i]/data[i-1])) 
        return np.array(plog, dtype = np.float64)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        """Get the item at the given index."""
        x = self.data[idx, :-4]
        y = self.data[idx, -2:]

        # Convert the data to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

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


class TUAPDataset(Dataset):
    """Dataset for TUAP."""

    def __init__(self, csv_path, sequence_length, train=True, normalize=True):
        """Initialize the dataset.

        Parameters
        ----------
        csv_path : str
            Path to the csv file.
        sequence_length : int
            Length of the sequence.
        train : bool, optional
            Whether to use the train set or the test set, by default True
        normalize : bool, optional
            Whether to normalize the data, by default True
        """
        self.data = pd.read_csv(csv_path)
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna(inplace=True)
        self.sequence_length = sequence_length
        self.data['date'] = self.data['timestamp'].apply(lambda x: x.split(' ')[0])

        # Get all the data for each day
        self.data = self.data.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        # Get the numpy array of the data
        self.data = self.data[['open', 'high', 'low', 'close', 'volume']].values
        idx = int(len(self.data) * 0.8)
        # Convert idx to nearest 10
        idx = idx - (idx % 35)
        last_idx = len(self.data) - (len(self.data) % 35)
        if train:
            self.data = self.data[:idx]
        else:
            self.data = self.data[last_idx-70:last_idx]

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get the item at the given index."""
        x = self.data

        # Convert the data to tensors
        x = torch.from_numpy(x).float()

        return x, False
        

def create_input_data(data):
    """
    For each 5 iteration of sequence_length, find 
        For each 5 iteration of sequence_length, find 
    For each 5 iteration of sequence_length, find 
    - Psuedo log return of the stock price
    - Standard deviation
    - Trend -> 1 if positive, 0 if negative
    - Average of close price
    - Close price of the last day
    """
    proc_data = []
    sequence_length = 5
    for i in range(0, len(data), sequence_length):
        if i + sequence_length < len(data):
            current_window = data[i:i + sequence_length, 3]
            average = np.mean(current_window, dtype=np.float64)
            standard_deviation = np.std(current_window, dtype=np.float64)
            # Slope of the line of best fit gives the trend of the stock (Going up or down)
            stock_trend,_ = np.polyfit(range(sequence_length), current_window.astype('float64'), 1)
            close= data[i + sequence_length-1, 3]
            proc_data.append([average, standard_deviation, stock_trend, close])

    proc_data = np.array(proc_data, dtype=np.float64)

    avg = proc_data[:,0].reshape(-1,1)
    standard_devs = proc_data[:,1].reshape(-1,1)
    trends = proc_data[:,2].reshape(-1,1)
    close = proc_data[:,3].reshape(-1,1)

    plog = [0]
    d = proc_data[:,0]
    for i in range(len(d)):
        if(i>0):
            plog.append(np.log(d[i]/d[i-1])) 

    log_avg = np.array(plog, dtype = np.float64).reshape(-1,1)
    x_time = np.concatenate([close, avg], axis=1)

    proc_data2 = np.concatenate([log_avg, standard_devs, trends],axis=1)
    lookback = 5
    proc_x, proc_y = [],[]
    for i in range(len(proc_data2) - lookback):
        proc_x.append(np.matrix(proc_data2[i:i+lookback, :]))
        proc_y.append(proc_data2[i+lookback, 0])
    proc_x = np.array(proc_x)
    proc_y = np.array(proc_y)
    X_train = np.reshape(proc_x, (proc_x.shape[0],proc_x.shape[1]*proc_x.shape[2]))
    X_time = x_time[lookback:]
    merged = np.concatenate([X_train,X_time],axis=1)
    merged = np.array(merged,dtype=float)
    proc_y = proc_y>0
    proc_y = proc_y.astype(int)

    # One hot encode the proc_y
    proc_y = np.eye(2)[proc_y]

    return np.concatenate([merged,proc_y],axis=1)

if __name__ == '__main__':
    # Create the datasetsymbol = 'AAPL'
    symbol='AAPL'
    # Load the data
    dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=5, train=True, normalize=False)
    # Get a random batch of training data
    x, y = dataset.get_random_batch()

    # Print the shapes of the data
    print(x.shape)
    print(y.shape)
    