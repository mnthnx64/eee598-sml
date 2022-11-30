"""
The file parses the Kaggle S&P 500 dataset and saves a csv file for each stock
in the dataset.

The dataset can be downloaded from:
https://www.kaggle.com/nickdl/snp-500-intraday-data/home

Technical Information
Dates range from 2017-09-11 to 2018-02-16 and the time interval is 1 minute.
This is a MultiIndex CSV file, to load in pandas use:
dataset = pd.read_csv('dataset.csv', index_col=0, header=[0, 1]).sort_index(axis=1)
Stocks that entered or exited the Index during the dataset time range are omitted.
"""

import pandas as pd
import os

# Path to the dataset
path = 'dataset/dataset.csv'
path_to_save = 'daataset/splited_s&p500'

# Load the dataset
dataset = pd.read_csv(path, index_col=0, header=[0, 1]).sort_index(axis=1)

# Get the list of stocks
stocks = dataset.columns.levels[0]
print(f'Stocks: {stocks}')

# Create a folder to save the stocks
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

# Save each stock in a separate csv file
for stock in stocks:
    stock_data = dataset[stock]
    stock_data.to_csv(f'{path_to_save}/{stock}.csv')
    print(f'Stock {stock} saved')