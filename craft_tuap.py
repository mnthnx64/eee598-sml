import torch
import torch.nn as nn
from utils.dataloader import TUAPDataset
from utils.model import OldModel

symbols = ['GOOG', 'AMZN', 'BLK', 'IBM', 'AAPL']
symbol = symbols[0]
# for symbol in symbols:
model = OldModel(19, 2)
# Load saved weights
checkpoint = torch.load(f'weights/{symbol}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
initial_epoch = checkpoint['epoch']
test_loss = checkpoint['loss']
best_loss = checkpoint['loss']
# num_epochs = checkpoint['num_epochs']
accuracy = checkpoint['accuracy']
best_accuracy = checkpoint['accuracy']
print(f"Model loaded from epoch {initial_epoch} with accuracy {accuracy:.4f}")

"""
Currently, the model (f) takes in an input (x) and returns an output (y).
We will craft a perturbation (delta) that will be added to the input (x) such that
the model will output the same output (y) i.e. f(x + delta) = y
"""

# Create random stock data
dataset = TUAPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', train=True, normalize=False)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)

# Get the first batch of data

