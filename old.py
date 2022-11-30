from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from numpy.random import seed
from math import sqrt
import numpy as np # linear algebra
import pandas as pd # data processing
from datetime import datetime
import timeit
from utils.dataloader import StockDataset
from utils.evaluate_model import EvaluateModel
import torch
import tqdm
from utils.model import LinearModel, LSTM
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Set the seed for reproducibility
seed(69)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Set the hyperparameters
input_size = 19
hidden_size = 32
num_layers = 2
output_size = 2
initial_epoch = 0
num_epochs = 100
learning_rate = 0.001
sequence_length = 5
symbol = 'AAPL'

# Load the data
dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=True, normalize=False)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)
test_dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=False, normalize=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

# Create the model
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# Create the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load Model if exits
if not os.path.exists('weights'):
    os.mkdir('weights')
else:
    if os.path.exists(f'weights/{symbol}.pth'):
        checkpoint = torch.load(f'weights/{symbol}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch']
        test_loss = checkpoint['loss']
        num_epochs = checkpoint['num_epochs']
        print(f"Model loaded from epoch {initial_epoch} with loss {test_loss}")

if not os.path.exists('plots'):
    os.mkdir('plots')

# Train the model
losses = {
    'train': [],
    'test': []
}

for epoch in range(initial_epoch, num_epochs):
    model.train()
    for i, (prices, price_pred) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}'):
        # Convert the data to torch tensors
        prices = prices.to(device)
        price_pred = price_pred.to(device)

        # Forward pass
        outputs = model(prices)
        train_loss = criterion(outputs, price_pred)

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {train_loss.item():.4f}')

    losses['train'].append(train_loss.item())

    # Run the model on the test set
    model.eval()
    with torch.no_grad():
        for prices, price_pred in tqdm.tqdm(test_loader, total=len(test_loader), desc=f'Test Epoch {epoch+1}/{num_epochs}'):
            # Convert the data to torch tensors
            prices = prices.to(device)
            price_pred = price_pred.to(device)

            # Forward pass
            outputs = model(prices)
            test_loss = criterion(outputs, price_pred)

    # Update losses
    losses['test'].append(test_loss.item())

    # Save the model checkpoint and best model
    if epoch == 0:
        checkpoint = {
            'epoch': epoch,
            'num_epochs': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss.item()
        }
        torch.save(checkpoint, f'weights/{symbol}.pth')
        best_loss = test_loss.item()
    else:
        if test_loss.item() < best_loss:
            checkpoint = {
                'epoch': epoch,
                'num_epochs': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss.item()
            }
            torch.save(checkpoint, f'weights/{symbol}.pth')
            best_loss = test_loss.item()

# Plot the losses
plt.plot(losses['train'], label='Train')
plt.plot(losses['test'], label='Test')
plt.legend()
plt.savefig(f'plots/{symbol}.png')

# Evaluate the model for different metrics
evaluator = EvaluateModel(model, train_loader, criterion, device)
evaluator.evaluate()
