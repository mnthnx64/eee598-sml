from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from numpy.random import seed
from math import sqrt
import numpy as np # linear algebra
import pandas as pd # data processing
from datetime import datetime
import timeit
from utils.dataloader import SPDataset
import torch
import tqdm
from utils.model import LinearModel, LSTM

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Set the seed for reproducibility
seed(42)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the hyperparameters
input_size = 7
hidden_size = 32
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.001
sequence_length = 5
symbol = 'AAPL'

# Load the data
dataset = SPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=True, normalize=False)

# Create the data loader
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)

# Create the model
model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)

# Create the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load model if exists
try:
    model.load_state_dict(torch.load(f'weights/{symbol}.pth'))
    print("Loaded model")
except:
    print("No model found")

# Train the model
for epoch in range(num_epochs):
    for i, (prices, price_pred) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}'):
        # Convert the data to torch tensors
        prices = prices.to(device)
        price_pred = price_pred.to(device)

        # Forward pass
        outputs = model(prices)
        loss = criterion(outputs, price_pred)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Save the model checkpoint and best model
    if epoch == 0:
        torch.save(model.state_dict(), 'best_model.ckpt')
        best_loss = loss.item()
    else:
        if loss.item() < best_loss:
            torch.save(model.state_dict(), 'best_model.ckpt')
            best_loss = loss.item()


