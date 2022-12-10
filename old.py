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
from utils.model import OldModelDNN, GradientBoostClassifier
import os
import yaml
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Set the seed for reproducibility
seed(14214)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Set the hyperparameters
input_size = 17
hidden_size = 32
num_layers = 2
output_size = 2
initial_epoch = 0
num_epochs = 30
learning_rate = 0.001
sequence_length = 5
symbols = ['GOOG', 'AMZN', 'BLK', 'IBM', 'AAPL']
models = [GradientBoostClassifier(input_size, output_size), OldModelDNN(input_size, output_size)]

# Create the tensorboard writer
log_folder = f"runs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(log_folder)

for model_index, m in enumerate(models):
    model_name = m.__class__.__name__
    print("Training model: ", model_name)
    best_accuracies = []
    for symbol in symbols:
        print("Training on symbol: ", symbol)

        # Load the data
        dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=True, normalize=False)
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        test_dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=False, normalize=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

        # Create the model
        model = m.to(device)

        # Create the loss function and the optimizer
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Add graph to tensorboard
        writer.add_graph(model, next(iter(train_loader))[0].to(device))

        # Load Model if exits
        if not os.path.exists('weights'):
            os.mkdir('weights')
        else:
            if os.path.exists(f'weights/{model_name}_{symbol}.pth'):
                checkpoint = torch.load(f'weights/{model_name}_{symbol}.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                initial_epoch = checkpoint['epoch']
                test_loss = checkpoint['loss']
                best_loss = checkpoint['loss']
                # num_epochs = checkpoint['num_epochs']
                accuracy = checkpoint['accuracy']
                best_accuracy = checkpoint['accuracy']
                print(f"Model loaded from epoch {initial_epoch} with accuracy {accuracy:.4f}")

        if not os.path.exists('plots'):
            os.mkdir('plots')

        # Train the model
        accuracy = {
            'train': [],
            'test': []
        }

        for epoch in tqdm.tqdm(range(initial_epoch, num_epochs), total=num_epochs, initial=initial_epoch, desc="Epoch"):
            model.train()
            for i, (prices, price_pred) in enumerate(train_loader):
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

            train_acc = EvaluateModel(model, train_loader, criterion, device).evaluate()['accuracy']
            accuracy['train'].append(train_acc)

            # Run the model on the test set
            model.eval()
            with torch.no_grad():
                for prices, price_pred in test_loader:
                    # Convert the data to torch tensors
                    prices = prices.to(device)
                    price_pred = price_pred.to(device)

                    # Forward pass
                    outputs = model(prices)
                    test_loss = criterion(outputs, price_pred)

            # Update losses
            test_acc = EvaluateModel(model, test_loader, criterion, device).evaluate()['accuracy']
            accuracy['test'].append(test_acc)

            # Add a plot of training and testing accuracy to writer
            writer.add_scalars(f"Acc/{model_name}_{symbol}", {"train": train_acc, "test": test_acc}, epoch)
            writer.add_scalars(f"Loss/{model_name}_{symbol}", {"train": train_loss.item(), "test": test_loss.item()}, epoch)

            # Save the model checkpoint and best model
            if epoch == 0:
                best_accuracy = accuracy['test'][-1]
                checkpoint = {
                    'epoch': epoch,
                    'num_epochs': num_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss.item(),
                    'accuracy': best_accuracy
                }
                torch.save(checkpoint, f'weights/{model_name}_{symbol}.pth')
            else:
                if accuracy['test'][-1] > best_accuracy:
                    checkpoint = {
                        'epoch': epoch,
                        'num_epochs': num_epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss.item(),
                        'accuracy': accuracy['test'][-1]
                    }
                    torch.save(checkpoint, f'weights/{model_name}_{symbol}.pth')
                    best_accuracy = accuracy['test'][-1]

        best_accuracies.append(best_accuracy)
    
    writer.add_scalars("Acc/best", {symbol: best_accuracy for symbol, best_accuracy in zip(symbols, best_accuracies)}, model_index)
    writer.add_scalars("Model/avg", {model.__class__.__name__: np.mean(best_accuracy) }, model_index)

writer.flush()
writer.close()