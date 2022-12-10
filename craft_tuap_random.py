import pickle
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from utils.dataloader import TUAPDataset, create_input_data
from utils.model import OldModelDNN
import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

log_folder = "runs/tuap/random"
writer = SummaryWriter(log_folder)

input_size = 17
output_size = 2
symbols = ['GOOG', 'AMZN', 'BLK', 'IBM', 'AAPL']
symbols = [symbols[0]]
models = [OldModelDNN(input_size, output_size)]

"""
Currently, the model (f) takes in an input (x) and returns an output (y).
We will craft a perturbation (delta) that will be added to the input (x) such that
the model will output the same output (y) i.e. f(x + delta) = y

"""

# Fooling output targets
fooling_targets = [torch.tensor([0., 1.]), torch.tensor([1., 0.])]

# Perturbation delta (O, H, L, C, V)
delta = torch.randn(35, 5, requires_grad=True)

# Model f
model = models[0]

# Train the model
num_epochs = 20

for symbol in symbols:
    model_name = model.__class__.__name__
    # Load saved weights
    checkpoint = torch.load(f'weights/{model_name}_{symbol}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    accuracy = checkpoint['accuracy']
    best_accuracy = checkpoint['accuracy']
    print(f"Model loaded with best accuracy {accuracy:.4f}")
    f = model
    f.eval()

    # Create random stock data
    train_dataset = TUAPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', train=True, normalize=False, sequence_length=5)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=False)

    test_dataset = TUAPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', train=False, normalize=False, sequence_length=5)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    x_train = next(iter(train_loader))[0].view(-1, 35, 5)
    x_test = next(iter(test_loader))[0].view(-1, 35, 5)
    for epoch in tqdm.tqdm(range(num_epochs)):
        for xi in x_train:
            # Forward pass
            x_hat = torch.from_numpy(create_input_data((xi + delta).detach().numpy())[:, :-2]).float()
            outputs = f(x_hat)
            y = fooling_targets[1].repeat(outputs.shape[0], 1)

        # Calculate accuracy
        with torch.no_grad():
            correct = 0
            total = 0
            for xi in x_test:
                x_hat = torch.from_numpy(create_input_data((xi + delta).detach().numpy())[:, :-2]).float()
                outputs = f(x_hat)
                y = fooling_targets[1].repeat(outputs.shape[0], 1)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            accuracy = correct / total
        writer.add_scalar('Acc/TUAP/test', accuracy, epoch)


model_list = ['GB','LR','SVM']

for model in model_list:
    delta = torch.randn(35, 5, requires_grad=True)

    for symbol in symbols:
        model = pickle.load(open(f'weights/{model}_{symbol}.sav', 'rb'))
        for epoch in tqdm.tqdm(range(num_epochs)):
            for xi in x_train:
                xy = create_input_data((xi + delta).detach().numpy())
                x_hat = xy[:, :-2]
                y_hat = xy[:, -2:-1]
                cls_proba = model.predict_proba(x_hat)
                cls_proba = cls_proba[:,1]
        
            # Calculate accuracy
            with torch.no_grad():
                correct = 0
                total = 0
                for xi in x_test:
                    xy = create_input_data((xi + delta).detach().numpy())
                    x_hat = xy[:, :-2]
                    y_hat = xy[:, -2:-1]
                    cls_proba = model.predict_proba(x_hat)
                    cls_proba = cls_proba[:,1]
                    cls_pred = model.predict(x_hat)
                    total += y_hat.size
                    correct += (cls_pred == y_hat).sum()
                accuracy = correct / total
            writer.add_scalar(f'Acc/TUAP/test/{model}', accuracy, epoch)

        # Save the TUAP model
        torch.save(delta, f'weights/TUAP_{symbol}.pth')

writer.flush()
writer.close()
