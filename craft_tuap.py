import torch
import torch.nn as nn
from utils.dataloader import TUAPDataset, create_input_data
from utils.model import GradientBoostClassifier, OldModelDNN
import tqdm
from torch.utils.tensorboard import SummaryWriter

log_folder = "runs/tuap"
writer = SummaryWriter(log_folder)

input_size = 17
output_size = 2
symbols = ['GOOG', 'AMZN', 'BLK', 'IBM', 'AAPL']
symbol = symbols[0]
models = [GradientBoostClassifier(input_size, output_size), OldModelDNN(input_size, output_size)]
# for symbol in symbols:
model = models[0]
model_name = model.__class__.__name__
# Load saved weights
checkpoint = torch.load(f'weights/{model_name}_{symbol}.pth')
model.load_state_dict(checkpoint['model_state_dict'])
accuracy = checkpoint['accuracy']
best_accuracy = checkpoint['accuracy']
print(f"Model loaded with best accuracy {accuracy:.4f}")

"""
Currently, the model (f) takes in an input (x) and returns an output (y).
We will craft a perturbation (delta) that will be added to the input (x) such that
the model will output the same output (y) i.e. f(x + delta) = y
"""
# Model f
f = model

# Fooling output targets
fooling_targets = [torch.tensor([0., 1.]), torch.tensor([1., 0.])]

# Create random stock data
train_dataset = TUAPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', train=True, normalize=False, sequence_length=5)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=False)

test_dataset = TUAPDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', train=False, normalize=False, sequence_length=5)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

x_train = next(iter(train_loader))[0].view(-1, 10, 5)
x_test = next(iter(test_loader))[0].view(-1, 10, 5)
# Convert x_train shape to (batch_size, 10, 5) and drop rest of the data


# Perturbation delta (O, H, L, C, V)
delta = torch.randn(10, 5, requires_grad=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam([delta], lr=0.001)

# Train the model
num_epochs = 20
for epoch in tqdm.tqdm(range(num_epochs)):
    f.train()
    for xi in x_train:
        # Forward pass
        x_hat = torch.from_numpy(create_input_data((xi + delta).detach().numpy())[:, :-2]).float()
        outputs = torch.sigmoid(f(x_hat))
        y = fooling_targets[0].repeat(outputs.shape[0], 1)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    f.eval()
    with torch.no_grad():
        for xi in x_test:
            x_hat = torch.from_numpy(create_input_data((xi + delta).detach().numpy())[:, :-2]).float()
            outputs = torch.sigmoid(f(x_hat))
            y = fooling_targets[1].repeat(outputs.shape[0], 1)
            loss = criterion(outputs, y)

    # Calculate accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for xi in x_test:
            x_hat = torch.from_numpy(create_input_data((xi + delta).detach().numpy())[:, :-2]).float()
            outputs = torch.sigmoid(f(x_hat))
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        accuracy = correct / total
        # if accuracy > best_accuracy:

        writer.add_scalar('Accuracy/test', accuracy, epoch)


pass
