import torch
from utils.dataloader import StockDataset
from utils.model import OldModelDNN

class EvaluateModel():
    def __init__(self, model, dataloader, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        """
        Evaluate and get the metrics of the model.

        Returns
        -------
        metrics : dict
            Dictionary of metrics.
        """
        self.model.eval()

        # Initialize the metrics
        loss = 0
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0

        # Iterate over the data
        for prices, price_pred in self.dataloader:
            # Convert the data to torch tensors
            prices = prices.to(self.device)
            price_pred = price_pred.to(self.device)

            # Forward pass
            outputs = self.model(prices)
            loss += self.criterion(outputs, price_pred).item()
            tp, tn, fp, fn = self.calculate_tfpn(torch.argmax(outputs, dim=1), torch.argmax(price_pred, dim=1))
            try:
                precision += tp / (tp + fp)
                recall += tp / (tp + fn)
                f1 += 2 * (precision * recall) / (precision + recall)
                accuracy += (tp + tn) / (tp + tn + fp + fn) * 100
            except ZeroDivisionError:
                pass

        # Return the metrics
        metrics = {
            'loss': loss / len(self.dataloader),
            'accuracy': accuracy / len(self.dataloader),
            'precision': precision / len(self.dataloader),
            'recall': recall / len(self.dataloader),
            'f1': f1 / len(self.dataloader)
        }
        return metrics

    def calculate_tfpn(self, outputs, y_label):
        """
        Calculate the True positives, True negatives, False positives, and False negatives.

        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.
        y_label : torch.Tensor
            The labels.

        Returns
        -------
        tp : int
            The number of true positives.
        tn : int
            The number of true negatives.
        fp : int
            The number of false positives.
        fn : int
            The number of false negatives.
        """
        # Initialize the variables
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        # Iterate over the data
        for i in range(len(outputs)):
            if outputs[i] == 1 and y_label[i] == 1:
                tp += 1
            elif outputs[i] == 0 and y_label[i] == 0:
                tn += 1
            elif outputs[i] == 1 and y_label[i] == 0:
                fp += 1
            elif outputs[i] == 0 and y_label[i] == 1:
                fn += 1

        return tp, tn, fp, fn

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    input_size = 19
    hidden_size = 32
    num_layers = 2
    output_size = 2
    initial_epoch = 0
    num_epochs = 100
    learning_rate = 0.001
    sequence_length = 5
    symbol = 'AAPL'
    checkpioint = torch.load(f'weights/{symbol}.pth')
    model = OldModelDNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
    model.load_state_dict(checkpioint['model_state_dict'])
    test_dataset = StockDataset(csv_path=f'dataset/splitted_s&p500/{symbol}.csv', sequence_length=sequence_length, train=False, normalize=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
    criterion = torch.nn.MSELoss()
    evaluator = EvaluateModel(model, test_loader, criterion, device)