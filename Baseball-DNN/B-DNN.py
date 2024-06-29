import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import wandb

# Initialize W&B project
wandb.init(project="baseball-predictions")


# loading and preprocessing data
class BaseballDataset(Dataset):
    def __init__(self, features_file, targets_file):
        features = self.load_data(features_file)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(self.load_data(targets_file), dtype=torch.float32)
        # Create a binary mask for each feature: 1 if the feature is not -1, otherwise 0
        self.mask = torch.tensor([[1 if val != -1 else 0 for val in row] for row in features], dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.mask[idx], self.targets[idx]

    def load_data(self, file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                # Processes each line to extract features/targets
                data_point = [float(val) for val in line.strip().split()]
                data.append(data_point)
        return data
    

class BaseballModel(nn.Module):
    def __init__(self, input_size, mask_size, hidden_sizes, output_size):
        super(BaseballModel, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size + mask_size, hidden_sizes[0])])
        self.hidden_layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.activation = nn.Tanh()

    def forward(self, x, mask):
        # Concatenate features and mask before feeding them into the network
        x = torch.cat((x, mask), dim=1)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x.squeeze()


# main logic
class BaseballPredictions:
    def __init__(self, train_features_file, train_targets_file, dev_features_file, dev_targets_file, input_size, hidden_size, output_size):
        self.train_data = None
        self.dev_data = None
        self.load_data(train_features_file, train_targets_file, dev_features_file, dev_targets_file)
        mask_size = input_size  # Assuming mask size is equal to the feature size
        self.model = BaseballModel(input_size, mask_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=1e-5)

    def load_data(self, train_features_file, train_targets_file, dev_features_file, dev_targets_file):
        self.train_data = BaseballDataset(train_features_file, train_targets_file)
        self.dev_data = BaseballDataset(dev_features_file, dev_targets_file)

    # for use as baseline
    def compute_mean_err(self, targets_file):
        with open(targets_file, 'r') as file:
            targets = [float(line.strip()) for line in file]
        mean_target = np.mean(targets)
        return np.mean((targets - mean_target) ** 2)

    def train(self, num_epochs, batch_size):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for features, mask, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features, mask)
                loss = self.criterion(outputs, targets.squeeze())
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * features.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
            # Log model to W&B
            wandb.log({"train loss": epoch_loss})

    def evaluate(self, batch_size):
        dev_loader = DataLoader(self.dev_data, batch_size=batch_size)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, mask, targets in dev_loader:
                outputs = self.model(features, mask)
                loss = self.criterion(outputs, targets.squeeze())
                total_loss += loss.item() * features.size(0)
        avg_loss = total_loss / len(dev_loader.dataset)
        print(f"Development Loss: {avg_loss:.4f}")
        # Log model to W&B
        wandb.log({"dev loss": avg_loss})

input_size = 100  # Number of features
hidden_size = [128]  # Number of hidden units/layers
output_size = 1  # Predicted strikeouts for the next 10 games
num_epochs = 20
batch_size = 256
baseball_model = BaseballPredictions('train.X', 'train.RT', 'dev.X', 'dev.RT', input_size, hidden_size, output_size)
baseball_model.train(num_epochs, batch_size)
baseball_model.evaluate(batch_size)
print(f"Baseline mse: {baseball_model.compute_mean_err('train.RT'):.4f}")

wandb.log({"num hidden layers": len(hidden_size), "hidden layer sizes": hidden_size[0]})
wandb.log({"num epochs": num_epochs, "batch size": batch_size})