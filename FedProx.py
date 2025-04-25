

import ssl
import urllib.request
import matplotlib.pyplot as plt
import torchvision
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import flwr as fl
from flwr.common import Metrics
from typing import Dict, Tuple, List
from flwr.server.strategy import FedProx
from flwr.client import NumPyClient

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model for CIFAR-10 (same as before)
class CIFARNet(nn.Module):
    def __init__(self):
        super(CIFARNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [B, 64, 8, 8]
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data loading (same as before, but moved inside the client)
def load_cifar10(root_path="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize CIFAR images
    ])
    trainset = datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
    return trainset, testset

def create_dataloaders(trainset, testset, batch_size=64):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloader, testloader

# Flower Client
class CIFAR10Client(NumPyClient):
    def __init__(self, cid, trainloader, testloader, client_config):
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = CIFARNet().to(DEVICE)
        self.client_config = client_config
        self.mu = self.client_config.get("mu", 0.1) # Get mu from client config

    def get_parameters(self, config: Dict[str, fl.common.NDArrays]) -> fl.common.NDArrays:
        """Get parameters of the local model."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> None:
        """Set parameters of the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[fl.common.NDArrays, int, Dict[str, fl.common.Scalar]]:
        """Train the model on the locally held dataset."""
        self.set_parameters(parameters, config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        epochs = self.client_config.get("epochs", 1)
        global_weights = self._parameters_to_state_dict(parameters)

        for _ in range(epochs):
            self._train(self.model, self.trainloader, optimizer, global_weights, self.mu)

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate the model on the locally held dataset."""
        self.set_parameters(parameters, config)
        loss, accuracy = self._test(self.model, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

    def _parameters_to_state_dict(self, parameters: fl.common.NDArrays) -> Dict[str, torch.Tensor]:
        """Convert NumPy parameters to a PyTorch state dictionary."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v).to(DEVICE) for k, v in params_dict}
        return state_dict

    def _train(self, model, loader, optimizer, global_weights, mu):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)

            # Add proximal term
            if mu > 0:
                local_weights = self.model.state_dict()
                proximal_term = 0.0
                for name, param in local_weights.items():
                    if name in global_weights:
                        proximal_term += torch.sum((param - global_weights[name]) ** 2)
                loss += (mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

    def _test(self, model, loader):
        model.eval()
        correct, total = 0, 0
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = F.cross_entropy(outputs, y, reduction='sum')
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

# Generate n random clients with different dataset splits
def generate_cifar10_clients(num_clients=3, client_config=None):
    if client_config is None:
        client_config = {}
    trainset, testset = load_cifar10()
    train_partitions = torch.utils.data.random_split(trainset, [len(trainset) // num_clients] * num_clients)
    test_partitions = torch.utils.data.random_split(testset, [len(testset) // num_clients] * num_clients)
    clients = []
    for i in range(num_clients):
        trainloader = DataLoader(train_partitions[i], batch_size=64, shuffle=True)
        testloader = DataLoader(test_partitions[i], batch_size=64)
        clients.append(CIFAR10Client(str(i), trainloader, testloader, client_config))
    return clients

def client_fn(cid: str):
    """Create a Flower client representing a single device."""
    trainset, testset = load_cifar10()
    num_clients = 3 # Assuming 3 clients for this example

    # Handle uneven splits for the training set
    base_size_train = len(trainset) // num_clients
    remainder_train = len(trainset) % num_clients
    train_lengths = [base_size_train] * num_clients
    for i in range(remainder_train):
        train_lengths[i] += 1
    train_partitions = torch.utils.data.random_split(trainset, train_lengths)

    # Handle uneven splits for the test set
    base_size_test = len(testset) // num_clients
    remainder_test = len(testset) % num_clients
    test_lengths = [base_size_test] * num_clients
    for i in range(remainder_test):
        test_lengths[i] += 1
    test_partitions = torch.utils.data.random_split(testset, test_lengths)

    client_id = int(cid)
    trainloader = DataLoader(train_partitions[client_id], batch_size=64, shuffle=True)
    testloader = DataLoader(test_partitions[client_id], batch_size=64)
    client_config = {"epochs": 1, "mu": 0.1} # Configure client-specific parameters
    return CIFAR10Client(cid, trainloader, testloader, client_config)

# Configure the FedProx strategy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted accuracy)
    return {"accuracy": sum(accuracies) / sum(examples)}

strategy = FedProx(
    fraction_fit=1.0,  # Sample all available clients for each round
    fraction_evaluate=1.0,  # Evaluate all available clients
    min_fit_clients=3,  # Minimum number of clients to perform fit
    min_evaluate_clients=3,  # Minimum number of clients to perform evaluation
    min_available_clients=3,  # Minimum number of total clients in the system
    evaluate_metrics_aggregation_fn=weighted_average, # Aggregate accuracy
    proximal_mu=0.1 # Set the FedProx mu directly here (if your Flower version supports it)
)

# Run the Flower simulation
if __name__ == "__main__":
    num_clients = 3 # You can adjust the number of clients
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),  # Number of FL rounds
        strategy=strategy,
        client_resources={"num_cpus": 1, "memory": 4096}, # Use 'num_cpus' instead of 'cpu'
    )
