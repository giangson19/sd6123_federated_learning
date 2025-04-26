
# Flower Client
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import flwr as fl
from flwr.common import Metrics
from typing import Dict, Tuple, List
from flwr.server.strategy import FedProx, FedAvgM
from flwr.client import NumPyClient

from model import CIFARNet  # Assuming CIFARNet is defined in model.py
from data import load_cifar10, create_dataloaders  # Assuming data loading functions are in data.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        acc_top1, acc_top5, avg_loss = self._test(self.model, self.testloader)
        return avg_loss, len(self.testloader.dataset), {"accuracy_top1": acc_top1, 'accuracy_top5': acc_top5, 'avg_loss': avg_loss}

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
        # Top 5 Accuracy:
        def compute_top5_accuracy(output, target):
            # Get top 5 predictions for each sample
            _, top5 = output.topk(5, dim=1)
            # Expand the target to match top5 shape
            target_expanded = target.view(-1, 1).expand_as(top5)
            # Count how many times the target label appears in the top 5
            correct_top5 = (top5 == target_expanded).sum().item()
            return 100 * correct_top5 / target.size(0)
        
        model.eval()
        correct_top1, correct_top5, total = 0, 0, 0
        total_loss = 0  # Add this to accumulate test loss
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                outputs = model(x)
                loss = F.cross_entropy(outputs, y)  # Compute test loss
                total_loss += loss.item()

                pred_top1 = outputs.argmax(dim=1)
                correct_top1 += (pred_top1 == y).sum().item()
                correct_top5 += compute_top5_accuracy(outputs, y) * y.size(0) / 100
                total += y.size(0)

        avg_loss = total_loss / total
        accuracy = correct_top1 / total
        acc_top5 = correct_top5 / total
        return accuracy, acc_top5, avg_loss

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

    
def client_fn(cid: str, num_clients: int = None):
    """Create a Flower client representing a single device."""
    # Get num_clients from args if not provided directly
    if num_clients is None:
        # Access args from the global scope
        num_clients = args.num_clients
        
    trainset, testset = load_cifar10()

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
