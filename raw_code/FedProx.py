

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
from flwr.server.strategy import FedProx, FedAvgM
from flwr.client import NumPyClient

from model import CIFARNet  # Assuming CIFARNet is defined in model.py
from data import load_cifar10, create_dataloaders  # Assuming data loading functions are in data.py
from client import CIFAR10Client, client_fn  # Assuming CIFAR10Client is defined in client.py

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configure the FedProx strategy
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies_top1 = [num_examples * m["accuracy_top1"] for num_examples, m in metrics]
    accuracies_top5 = [num_examples * m["accuracy_top5"] for num_examples, m in metrics]
    losses = [num_examples * m["avg_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted accuracy)
    return {"accuracies_top1": sum(accuracies_top1) / sum(examples),
            "accuracies_top5": sum(accuracies_top5) / sum(examples),
            "losses": sum(losses) / sum(examples)
            }
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
