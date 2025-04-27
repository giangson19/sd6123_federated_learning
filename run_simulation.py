import ssl
import urllib.request
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import argparse
import json
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import flwr as fl
from flwr.common import Metrics
from typing import Dict, Tuple, List
# from flwr.server.strategy import FedAvgM
from server import FedAvg, FedProx, FedAvgM, FedAdam
from client import client_fn  # Assuming CIFAR10Client is defined in client.py
from model import CIFARNet  # Assuming CIFARNet is defined in model.py
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Configure metrics aggregation function
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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Federated Learning with Flower')
    
    # Strategy selection
    parser.add_argument('--strategy', type=str, default='fedavgm', 
                        choices=['fedavgm', 'fedprox', 'fedavg', 'fedadam'],
                        help='FL strategy to use (default: fedavgm)')
    
    # Common strategy parameters
    parser.add_argument('--fraction-fit', type=float, default=1.0,
                        help='Fraction of clients to sample for training (default: 1.0)')
    parser.add_argument('--fraction-evaluate', type=float, default=1.0,
                        help='Fraction of clients to sample for evaluation (default: 1.0)')
    parser.add_argument('--min-fit-clients', type=int, default=3,
                        help='Minimum number of clients for training (default: 3)')
    parser.add_argument('--min-evaluate-clients', type=int, default=3,
                        help='Minimum number of clients for evaluation (default: 3)')
    parser.add_argument('--min-available-clients', type=int, default=3,
                        help='Minimum number of available clients (default: 3)')
    
    # Strategy-specific parameters
    parser.add_argument('--server-learning-rate', type=float, default=0.1,
                        help='Server-side learning rate for FedAvgM (default: 0.1)')
    parser.add_argument('--server-momentum', type=float, default=0.9,
                        help='Server-side momentum for FedAvgM (default: 0.9)')
    parser.add_argument('--proximal-mu', type=float, default=0.1,
                        help='Proximal term coefficient for FedProx (default: 0.1)')
    
    # FedAdam parameters
    parser.add_argument('--eta', type=float, default=1e-1,
                        help='Server-side learning rate for FedAdam (default: 1e-1)')
    parser.add_argument('--eta-l', type=float, default=1e-1,
                        help='Client-side learning rate for FedAdam (default: 1e-1)')
    parser.add_argument('--beta-1', type=float, default=0.9,
                        help='Momentum parameter for FedAdam (default: 0.9)')
    parser.add_argument('--beta-2', type=float, default=0.99,
                        help='Second moment parameter for FedAdam (default: 0.99)')
    parser.add_argument('--tau', type=float, default=1e-9,
                        help='Adaptability degree for FedAdam (default: 1e-9)')
    
    # Simulation parameters
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Number of clients to simulate (default: 3)')
    parser.add_argument('--num-rounds', type=int, default=10,
                        help='Number of federated learning rounds (default: 10)')
    
    return parser.parse_args()

def get_strategy(args):
    """Create strategy instance based on command-line arguments"""
    common_params = {
        'fraction_fit': args.fraction_fit,
        'fraction_evaluate': args.fraction_evaluate,
        'min_fit_clients': args.min_fit_clients,
        'min_evaluate_clients': args.min_evaluate_clients,
        'min_available_clients': args.min_available_clients,
        'evaluate_metrics_aggregation_fn': weighted_average,
    }
    
    # Convert model weights to Flower Parameters
    model = CIFARNet().to(DEVICE)
    model_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(model_weights)
    
    if args.strategy.lower() == 'fedavgm':
        return FedAvgM(
            **common_params,
            initial_parameters=initial_parameters,
            server_learning_rate=args.server_learning_rate,
            server_momentum=args.server_momentum,
        )
    elif args.strategy.lower() == 'fedadam':
        return FedAdam(
            **common_params,
            initial_parameters=initial_parameters,
            eta = args.eta,
            eta_l = args.eta_l,
            beta_1 = args.beta_1,
            beta_2 = args.beta_2,
            tau = args.tau,
        )
    elif args.strategy.lower() == 'fedprox':
        return FedProx(
            **common_params,
            proximal_mu=args.proximal_mu,
        )
    elif args.strategy.lower() == 'fedavg':
        return FedAvg(**common_params)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

# Run the Flower simulation
if __name__ == "__main__":
    args = parse_arguments()
    strategy = get_strategy(args)
    
    print(f"Starting federated learning with {args.strategy} strategy")
    print(f"Number of clients: {args.num_clients}")
    print(f"Number of rounds: {args.num_rounds}")
    
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, args.num_clients),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "memory": 4096},
    )

    with open(f"outputs/history_{args.strategy}.json", "w") as f:
        json.dump(history.metrics_distributed, f)