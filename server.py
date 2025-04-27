
# Flower Client
import flwr as fl
import numpy as np 
from model import CIFARNet  # Assuming CIFARNet is defined in model.py
import torch
from collections import OrderedDict    
from typing import List, Optional, Union
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CIFARNet().to(DEVICE)

def save_model(model, strategy, aggregated_parameters):
    """Save the model to disk"""
    
    print(f"Saving aggregated_parameters...")

    # Convert `Parameters` to `list[np.ndarray]`
    aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
        aggregated_parameters
    )

    # Convert `list[np.ndarray]` to PyTorch `state_dict`
    params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
    os.makedirs('models', exist_ok=True) 
    # Save the model to disk
    torch.save(model.state_dict(), f"models/{strategy}_model.pth")

class FedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregamte model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            save_model(net, "fedavg", aggregated_parameters)
            
        return aggregated_parameters, aggregated_metrics
    
    
class FedProx(fl.server.strategy.FedProx):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using FedProx and store checkpoint"""

        # Call aggregate_fit from base class (FedProx) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            save_model(net, "fedprox", aggregated_parameters)
            
        return aggregated_parameters, aggregated_metrics

class FedAvgM(fl.server.strategy.FedAvgM):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate model weights using FedAvgM and store checkpoint"""

        # Call aggregate_fit from base class (FedAvgM) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            save_model(net, "fedavgm", aggregated_parameters)

        return aggregated_parameters, aggregated_metrics
    
class FedAdam(fl.server.strategy.FedAdam):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregamte model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            save_model(net, "fedadam", aggregated_parameters)
            
        return aggregated_parameters, aggregated_metrics
    
