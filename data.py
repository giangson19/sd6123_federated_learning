
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