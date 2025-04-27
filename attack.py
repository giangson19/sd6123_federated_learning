import ssl
import urllib.request
import matplotlib.pyplot as plt 
import os
import glob
import torchvision
import argparse
import sys

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import CIFARNet
from data import load_cifar10, create_dataloaders
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Membership Inference Attack on Federated Learning Models')
    parser.add_argument('--strategy', type=str, default="fedavg", 
                        choices=["fedavg", "fedprox", "fedavgm", "fedadam", 'differentialprivacyclientsideadaptiveclipping'],
                        help='Federated learning strategy')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for loading data')
    parser.add_argument('--show_images', action='store_true',
                        help='Whether to show sample images')
    return parser.parse_args()

def imshow(img):
    # Unnormalize CIFAR-10 images
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_confidence_scores(model, dataloader, device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            max_scores, _ = probs.max(dim=1)
            scores.extend(max_scores.cpu().numpy())
            labels.extend([1] * x.size(0))  # initially assume "members"
    return np.array(scores), np.array(labels)

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CIFARNet().to(device)
    
        
    model_file = 'models/{}_model.pth'.format(args.strategy)
    print("Loading pre-trained model from:", model_file)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.eval()

    # Load data
    trainset, testset = load_cifar10(root_path="./data")
    trainloader, testloader = create_dataloaders(trainset, testset, batch_size=args.batch_size)

    # Show images if requested
    if args.show_images:
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        imshow(torchvision.utils.make_grid(images))
        
        # Make predictions
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        print(f'Predicted labels: {predicted.cpu().numpy()}')

    # Perform Membership Inference Attack
    print(f"Running MIA against {args.strategy} model...")
    
    # Get confidence scores
    train_scores, train_labels = get_confidence_scores(model, trainloader, device)
    test_scores, test_labels = get_confidence_scores(model, testloader, device)
    test_labels[:] = 0  # label test samples as "non-members"

    # Combine data
    all_scores = np.concatenate([train_scores, test_scores])
    all_labels = np.concatenate([train_labels, test_labels])

    # Train attack model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    attack_model = LogisticRegression()
    all_scores = all_scores.reshape(-1, 1)
    attack_model.fit(all_scores, all_labels)

    # Predict and evaluate attack success
    predictions = attack_model.predict(all_scores)
    mia_accuracy = accuracy_score(all_labels, predictions)
    print(f"MIA against {args.strategy} - Attack Success Rate: {mia_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()