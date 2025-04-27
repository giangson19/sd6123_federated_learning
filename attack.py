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
import numpy as np
from model import CIFARNet
from data import load_cifar10, create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Membership Inference Attack on Federated Learning Models')
    parser.add_argument('--strategy', type=str, default="differentialprivacyclientsidefixedclipping", 
                        choices=['centralized', "fedavg", "fedprox", "fedavgm", "fedadam", 'differentialprivacyclientsidefixedclipping'],
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

def get_attack_features(model, dataloader, device, member=True):
    model.eval()
    features = []
    labels = []
    label_value = 1 if member else 0
    criterion = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)

            top1_confidence, _ = probs.topk(1, dim=1)
            top2_confidence = probs.topk(2, dim=1)[0][:, 1]

            confidence_gap = top1_confidence.squeeze() - top2_confidence
            entropy = -(probs * (probs + 1e-12).log()).sum(dim=1)
            loss = criterion(outputs, y)

            batch_features = torch.stack([
                top1_confidence.squeeze(),
                confidence_gap,
                entropy,
                loss
            ], dim=1)

            features.append(batch_features.cpu().numpy())
            labels.extend([label_value] * x.size(0))

    features = np.vstack(features)
    labels = np.array(labels)
    return features, labels

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = CIFARNet().to(device)
    model_file = 'models/{}_model.pth'.format(args.strategy)
    print("Loading pre-trained model from:", model_file)
    state_dict = torch.load(model_file, map_location=device)
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

    perm = np.random.permutation(len(all_scores))
    all_scores = all_scores[perm]
    all_labels = all_labels[perm]

    # Train attack model
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    attack_model = LogisticRegression()
    all_scores = all_scores.reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(all_scores, all_labels, test_size=0.3, random_state=42)
    attack_model.fit(X_train, y_train)

    predictions = attack_model.predict(X_val)
    mia_accuracy = accuracy_score(y_val, predictions)
    print(f"MIA against {args.strategy} - Attack Success Rate: {mia_accuracy * 100:.2f}%")

    print(f"Train (members) mean confidence: {train_scores.mean():.4f}")
    print(f"Test (non-members) mean confidence: {test_scores.mean():.4f}")

    # Enhanced attack with richer features
    print(f"Running Enhanced MIA attack with richer features against {args.strategy} model...")

    train_features, train_labels = get_attack_features(model, trainloader, device, member=True)
    test_features, test_labels = get_attack_features(model, testloader, device, member=False)

    all_features = np.vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels])

    perm = np.random.permutation(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]

    attack_model = LogisticRegression(max_iter=1000)

    X_train, X_val, y_train, y_val = train_test_split(all_features, all_labels, test_size=0.3, random_state=42)
    attack_model.fit(X_train, y_train)

    predictions = attack_model.predict(X_val)
    mia_accuracy = accuracy_score(y_val, predictions)

    print(f"Enhanced MIA Attack Success Rate against {args.strategy}: {mia_accuracy * 100:.2f}%")
    import matplotlib.pyplot as plt

    # For simple confidence-based features
    plt.hist(train_scores, bins=50, alpha=0.5, label="Members")
    plt.hist(test_scores, bins=50, alpha=0.5, label="Non-Members")
    plt.legend()
    plt.title("Top-1 Confidence Distributions")
    plt.show()
if __name__ == "__main__":
    main()

