import ssl
import urllib.request
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ssl._create_default_https_context = ssl._create_unverified_context

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model for CIFAR-10
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

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize CIFAR images
])

trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64)

# Define the output dictionary to store results
output_data = {
    "training_losses": [],
    "test_accuracies": [],
    "mia_attack_success_rate": None
}

# Training function
def train(model, loader, optimizer, epochs=50):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Save loss for visualization
        avg_loss = total_loss / len(loader)
        output_data["training_losses"].append(avg_loss)

        # Calculate test accuracy and store it
        acc_top1, acc_top5, test_loss = test(model, testloader)
        output_data["test_accuracies"].append({
            "epoch": epoch + 1,
            "top_1_accuracy": acc_top1,
            "top_5_accuracy": acc_top5,
            "test_loss": test_loss
        })

        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Test Loss = {test_loss:.4f}, Top-1 = {acc_top1:.2f}%, Top-5 = {acc_top5:.2f}%")

# Top 5 Accuracy computation
def compute_top5_accuracy(output, target):
    _, top5 = output.topk(5, dim=1)
    target_expanded = target.view(-1, 1).expand_as(top5)
    correct_top5 = (top5 == target_expanded).sum().item()
    return 100 * correct_top5 / target.size(0)

# Testing function
def test(model, loader):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)  # Compute test loss
            total_loss += loss.item()

            pred_top1 = outputs.argmax(dim=1)
            correct_top1 += (pred_top1 == y).sum().item()
            correct_top5 += compute_top5_accuracy(outputs, y) * y.size(0) / 100
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc_top1 = 100 * correct_top1 / total
    acc_top5 = 100 * correct_top5 / total
    return acc_top1, acc_top5, avg_loss

# Confidence Scores for MIA
def get_confidence_scores(model, dataloader):
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
            labels.extend([1] * x.size(0))  # Initially assume "members"
    return np.array(scores), np.array(labels)

# MIA Attack Success Computation
def compute_mia_attack(model):
    train_scores, train_labels = get_confidence_scores(model, trainloader)
    test_scores, test_labels = get_confidence_scores(model, testloader)
    test_labels[:] = 0  # Label test samples as "non-members"

    all_scores = np.concatenate([train_scores, test_scores])
    all_labels = np.concatenate([train_labels, test_labels])

    attack_model = LogisticRegression()
    all_scores = all_scores.reshape(-1, 1)
    attack_model.fit(all_scores, all_labels)

    predictions = attack_model.predict(all_scores)
    mia_accuracy = accuracy_score(all_labels, predictions)
    output_data["mia_attack_success_rate"] = mia_accuracy * 100
    print(f"MIA Attack Success Rate: {mia_accuracy * 100:.2f}%")

# Run the model
model = CIFARNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, trainloader, optimizer, epochs=10)
compute_mia_attack(model)

# Save results to JSON
with open("training_and_mia_results.json", "w") as json_file:
    json.dump(output_data, json_file, indent=4)

print("Results have been saved to 'training_and_mia_results.json'.")
