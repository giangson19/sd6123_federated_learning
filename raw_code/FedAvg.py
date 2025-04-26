import ssl
import urllib.request
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

# Track metrics
train_losses = []
test_accuracies = []

# Training function
def train(model, loader, optimizer, epochs=10, round_number=0):
    with open("output.txt", "a") as f:
        f.write(f"Round {round_number}:\n")
        
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_end_time = time.time()
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        acc_top1, acc_top5, test_loss = test(model, testloader)
        test_accuracies.append(acc_top1)

        # Logging metrics to the file
        with open("output.txt", "a") as f:
            f.write(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Test Loss = {test_loss:.4f}, "
                    f"Top-1 Accuracy = {acc_top1:.2f}%, Top-5 Accuracy = {acc_top5:.2f}%\n")
            f.write(f"Client Training Time (Epoch): {epoch_end_time - start_time:.2f}s\n")

# Top 5 Accuracy
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
            loss = F.cross_entropy(outputs, y)
            total_loss += loss.item()

            pred_top1 = outputs.argmax(dim=1)
            correct_top1 += (pred_top1 == y).sum().item()
            correct_top5 += compute_top5_accuracy(outputs, y) * y.size(0) / 100
            total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc_top1 = 100 * correct_top1 / total
    acc_top5 = 100 * correct_top5 / total
    return acc_top1, acc_top5, avg_loss

# Run
model = CIFARNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Writing initial header
with open("output.txt", "w") as f:
    f.write("Federated Learning Training and Evaluation Report\n")
    f.write("--------------------------------------------------\n\n")

# Run federated learning rounds
for round_number in range(1, 11):  # 10 rounds
    train(model, trainloader, optimizer, epochs=1, round_number=round_number)

# Final Evaluation
with open("output.txt", "a") as f:
    f.write("\nFinal Evaluation (After 10 Rounds):\n")
    acc_top1, acc_top5, avg_loss = test(model, testloader)
    f.write(f"Top-1 Accuracy (Central): {acc_top1:.2f}%\n")
    f.write(f"Top-5 Accuracy (Central): {acc_top5:.2f}%\n")
    f.write(f"Cross-Entropy Loss (Central): {avg_loss:.2f}\n")

# Save model
torch.save(model.state_dict(), "centralized_cifar10_model.pth")

print("Training complete, metrics saved to 'output.txt'")
