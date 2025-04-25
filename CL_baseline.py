import ssl
import urllib.request
import matplotlib.pyplot as plt 

import torchvision

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # Import Matplotlib

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
def train(model, loader, optimizer, epochs=10):
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
        train_losses.append(avg_loss)

        acc_top1, acc_top5, test_loss = test(model, testloader)
        test_accuracies.append(acc_top1)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Test Loss = {test_loss:.4f}, Top-1 = {acc_top1:.2f}%, Top-5 = {acc_top5:.2f}%")


# Top 5 Accuracy:
def compute_top5_accuracy(output, target):
    # Get top 5 predictions for each sample
    _, top5 = output.topk(5, dim=1)
    # Expand the target to match top5 shape
    target_expanded = target.view(-1, 1).expand_as(top5)
    # Count how many times the target label appears in the top 5
    correct_top5 = (top5 == target_expanded).sum().item()
    return 100 * correct_top5 / target.size(0)

# Testing function
def test(model, loader):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    total_loss = 0  # Add this to accumulate test loss
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


# Run
model = CIFARNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, trainloader, optimizer, epochs=10)

# Plot training loss vs epoch
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), train_losses, label="Training Loss", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)

# Plot test accuracy vs epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), test_accuracies, label="Test Accuracy", color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy Over Epochs")
plt.grid(True)

plt.tight_layout()
plt.show()

# Save model
torch.save(model.state_dict(), "centralized_cifar10_model.pth")


import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    # Unnormalize CIFAR-10 images
    img = img / 2 + 0.5  # Denormalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Visualize 4 random images from the test set
dataiter = iter(testloader)
images, labels = next(dataiter)

# Display imagesp
imshow(torchvision.utils.make_grid(images))

# Make predictions
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)
print(f'Predicted labels: {predicted}')

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
            labels.extend([1] * x.size(0))  # initially assume "members"
    return np.array(scores), np.array(labels)

# Get confidence scores
train_scores, train_labels = get_confidence_scores(model, trainloader)
test_scores, test_labels = get_confidence_scores(model, testloader)
test_labels[:] = 0  # label test samples as "non-members"

# Combine
all_scores = np.concatenate([train_scores, test_scores])
all_labels = np.concatenate([train_labels, test_labels])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

attack_model = LogisticRegression()
all_scores = all_scores.reshape(-1, 1)
attack_model.fit(all_scores, all_labels)

# Predict and evaluate attack success
predictions = attack_model.predict(all_scores)
mia_accuracy = accuracy_score(all_labels, predictions)
print(f"MIA Attack Success Rate: {mia_accuracy * 100:.2f}%")