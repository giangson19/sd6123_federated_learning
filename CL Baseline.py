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

trainset = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
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

        # Evaluate test accuracy
        accuracy = test(model, testloader)
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%")

# Testing function
def test(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    accuracy = 100 * correct / total
    return accuracy

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