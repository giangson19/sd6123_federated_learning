import ssl
import urllib.request
import matplotlib.pyplot as plt 
import os
import glob
import torchvision

ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # Import Matplotlib
from model import CIFARNet  # Assuming CIFARNet is defined in model.py
from data import load_cifar10, create_dataloaders  # Assuming data loading functions are in data.py
import numpy as np
import matplotlib.pyplot as plt

# Loading trained global model
strategy = "fedavg"  # or "fedprox" or "fedavgm"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFARNet().to(DEVICE)
list_of_files = [fname for fname in glob.glob(f"models/{strategy}/model_round_*")]
latest_round_file = max(list_of_files, key=os.path.getctime)
print("Loading pre-trained model from: ", latest_round_file)
state_dict = torch.load(latest_round_file)
model.load_state_dict(state_dict)
model.eval()


# load data
trainset, testset = load_cifar10(root_path="./data")
trainloader, testloader = create_dataloaders(trainset, testset, batch_size=64)


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
outputs = model(images.to(DEVICE))
_, predicted = torch.max(outputs, 1)
print(f'Predicted labels: {predicted}')

def get_confidence_scores(model, dataloader):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
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