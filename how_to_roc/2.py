import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = torch.load("binary_classification_model.pth")
    model.classifier[6] = nn.Linear(4096, 2)
    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root="test", transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
# Initialize lists to store true labels and model's score
    true_labels = []
    model_scores = []

    # Evaluate the model
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            sm = torch.nn.Softmax(dim=1)  # Create a Softmax 
            probabilities = sm(outputs)  # Get the probability of each class
            
            true_labels.extend(labels.cpu().numpy())
            model_scores.extend(probabilities[:, 1].cpu().numpy())  # Get the score of the positive class

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(true_labels, model_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()