import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

if __name__ == "__main__":
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define data directories for train and validation
    train_data_dir = "train"
    val_data_dir = "test"

    # Define hyperparameters
    num_classes = 2  # Binary classification
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 16

    # Create data loaders for train and validation
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_dataset = torchvision.datasets.ImageFolder(root=val_data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define the model
    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, num_classes)

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print(f"[{epoch + 1}, {i + 1}], loss: {running_loss / 10:.3f}")
                running_loss = 0.0

        epoch_delta_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} finished in {epoch_delta_time:.2f} seconds")

    # Validation loop
    model.eval()
    y_true = []
    y_pred = []

    # with torch.no_grad():
    #     for data in val_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = model(images)
    #         _, predicted = torch.max(outputs, 1)

    #         y_true.extend(labels.cpu().numpy())
    #         y_pred.extend(predicted.cpu().numpy())

    # correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # print(f"F1Score: {f1score(y_true, y_pred):.4f}")

    # Save the trained model
    torch.save(model, 'binary_classification_model.pth')