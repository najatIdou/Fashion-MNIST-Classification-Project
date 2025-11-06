import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

def train_resnet50(X_train, y_train, X_test, y_test):
    # Convert data to PyTorch tensors and reshape to match ResNet input requirements
    X_train = torch.tensor(X_train).float().permute(0, 3, 1, 2) # Convert to (channels, height, width)
    X_test = torch.tensor(X_test).float().permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    # Create DataLoaders for batch processing
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)

    # Load pre-trained ResNet-50 model and modify the final layer
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Dropout for regularization
        nn.Linear(num_features, 10) # Output layer for 10 classes
    )  # Adjust for 10 classes

    # Freeze all layers except the last fully connected (FC) layer and layer4
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Add LR scheduler

    # Training and validation loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(10):  # Train for 10 epochs
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()  # Adjust learning rate after each epoch

    # Evaluate on test data
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

    return model, all_preds
