import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    eval_loss = total_loss / len(loader.dataset)
    eval_acc = accuracy_score(y_true, y_pred)
    eval_f1 = f1_score(y_true, y_pred, average="macro")

    return eval_loss, eval_acc, eval_f1, y_true, y_pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers
    )

    model = SimpleCNN(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "best_cnn_binary.pth")

    print("Device =", device)
    print("Classes =", class_names)
    print("Train images =", len(train_loader.dataset))
    print("Val images =", len(val_loader.dataset))
    print("Test images =", len(test_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss = {train_loss:.4f}, "
            f"Train Acc = {train_acc:.4f}, "
            f"Train Macro-F1 = {train_f1:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_acc:.4f}, "
            f"Val Macro-F1 = {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    test_precision = precision_score(y_true, y_pred, average="macro")
    test_recall = recall_score(y_true, y_pred, average="macro")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print("\nCNN Classification Result")
    print("Test Loss =", round(test_loss, 4))
    print("Test Accuracy =", round(test_acc, 4))
    print("Test Macro-F1 =", round(test_f1, 4))
    print("Test Precision =", round(test_precision, 4))
    print("Test Recall =", round(test_recall, 4))
    print("\nClassification Report")
    print(report)
    print("\nConfusion Matrix")
    print(cm)

    result_path = os.path.join(args.output_dir, "result_cnn_binary.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("CNN Classification Result\n")
        f.write(f"Classes = {class_names}\n\n")
        f.write(f"Test Loss = {test_loss:.4f}\n")
        f.write(f"Test Accuracy = {test_acc:.4f}\n")
        f.write(f"Test Macro-F1 = {test_f1:.4f}\n")
        f.write(f"Test Precision = {test_precision:.4f}\n")
        f.write(f"Test Recall = {test_recall:.4f}\n\n")
        f.write("Classification Report\n")
        f.write(report)
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))

    np.save(os.path.join(args.output_dir, "confusion_matrix.npy"), cm)

    print("\nSaved result file =", result_path)


if __name__ == "__main__":
    main()