import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = (preds == labels).sum().item()
    accuracy = corrects / len(labels) * 100
    return accuracy


def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_corrects = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            if labels.ndim > 1:
                labels = labels.squeeze()

            outputs = model(images)

            # Kiểm tra kích thước của outputs và labels
            print("Output shape:", outputs.shape)
            print("Labels shape:", labels.shape)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_corrects += (torch.max(outputs, 1)[1] == labels).sum().item()
            total_samples += len(labels)

        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_corrects / total_samples * 100
        train_losses.append(avg_loss)
        train_accuracies.append(avg_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%')

        model.eval()
        val_corrects = 0
        val_samples = 0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                if val_labels.ndim > 1:
                    val_labels = val_labels.squeeze()

                val_outputs = model(val_images)

                # Kiểm tra kích thước của val_outputs và val_labels
                print("Validation Output shape:", val_outputs.shape)
                print("Validation Labels shape:", val_labels.shape)

                val_corrects += (torch.max(val_outputs, 1)[1] == val_labels).sum().item()
                val_samples += len(val_labels)

        val_accuracy = val_corrects / val_samples * 100
        val_accuracies.append(val_accuracy)
        print(f'Validation Accuracy: {val_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total Training Time: {total_time:.2f} seconds')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='x')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    return model
