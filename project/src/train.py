import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time

def train_model(model, train_loader, device, num_epochs=5, learning_rate=0.001):
    class_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_pixels = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.squeeze(1)

            outputs = model(images)
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, masks.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Tính toán độ chính xác
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == masks).sum().item()
            total_pixels += masks.numel()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = (correct_predictions / total_pixels) * 100

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    end_time = time.time()  # Kết thúc thời gian
    total_time = end_time - start_time

    print(f"\nTotal Time: {total_time:.2f} giây")
    print(f"Final Acuracy: {accuracy:.2f}%")

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, marker='o', color='orange', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    return model

