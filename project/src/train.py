import torch
import torch.optim as optim
import torch.nn as nn

def train_model(model, train_loader, device, num_epochs=20, learning_rate=0.001):
    # # class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            masks = masks.squeeze(1)  # Ensure the shape is correct

            # Forward pass
            outputs = model(images)

            # Resize outputs to match mask size
            # Ensure the output is resized back to the original image/mask size
            outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, masks.long())  # Use the weighted loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model
