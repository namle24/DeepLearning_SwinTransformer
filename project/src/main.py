import torch
from dataset import get_dataloader
from model import SwinTransform
from train import train_model
from utils import get_device

def main():
    device = get_device()

    train_loader = get_dataloader(
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/images',
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/masks'
    )

    model = SwinTransform(num_classes=7).to(device)

    print(f"Model is running on: {device}")

    try:
        images, masks = next(iter(train_loader))
        print(f"Input image size: {images.shape}")
        print(f"Input mask size: {masks.shape}")
    except StopIteration:
        print("DataLoader is empty.")
        return

    model = train_model(model, train_loader, device, num_epochs=20, learning_rate=0.00001)

    torch.save(model.state_dict(), '/home/namle/Desktop/DeepLearning_SwinTransformer/project/src/swin_model.pth')

if __name__ == "__main__":
    main()


