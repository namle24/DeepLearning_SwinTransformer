from dataset import get_dataloader
from model import SimpleSwinTransformer
import torch
from train import train_model
import time

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/images',
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/masks',
        batch_size=4
    )
    val_loader = get_dataloader(
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/images',
        '/home/namle/Desktop/DeepLearning_SwinTransformer/project/data/train/masks',
        batch_size=4
    )

    model = SimpleSwinTransformer(num_classes=7).to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=0.001)

if __name__ == "__main__":
    main()
