import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

color_mapping = {
    (0, 0, 0): 0,  # Unknown
    (0, 255, 0): 1,  # Forest
    (255, 255, 0): 2,  # Agricultural
    (255, 0, 255): 3,  # Rangeland
    (255, 0, 0): 4,  # Urban
    (0, 0, 255): 5,  # Water
    (255, 255, 255): 6  # Barren
}


def rgb_to_class(mask):
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for color, class_id in color_mapping.items():
        class_mask[(mask == color).all(axis=-1)] = class_id

    return class_mask

class RemoteSensingDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(image_dir) if '_sat' in f])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        mask_file = img_file.replace('_sat', '_mask').replace('.jpg', '.png')

        img_path = os.path.join(self.image_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')

        mask = rgb_to_class(mask)

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.long)

        return image, mask


def get_dataloader(image_dir, mask_dir, batch_size=4, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Điều chỉnh độ sáng, tương phản
        transforms.ToTensor()
    ])

    dataset = RemoteSensingDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
