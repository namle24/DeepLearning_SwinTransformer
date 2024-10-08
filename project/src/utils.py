import torch
import numpy as np

class_names = [
    'Unknown',
    'Forest',
    'Agricultural',
    'Rangeland',
    'Urban',
    'Water',
    'Barren'
]

def calculate_class_percentages(pred, n_classes=7):
    class_percentages = {}
    total_pixels = pred.size

    for cls in range(n_classes):
        class_pixels = (pred == cls).sum()
        class_percentages[class_names[cls]] = (class_pixels / total_pixels) * 100

    return class_percentages

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')



