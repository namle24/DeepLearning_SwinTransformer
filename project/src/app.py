import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import SwinTransform
from utils import calculate_class_percentages, get_device

color_mapping = {
    (0, 0, 0): 0,  # Unknown
    (0, 255, 0): 1,  # Forest
    (255, 255, 0): 2,  # Agricultural
    (255, 0, 255): 3,  # Rangeland
    (255, 0, 0): 4,  # Urban
    (0, 0, 255): 5,  # Water
    (255, 255, 255): 6  # Barren
}

inverse_color_mapping = {v: k for k, v in color_mapping.items()}


def load_model(model_path, num_classes=7):
    device = get_device()
    model = SwinTransform(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def visualize_output(image, pred, inverse_color_mapping):
    height, width = pred.shape
    segmented_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in inverse_color_mapping.items():
        segmented_image[pred == class_id] = color

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title("Segmentation Result")
    plt.imshow(segmented_image)
    plt.show()
    return segmented_image


def predict_and_analyze(image, model, device, num_classes=7):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    class_percentages = calculate_class_percentages(pred, num_classes)

    return pred, class_percentages


st.title("Satellite Image Land Cover Segmentation")

uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Satellite Image', use_column_width=True)

    model, device = load_model('project/src/swin_model.pth')

    if st.button('Analyze Image'):
        st.write("Analyzing the image...")

        pred, results = predict_and_analyze(image, model, device)

        st.write("Land cover percentages in the image:")
        for class_name, percentage in results.items():
            st.write(f"{class_name}: {percentage:.2f}%")

        segmented_image = visualize_output(image, pred, inverse_color_mapping)
        st.image(segmented_image, caption='Segmented Image', use_column_width=True)
