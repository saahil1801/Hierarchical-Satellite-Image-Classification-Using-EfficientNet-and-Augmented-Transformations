import streamlit as st
from pathlib import Path, PurePath
import random
import matplotlib.pyplot as plt
from PIL import Image
import math
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List
from super_gradients.training import models
from torch.utils.data import DataLoader
import pathlib
import textwrap

class config:
    DOWNLOAD_DIR = 'Satellite'
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    TEST_DIR = 'data/test'
    INPUT_HEIGHT = 224
    INPUT_WIDTH = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names=['cloudy', 'desert', 'green_area', 'water']

# Load the best model
best_full_model = models.get('efficientnet_b0', num_classes=len(class_names), checkpoint_path='checkpoints/0_Baseline_Experiment/RUN_20240812_232436_358147/ckpt_best.pth')

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.CenterCrop((config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(size=(config.INPUT_HEIGHT, config.INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
])

# Function to display random images from the test set
def plot_random_test_images(model, num_images_to_plot=10):
    test_image_path_list = list(Path(config.TEST_DIR).glob("*/*.jpg"))
    test_image_path_sample = random.sample(population=test_image_path_list, k=num_images_to_plot)

    num_rows = int(np.ceil(num_images_to_plot / 5))
    fig, ax = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
    ax = ax.flatten()

    for i, image_path in enumerate(test_image_path_sample):
        pred_and_plot_image(model=model, image_path=image_path, class_names=class_names, subplot=(num_rows, 5, i+1), image_size=(config.INPUT_HEIGHT, config.INPUT_WIDTH))

    plt.subplots_adjust(wspace=1)
    st.pyplot(fig)

# Function to predict and plot image
def pred_and_plot_image(image_path: str, subplot: Tuple[int, int, int], class_names: List[str] = class_names, model: torch.nn.Module = best_full_model, image_size: Tuple[int, int] = (config.INPUT_HEIGHT, config.INPUT_WIDTH), transform: torchvision.transforms = None, device: torch.device=config.DEVICE):
    if isinstance(image_path, pathlib.PosixPath):
        img = Image.open(image_path).convert("RGB")
    else:
        img = Image.open(image_path).convert("RGB")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])
    transformed_image = transform(img)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = transformed_image.unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    ground_truth = PurePath(image_path).parent.name
    plt.subplot(*subplot)
    plt.imshow(img)
    title = f"Ground Truth: {ground_truth} | Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    plt.title("\n".join(textwrap.wrap(title, width=20)))
    plt.axis(False)

# Streamlit App
st.title("Satellite Image Classification")

# Upload an image for prediction
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    fig, ax = plt.subplots(figsize=(5, 5))
    pred_and_plot_image(image_path=uploaded_file, subplot=(1, 1, 1))
    st.pyplot(fig)