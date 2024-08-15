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

class_names = ['cloudy', 'desert', 'green_area', 'water']

# Load the best model
best_full_model = models.get('efficientnet_b0', num_classes=len(class_names), checkpoint_path='checkpoints/0_Baseline_Experiment/RUN_20240812_232436_358147/ckpt_best.pth')

# Function to predict and plot image
def pred_and_plot_image(image: Image.Image, subplot: Tuple[int, int, int], class_names: List[str] = class_names, model: torch.nn.Module = best_full_model, image_size: Tuple[int, int] = (config.INPUT_HEIGHT, config.INPUT_WIDTH), transform: torchvision.transforms = None, device: torch.device=config.DEVICE):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

    transformed_image = transform(image)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        transformed_image = transformed_image.unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.subplot(*subplot)
    plt.imshow(image)
    title = f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    plt.title("\n".join(textwrap.wrap(title, width=20)))
    plt.axis(False)

# Streamlit App
st.title("Satellite Image Classification")

# Upload an image for prediction
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if st.button("Upload"):
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        with st.status("Classifying..."):
            fig, ax = plt.subplots(figsize=(5, 5))
            pred_and_plot_image(image=img, subplot=(1, 1, 1))
        st.pyplot(fig)
        
    