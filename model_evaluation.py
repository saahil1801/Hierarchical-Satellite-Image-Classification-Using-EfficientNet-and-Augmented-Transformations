import requests
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import pathlib
from typing import List, Tuple
from PIL import Image
import textwrap
import torch
import random
import torchvision
from config import Config
import numpy as np


def pred_and_plot_image(image_path: str, subplot: Tuple[int, int, int], class_names: list, model: torch.nn.Module, image_size: Tuple[int, int] = (Config.INPUT_HEIGHT, Config.INPUT_WIDTH), transform: torchvision.transforms = None, device: torch.device = Config.DEVICE):
    if isinstance(image_path, pathlib.PosixPath):
        img = Image.open(image_path).convert("RGB")
    else: 
        img = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
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
    if isinstance(image_path, pathlib.PosixPath):
        title = f"Ground Truth: {ground_truth} | Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    else:
        title = f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    plt.title("\n".join(textwrap.wrap(title, width=20)))
    plt.axis(False)

def plot_random_test_images(model):
    num_images_to_plot = 30
    test_image_path_list = list(Path(Config.TEST_DIR).glob("*/*.jpg"))
    test_image_path_sample = random.sample(population=test_image_path_list, k=num_images_to_plot)

    num_rows = int(np.ceil(num_images_to_plot / 5))
    fig, ax = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
    ax = ax.flatten()
    class_names=['cloudy', 'desert', 'green_area', 'water']
    for i, image_path in enumerate(test_image_path_sample):
        pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            subplot=(num_rows, 5, i+1),
                            image_size=(Config.INPUT_HEIGHT, Config.INPUT_WIDTH))

    plt.subplots_adjust(wspace=1)
    plt.show()
