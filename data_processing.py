from pathlib import Path
import shutil
import random
import matplotlib.pyplot as plt
from PIL import Image
import math

def split_image_folder(image_paths: list, folder: str):
    data_path = Path(folder)
    if not data_path.is_dir():
        data_path.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        full_path = Path(path)
        image_name = full_path.name
        label = full_path.parent.name
        label_folder = data_path / label

        if not label_folder.is_dir():
            label_folder.mkdir(parents=True, exist_ok=True)

        destination = label_folder / image_name
        shutil.copy(path, destination)

def examine_images(images: list):
    num_images = len(images)
    num_rows = int(math.ceil(num_images / 5))
    num_cols = 5
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 30), tight_layout=True)
    axs = axs.ravel()

    for i, image_path in enumerate(images[:num_images]):
        image = Image.open(image_path)
        label = Path(image_path).parent.name
        axs[i].imshow(image)
        axs[i].set_title(f"Pose: {label}", fontsize=40)
        axs[i].axis('off')
    plt.show()
