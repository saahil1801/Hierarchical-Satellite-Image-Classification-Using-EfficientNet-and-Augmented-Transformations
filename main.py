from config import Config
from data_processing import split_image_folder, examine_images
from dataloaders import create_dataloaders
from model_training import train_model
from model_evaluation import plot_random_test_images
from utils import plot_image_counts 
from transforms import train_transforms, val_transforms, test_transforms
import random
import os
from pathlib import Path
from imutils import paths
from sklearn.model_selection import train_test_split

def main():
    # Load image paths and split them into train, validation, and test sets
    image_paths = list(sorted(paths.list_images(Config.DOWNLOAD_DIR)))
    class_names = [Path(x).parent.name for x in image_paths]
    train_paths, rest_of_paths = train_test_split(image_paths, stratify=class_names, test_size=0.15, shuffle=True, random_state=42)
    class_names_ = [Path(x).parent.name for x in rest_of_paths]
    val_paths, test_paths = train_test_split(rest_of_paths, stratify=class_names_, test_size=0.50, shuffle=True, random_state=42)
    
    # Split folders
    split_image_folder(train_paths, Config.TRAIN_DIR)
    split_image_folder(val_paths, Config.VAL_DIR)
    split_image_folder(test_paths, Config.TEST_DIR)
    
    # Examine some images
    train_image_path_list = list(Path(Config.TRAIN_DIR).glob("*/*.jpg"))
    train_image_path_sample = random.sample(population=train_image_path_list, k=20)
    examine_images(train_image_path_sample)
    
    # Plot image counts
    plot_image_counts(Path(Config.TRAIN_DIR))
    
    # Create dataloaders
    train_dataloader, valid_dataloader, test_dataloader, class_names = create_dataloaders(
        train_dir=Config.TRAIN_DIR,
        val_dir=Config.VAL_DIR,
        test_dir=Config.TEST_DIR,
        train_transform=train_transforms,
        val_transform=val_transforms,
        test_transform=val_transforms,
        batch_size=16
    )
    
    # Train model
    best_full_model, full_model_trainer = train_model(train_dataloader, valid_dataloader, class_names)
    
    # Evaluate model
    plot_random_test_images(best_full_model)

if __name__ == "__main__":
    main()
