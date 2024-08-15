from torchvision import transforms
from config import Config

# Define transformations
train_transforms = transforms.Compose([
    transforms.Resize(size=(Config.INPUT_HEIGHT, Config.INPUT_WIDTH)),
    transforms.CenterCrop((Config.INPUT_HEIGHT, Config.INPUT_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomCrop(size=(Config.INPUT_HEIGHT, Config.INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize(size=(Config.INPUT_HEIGHT, Config.INPUT_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
])

test_transforms = val_transforms  # Use the same transforms as validation
