from torchvision import datasets
from torch.utils.data import DataLoader
from config import Config
from transforms import train_transforms, val_transforms

def create_dataloaders(
    train_dir: str, 
    val_dir: str,
    test_dir: str,
    train_transform,
    val_transform,
    test_transform,
    batch_size: int, 
    num_workers: int=2
):
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
