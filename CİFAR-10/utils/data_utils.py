import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(mean, std, image_size=32):
    train_transform = transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),

        # PIL augmentations
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02
        ),

        # Convert PIL → Tensor
        transforms.ToTensor(),

        # Tensor augmentations (doğru yer!)
        transforms.RandomErasing(p=0.25),

        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_transform, test_transform

_DATASET_CACHE = {}  # prevents reloading CIFAR multiple times


def get_cifar10_datasets(root, mean, std, image_size=32, download=False):
    """
    CIFAR10 dataset loader with caching.
    BO sırasında her iterasyonda yeniden indirilmez.
    """

    global _DATASET_CACHE
    key = f"{root}_{image_size}"

    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    train_tf, test_tf = get_transforms(mean, std, image_size=image_size)

    train_dataset = datasets.CIFAR10(
        root=root, train=True, transform=train_tf, download=download
    )

    test_dataset = datasets.CIFAR10(
        root=root, train=False, transform=test_tf, download=download
    )

    _DATASET_CACHE[key] = (train_dataset, test_dataset)
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size, num_workers=4):
    """
    Optimized DataLoader with pinned memory + persistent_workers.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, test_loader
