from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SEED = int(os.getenv("SEED", "42"))


def _resolve_num_workers(num_workers: int | None) -> int:
    """Resolve a safe dataloader worker count.

    Args:
        num_workers: Explicit worker count or ``None`` for automatic selection.

    Returns:
        Non-negative number of workers.
    """
    return max((os.cpu_count() or 2) - 2, 0) if num_workers is None else max(num_workers, 0)


def _default_train_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Create the default augmentation pipeline for training.

    Args:
        image_size: Target image size as ``(height, width)``.

    Returns:
        Torchvision transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
        ]
    )


def _default_eval_transform(image_size: tuple[int, int]) -> transforms.Compose:
    """Create the deterministic preprocessing pipeline for evaluation.

    Args:
        image_size: Target image size as ``(height, width)``.

    Returns:
        Torchvision transform pipeline.
    """
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])


def get_train_dataloader(
    train_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool = True,
    transform: Any | None = None,
    num_workers: int | None = None,
) -> DataLoader[Any]:
    """Build a dataloader from an ImageFolder training directory.

    Args:
        train_dir: Directory with class subfolders.
        image_size: Target image size as ``(height, width)``.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples.
        transform: Optional custom torchvision transform.
        num_workers: Optional dataloader worker count.

    Returns:
        Training dataloader.
    """
    dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform or _default_train_transform(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=_resolve_num_workers(num_workers),
    )


def get_test_dataloader(
    test_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    transform: Any | None = None,
    num_workers: int | None = None,
) -> DataLoader[Any]:
    """Build a deterministic dataloader from an ImageFolder test directory.

    Args:
        test_dir: Directory with class subfolders.
        image_size: Target image size as ``(height, width)``.
        batch_size: Number of samples per batch.
        transform: Optional custom torchvision transform.
        num_workers: Optional dataloader worker count.

    Returns:
        Test dataloader.
    """
    dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform or _default_eval_transform(image_size),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_resolve_num_workers(num_workers),
    )


def get_train_val_dataloaders(
    train_dir: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    val_split: float = 0.2,
    transform: Any | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Create train and validation dataloaders from one ImageFolder directory.

    Args:
        train_dir: Directory with class subfolders.
        image_size: Target image size as ``(height, width)``.
        batch_size: Number of samples per batch.
        val_split: Fraction of samples reserved for validation.
        transform: Optional training transform; validation stays deterministic.
        num_workers: Optional dataloader worker count.

    Returns:
        Tuple of ``(train_dataloader, validation_dataloader)``.

    Raises:
        ValueError: If ``val_split`` or dataset size is invalid.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=transform or _default_train_transform(image_size),
    )
    val_dataset = datasets.ImageFolder(root=train_dir, transform=_default_eval_transform(image_size))

    val_size = max(1, int(len(train_dataset) * val_split))
    train_size = len(train_dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training dataset must contain at least two samples.")

    indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    workers = _resolve_num_workers(num_workers)

    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=workers),
        DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=workers),
    )


get_train_val_dataloader = get_train_val_dataloaders
