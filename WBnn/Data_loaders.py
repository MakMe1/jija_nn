from pathlib import Path
from typing import List
from torch.utils.data import Dataset
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
import numpy as np
from skimage.io import imread as gif_imread
from catalyst import utils

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(30, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")
    plt.show()


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = utils.imread(image_path)
    mask = utils.imread(masks[index])

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)


def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)


def get_loaders(
                images: List[Path],
                masks: List[Path],
                random_state: int,
                valid_size: float = 0.2,
                batch_size: int = 32,
                num_workers: int = 4,
                test_mode = False,
                train_transforms_fn=None,
                valid_transforms_fn=None) -> dict:

    if not test_mode:
        indices = np.arange(len(images))
        # Let's divide the data set into train and valid parts.
        train_indices, valid_indices = train_test_split(
            indices, test_size=valid_size, random_state=random_state, shuffle=True)

        np_images = np.array(images)
        np_masks = np.array(masks)

        # Creates our train dataset
        train_dataset = SegmentationDataset(
            images=np_images[train_indices].tolist(),
            masks=np_masks[train_indices].tolist(),
            transforms=train_transforms_fn)

        # Creates our valid dataset
        valid_dataset = SegmentationDataset(
            images=np_images[valid_indices].tolist(),
            masks=np_masks[valid_indices].tolist(),
            transforms=valid_transforms_fn
        )

        # Catalyst uses normal torch.data.DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

        # And excpect to get an OrderedDict of loaders
        loaders = collections.OrderedDict()
        loaders["train"] = train_loader
        loaders["valid"] = valid_loader

    else:
        np_images = np.array(images)
        np_masks = np.array(masks)

        # Creates our valid dataset
        test_dataset = SegmentationDataset(
            images=np_images.tolist(),
            masks=np_masks.tolist(),
            transforms=train_transforms_fn
        )

        # Catalyst uses normal torch.data.DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
        )

        # And excpect to get an OrderedDict of loaders
        loaders = collections.OrderedDict()
        loaders["test"] = test_loader

    return loaders


class SegmentationDataset(Dataset):
    def __init__(
            self,
            images: List[Path],
            masks: List[Path] = None,
            transforms=None) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = utils.imread(image_path)

        result = {"image": image}

        if self.masks is not None and len(self.masks) != 0:
            mask = utils.imread(self.masks[idx])
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result
