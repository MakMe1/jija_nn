import albumentations as albu
from albumentations.pytorch import ToTensor


def pre_transforms(image_size=224*3):
    return [albu.Resize(image_size, image_size, p=1)]


def hard_transforms():
    result = [
        albu.RandomRotate90(),
        albu.Cutout(),
        albu.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.3
        ),
        albu.GridDistortion(p=0.3),
        albu.HueSaturationValue(p=0.3)
    ]

    return result


def resize_transforms(image_size=224*3):
    BORDER_CONSTANT = 0
    #rescale = albu.Compose([albu.Resize(image_size, image_size, p=1)])
    # Converts the image to a square of size image_size x image_size
    return albu.Compose([albu.Resize(image_size, image_size, p=1)])


def post_transforms():
    # we use ImageNet image normalization
    # and convert it to torch.Tensor
    return [albu.Normalize(), ToTensor()]


def compose(transforms_to_compose):
    # combine all augmentations into single pipeline
    result = albu.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result