import csv
import os

import cv2
from catalyst.utils import tracing
import torch
import numpy as np
from pathlib import Path

from Dataset_generator.masks_to_csv import get_centers
from WBnn.Data_loaders import show_examples, get_loaders
from WBnn.Album_augments import compose, \
                            post_transforms, \
                            pre_transforms

from torch.utils.data import DataLoader
from collections.abc import MutableMapping
from catalyst import utils


def test_model(path_data=None, log_cb=None, end_cb=None):
    if not path_data: return

    ROOT = Path(path_data)
    test_image_path = ROOT
    test_masks_path = ROOT / "train_masks"

    TEST_IMAGES = sorted(test_image_path.glob("*.JPG"))
    TEST_MASKS = sorted(test_masks_path.glob("*.JPG"))
    valid_transforms = compose([pre_transforms(), post_transforms()])

    num_workers: int = 4
    batch_size = 1
    new_path = os.path.join(os.path.abspath(os.getcwd()), '../WBnn/traced-forward.pth')
    print(f'{new_path=}')

    model = tracing.load_traced_model(new_path, device="cpu")
    if model is None:
        print('Model not loaded')
        exit(-1)

    loaders = get_loaders(
        test_mode = True,
        images=TEST_IMAGES,
        masks=TEST_MASKS,
        random_state=0,
        valid_size=0,
        train_transforms_fn=valid_transforms,
        batch_size=batch_size)

    out_file_name = 'bears.csv'
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    else:
        f = open(out_file_name, 'w', encoding='UTF8', newline='')
        f.close()

    f = open(out_file_name, 'a', encoding='UTF8', newline='')
    writer = csv.writer(f)

    for i in range(len(TEST_IMAGES)):
        batch = iter(loaders["test"]).next()
        model_input = batch["image"].to("cpu")
        out_batch = model(model_input)
        mask_ = torch.detach(out_batch).sigmoid()
        mask = utils.detach(mask_ > 0.05).astype("float")

        bb = np.asarray(utils.imread(TEST_IMAGES[i]))
        mm = cv2.resize(mask[0, 0], (bb.shape[1], bb.shape[0]))
        # show_examples(name=TEST_IMAGES[i],
        #               image=bb,
        #               mask=mm)
        get_centers(os.path.basename(TEST_IMAGES[i]), mm, writer)
        log_cb((i + 1) / len(TEST_IMAGES))
    f.close()
    end_cb()

if __name__ == '__main__':
    test_model(r'D:\Storage\HACKATON\Dataset_generator\test_dataset\train')
