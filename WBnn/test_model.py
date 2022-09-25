import cv2
from catalyst.utils import tracing
import torch
import numpy as np
from pathlib import Path
from WBnn.Data_loaders import show_examples, get_loaders
from WBnn.Album_augments import compose, \
                            post_transforms, \
                            pre_transforms

from torch.utils.data import DataLoader
from catalyst import utils


def test_model(path_data=None):
    if not path_data: return
    # path_data = r'C:\Users\Nyite\Desktop\WB_test\dataset\arctic_with_bears'
    ROOT = Path(path_data)
    test_image_path = ROOT / "train"
    test_masks_path = ROOT / "train_masks"

    TEST_IMAGES = sorted(test_image_path.glob("*.jpg"))
    TEST_MASKS = sorted(test_masks_path.glob("*.png"))
    valid_transforms = compose([pre_transforms(), post_transforms()])

    num_workers: int = 4
    batch_size = 1

    model = tracing.load_traced_model('WBnn/traced-forward.pth', device="cpu")
    if model is None:
        print('Model not loaded')
        exit(-1)

    loaders = get_loaders(
        # test_mode = True,
        images=TEST_IMAGES,
        masks=TEST_MASKS,
        random_state=0,
        valid_size=0.05,
        train_transforms_fn=valid_transforms,
        batch_size=batch_size)

    for i in range(len(TEST_IMAGES)):
        batch = iter(loaders["train"]).next()
        model_input = batch["image"].to("cpu")
        out_batch = model(model_input)
        mask_ = torch.detach(out_batch).sigmoid()
        mask = utils.detach(mask_ > 0.5).astype("float")

        bb = np.asarray(utils.imread(TEST_IMAGES[i]))
        mm = cv2.resize(mask[0, 0], (bb.shape[1], bb.shape[0]))
        show_examples(name=TEST_IMAGES[i],
                      image=bb,
                      mask=mm)

