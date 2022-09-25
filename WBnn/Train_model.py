import numpy as np
import os
import torch
from pathlib import Path
from Data_loaders import show_examples, show_random, get_loaders, SegmentationDataset
from Album_augments import compose, \
                            resize_transforms, \
                            hard_transforms, \
                            post_transforms, \
                            pre_transforms
from catalyst.utils import tracing
from torch.utils.data import DataLoader
import catalyst
from catalyst import utils
from torch import nn

from torch import optim
from catalyst.contrib.nn import RAdam, Lookahead
import segmentation_models_pytorch as smp
from torch import nn
from catalyst.contrib.nn import DiceLoss, IoULoss
from catalyst.dl import SupervisedRunner
from catalyst.dl import DiceCallback, IouCallback, \
  CriterionCallback, MetricAggregationCallback
from catalyst.contrib.callbacks import DrawMasksCallback


if __name__ == '__main__':

    # this variable will be used in `runner.train` and by default we disable FP16 mode
    is_fp16_used = False
    print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU

    SEED = 42
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True)

    path_data = r'C:\Users\Nyite\Desktop\WB_test\test_dataset_1000'
    ROOT = Path(path_data)

    train_image_path = ROOT / "train"
    train_mask_path = ROOT / "train_mask"
    test_image_path = ROOT / "test"

    ALL_IMAGES = sorted(train_image_path.glob("*.jpg"))
    print(len(ALL_IMAGES))
    ALL_MASKS = sorted(train_mask_path.glob("*.jpg"))
    print(len(ALL_MASKS))

    #show_random(ALL_IMAGES, ALL_MASKS, )

    if is_fp16_used:
        batch_size = 64
    else:
        batch_size = 1

    train_transforms = compose([
        resize_transforms(),
        hard_transforms(),
        post_transforms()
    ])
    valid_transforms = compose([pre_transforms(), post_transforms()])

    #show_transforms = compose([resize_transforms(), hard_transforms()])

    #show_random(ALL_IMAGES, ALL_MASKS, transforms=show_transforms)

    print(f"batch_size: {batch_size}")

    loaders = get_loaders(
        images=ALL_IMAGES,
        masks=ALL_MASKS,
        random_state=SEED,
        train_transforms_fn=train_transforms,
        valid_transforms_fn=valid_transforms,
        batch_size=batch_size)

    # --------------------- MODEL CONFIG PATH -------------------------------
    # We will use Feature Pyramid Network with pre-trained ResNeXt50 backbone
    model = smp.Unet(encoder_name="efficientnet-b3", classes=1)

    # we have multiple criterions
    criterion = {
        "dice": DiceLoss(),
        "iou": IoULoss(),
        "bce": nn.BCEWithLogitsLoss()}

    learning_rate = 0.001
    encoder_learning_rate = 0.0005

    # Since we use a pre-trained encoder, we will reduce the learning rate on it.
    layerwise_params = {"encoder*": dict(lr=encoder_learning_rate, weight_decay=0.00003)}

    # This function removes weight_decay for biases and applies our layerwise_params
    model_params = utils.process_model_params(model, layerwise_params=layerwise_params)

    # Catalyst has new SOTA optimizers out of box
    base_optimizer = RAdam(model_params, lr=learning_rate, weight_decay=0.0003)
    optimizer = Lookahead(base_optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    num_epochs = 10
    logdir = "./logs/segmentation"
    device = utils.get_device()
    print(f"device: {device}")

    if is_fp16_used:
        fp16_params = dict(opt_level="O1")  # params for FP16
    else:
        fp16_params = None
    print(f"FP16 params: {fp16_params}")
    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    # runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")
    # ----------------------------------------------------------------------------------

    # ------------------- MODEL TRAINING -----------------------------------------------

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
        # visualization
        DrawMasksCallback(output_key='logits',
                          input_image_key='image',
                          input_mask_key='mask',
                          summary_step=50
                          )
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=fp16_params,
        # prints train logs
        verbose=True,
    )
    # ----------------------------------------------------------------------


    # -------------------------- MODEL INFERENCE ---------------------------

    TEST_IMAGES = sorted(test_image_path.glob("*.jpg"))

    # create test dataset
    test_dataset = SegmentationDataset(
        TEST_IMAGES,
        transforms=valid_transforms)

    num_workers: int = 4

    infer_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)

    # this get predictions for the whole loader
    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(loader=infer_loader, resume=f"{logdir}/checkpoints/best.pth")
    )))

    print(type(predictions))
    print(predictions.shape)

    threshold = 0.5
    max_count = 5

    for i, (features, logits) in enumerate(zip(test_dataset, predictions)):
        image = utils.tensor_to_ndimage(features["image"])

        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype("float")

        show_examples(name="", image=image, mask=mask)

        if i >= max_count:
            break

    batch = next(iter(loaders["valid"]))
    # saves to `logdir` and returns a `ScriptModule` class
    runner.trace(model=model, batch=batch, logdir=logdir, fp16=is_fp16_used)

    # ---------------------------------------------------------------------

    if is_fp16_used:
        model = tracing.load_traced_model(
            f"{logdir}/trace/traced-forward-opt_O1.pth",
            device="cuda",
            opt_level="O1"
        )
    else:
        model = tracing.load_traced_model(
            f"{logdir}/trace/traced-forward.pth",
            device="cpu"
        )

    model_input = batch["image"].to("cuda" if is_fp16_used else "cpu")
    out_batch = model(model_input)