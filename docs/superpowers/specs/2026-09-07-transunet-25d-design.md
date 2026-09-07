# TransUNet 2D and 2.5D experiment pipeline

## Goal

Add a supervised binary lung-tumor segmentation experiment that supports one
implementation for both 2D and 2.5D operation. 2D uses one axial CT slice;
2.5D uses an odd-sized axial context window such as three or five slices and
predicts only the centre slice. All run-time settings are Python values in a
`CFG` dictionary at the beginning of train and inference scripts. No command
line arguments are used.

## Dataset and dataloaders

`data/transunet_dataset.py` will index cases named in the existing split files
under `data/config/`. For each case it reads paired PNGs from:

```
data/processed/<case_id>/images/<slice>.png
data/processed/<case_id>/labels/<slice>.png
```

The dataset returns a floating-point image tensor with `NUM_SLICES` channels,
a binary mask for the centre slice, the case ID, and the integer slice index.
Neighbour indices outside a volume are clamped to the first or last slice.
`NUM_SLICES` must be positive and odd; `1` implements 2D and values such as
`3` and `5` implement 2.5D.

The training transform uses Albumentations and applies every spatial operation
jointly to all context channels and the mask:

- horizontal flip, probability 0.5;
- affine rotation in [-15, 15] degrees, scale in [0.9, 1.1], and translation
  in [-10%, 10%] in each image dimension;
- light elastic deformation;
- gamma adjustment in [0.8, 1.2]; and
- contrast adjustment in [-15%, 15%].

Validation and inference do not augment. DataLoader parameters (batch sizes,
worker count, pin-memory, and seeds) are configuration values.

## Model and loss

`models/transunet.py` will provide a binary 2D TransUNet derived from the
provided reference: convolutional encoder with skip connections, ViT token
bottleneck, and upsampling decoder. The supplied implementation's attention
scaling and bottleneck shape assumptions will be corrected, and model settings
will be parameterized. `in_channels` equals `NUM_SLICES`; the final layer
returns one unnormalized logit per pixel for the centre-slice tumor mask.

`losses.py` will implement `BCEDiceLoss` as the sum of
`BCEWithLogitsLoss` and sigmoid Dice loss. `BCEWithLogitsLoss` is deliberately
used rather than applying `BCELoss` to a sigmoid output because it has the same
objective while avoiding numerical overflow.

## Training

`trains/train_transunet.py` contains a single top-level `CFG` dictionary with
paths, experiment name, `NUM_SLICES`, all model and data settings, optimizer,
number of epochs, checkpoint settings, W&B settings, and AMP setting. It will:

1. seed Python, NumPy, and PyTorch; save the resolved configuration;
2. build train/validation datasets and loaders from `train.txt` and `val.txt`;
3. train with AdamW and CUDA AMP when CUDA is available and `CFG["AMP"]` is
   enabled;
4. use a learning-rate schedule with linear warmup over the first 10% of total
   epochs, then cosine decay to `LR_MIN`;
5. evaluate each epoch, save `last.pt`, and replace `best.pt` only when
   `val_loss` decreases; and
6. stop after five consecutive epochs without a `val_loss` improvement.

W&B is optional: if disabled or unavailable, execution continues with console
logging. At epoch level, it logs exactly `train_loss`, `train_dice_2D`,
`train_dice_3D`, `val_loss`, `val_dice_2D`, and `val_dice_3D`, plus current
learning rate. Checkpoints include the model, optimizer, scheduler, scaler,
epoch, best validation loss, configuration, and model construction settings.

## Metrics

`utils/metrics.py` will contain pure NumPy metric helpers. Binary predictions
are formed with configurable sigmoid threshold, default 0.5.

Per-slice values are Dice, IoU, recall, precision, false-positive pixel count,
and false-negative pixel count. Dice, IoU, recall, and precision use an empty
set convention: both masks empty score 1; exactly one mask empty scores 0.

For each case, predictions are reconstructed in original slice order and
compared with the matching NIfTI mask from `data/nifti/<case_id>/mask.nii.gz`.
3D Dice and IoU are then computed for the complete binary volume. HD95 and
ASSD use physical spacing from that NIfTI mask. Surface distances are defined
only when both volumes contain foreground; such unavailable distances are
reported as `NaN` and omitted from the mean, with an availability count in the
report. This prevents arbitrary finite values from hiding failed empty-volume
predictions.

2D metrics are the arithmetic mean over all slices. 3D metrics are the
arithmetic mean over valid cases, so long scans do not receive extra weight.

## Inference

`inferences/infer_transunet.py` has its own top-level `CFG`, including
checkpoint path, split list, `NUM_SLICES`, device, batch size, threshold, and
output directory. It validates that checkpoint and inference channel/model
settings agree, produces a full binary prediction volume for each case, writes
it as NIfTI with the reference image geometry, and writes per-case plus summary
CSV/JSON reports with requested 2D and 3D metrics.

## Tests and validation

Focused pytest tests will cover context-window edge clamping and channel order,
joint augmentation output shape/binary masks, model output shape, BCE+Dice
loss behavior, schedule warmup/cosine endpoints, 2D empty-mask metrics, and
3D metric aggregation/spacing with small synthetic arrays. A CPU smoke test
will instantiate the model, pass synthetic `256 x 256` data through one
forward/backward step, and verify checkpoint-compatible output shape.

## Dependencies

`requirements.txt` will add the training/runtime packages required by this
pipeline: PyTorch, Albumentations, SciPy, and Weights & Biases. Existing
SimpleITK and Pillow remain responsible for NIfTI geometry and PNG loading.
