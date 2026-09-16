# The Segmentation Ceiling: Echocardiographic Ejection-Fraction Regression

Code for the paper *"The segmentation ceiling: why explicit left-ventricular masks
do not improve learned ejection-fraction regression."*

The repository contains the analysis, training, and figure-generation scripts used
in the paper. It is built on the [EchoNet-Dynamic](https://echonet.github.io/dynamic/)
dataset and a UniFormer-S video backbone.

## Data

Experiments use the publicly available **EchoNet-Dynamic** dataset, released under a
Stanford Research Use Agreement (<https://echonet.github.io/dynamic/>). The dataset
is **not** redistributed here; download it separately and point the scripts at your
local copy with `--data_dir`. Model checkpoints and generated data are likewise not
included.

## Environment

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, torchvision, numpy, pandas, scikit-learn, opencv-python,
matplotlib, tqdm.

## Repository contents

### Segmentation-ceiling analysis
- `measure_rho.py` — measures the within-patient ED/ES area-error correlation (rho)
  of the DeepLabV3 segmenter and reports the break-even area-error threshold.
- `make_ceiling_figure.py` — renders the closed-form segmentation-ceiling figure
  from the derived criterion and the measured rho.

### Training
- `echonet/utils/video_uniformer.py` — UniFormer-S EF regression training/eval.
- `echonet/utils/video_uniformer_area.py` — area-consistency auxiliary-task variants
  (per-bin and amplitude consistency).
- `echonet/utils/video_uniformer_nll.py` — heteroscedastic beta-NLL variant that
  predicts EF and its variance for per-prediction uncertainty.
- `models/uniformer.py` — UniFormer-S backbone with GroupNorm and 4-channel input.

### Segmentation masks
- `generate_masks_deeplabv3.py` — generates DeepLabV3 predicted LV masks used for the
  predicted-mask input channel.

### Uncertainty and figures
- `mc_dropout_recalib.py` — Monte-Carlo dropout uncertainty with variance recalibration.
- `bland_altman_from_clips.py` — Bland-Altman agreement plot from per-clip predictions.

## Reproducing the key results

Measure the segmentation-ceiling parameters (rho and the break-even threshold):

```bash
python3 measure_rho.py \
    --data_dir /path/to/EchoNet-Dynamic \
    --weights  /path/to/deeplabv3_resnet50.pt \
    --split test
```

Render the ceiling figure:

```bash
python3 make_ceiling_figure.py
```

Train the main EF regressor (EMA + augmentation):

```bash
python3 -m echonet.utils.video_uniformer \
    --data_dir /path/to/EchoNet-Dynamic \
    --output output/uniformer_ema_aug \
    --uniformer_weights /path/to/uniformer_small_k400_16x8.pth \
    --mask_source zero --augment --run_test
```

Train the heteroscedastic beta-NLL uncertainty variant:

```bash
python3 -m echonet.utils.video_uniformer_nll \
    --data_dir /path/to/EchoNet-Dynamic \
    --output output/uniformer_nll \
    --uniformer_weights /path/to/uniformer_small_k400_16x8.pth \
    --mask_source zero --augment --beta 0.5 --run_test
```

## Citation

If you use this code, please cite the paper (citation to be added upon publication).

## License

Code released for research use. The EchoNet-Dynamic dataset is governed by its own
Stanford Research Use Agreement and is not included in this repository.
