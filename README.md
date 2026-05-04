# Project 01: Semantic Segmentation

Open-Source AI Practice Project 1 semantic segmentation submission.

This repository trains a CNN-based semantic segmentation model for Pascal-VOC style 21-class prediction. The final model uses a TorchVision EfficientNet-B3 ImageNet-1K classification pretrained backbone with an FPN-like neck and segmentation head.

## 1. Project Structure

```text
project01/
├── src/
│   ├── config/
│   │   ├── config.py
│   │   └── default.yaml
│   ├── data/
│   │   ├── build.py
│   │   ├── transforms.py
│   │   ├── voc.py
│   │   └── coco_voc.py
│   ├── engine/
│   │   ├── checkpoint.py
│   │   ├── evaluator.py
│   │   └── trainer.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── build.py
│   │   └── seg_model.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── metric.py
│   │   └── seed.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── checkpoints/
│   └── model.pth
├── submit/
│   ├── img/
│   └── pred/
├── main.ipynb
├── pyproject.toml
└── README.md
```

## 2. Environment and Dependencies

The project was designed to run on Google Colab with a T4 or L4 GPU.

Install dependencies:

```bash
pip install torch torchvision
pip install wandb pycocotools onnx pyyaml tqdm numpy pillow matplotlib
```

In Colab, `main.ipynb` installs the required extra packages with:

```python
!pip install -q wandb pycocotools onnx
```

## 3. Dataset

The project uses Pascal-VOC style semantic segmentation with 21 classes:

```text
20 foreground classes + 1 background class
```

The code supports:

- Pascal-VOC training splits through `torchvision.datasets.VOCSegmentation`
- MS-COCO additional training data converted to VOC-style classes through `src/data/coco_voc.py`

Important rule:

```text
Validation or test annotations are not used for training.
```

The validation dataset is only used for validation mIoU and checkpoint selection.

## 4. Model

The submitted model is built by:

```python
from src.models.build import build_model

model = build_model(cfg)
```

Model summary:

```text
Backbone:
  TorchVision EfficientNet-B3
  ImageNet-1K classification pretrained weights

Neck:
  FPN-like c2/c3/c4/c5 feature fusion
  c4 and c5 high-level features are compressed with depthwise separable blocks

Head:
  Lightweight depthwise separable segmentation head
  Dilation is used in the high-level/head blocks to increase receptive field

Output:
  Logits are resized back to the input image size
  Final prediction is saved as class-index PNG
```

The model is CNN-based and does not use RNN or Transformer layers.

## 5. Configuration

Main configuration file:

```bash
src/config/default.yaml
```

Load config in Python:

```python
from src.config.config import load_config

cfg = load_config("src/config/default.yaml")
```

Important fields to check before running:

```yaml
runtime:
  device: cuda
  seed: 42

data:
  input_size: 512
  ignore_index: 255

checkpoint:
  save_dir: checkpoints
  resume_path: ""

submit:
  img_dir: submit/img
  pred_dir: submit/pred
```

## 6. Training

### Option A: Run with Python

From the project root:

```bash
python -m src.train
```

During training, checkpoints are saved to `cfg.checkpoint.save_dir`.

Typical outputs:

```text
checkpoints/last.pth
checkpoints/best.pth
```

For final submission, copy the selected checkpoint to:

```text
checkpoints/model.pth
```

Example:

```bash
mkdir -p checkpoints
cp checkpoints/best.pth checkpoints/model.pth
```

### Option B: Run with Colab Notebook

Open and run:

```text
main.ipynb
```

Notebook execution order:

```text
1. Install dependencies
2. Clone repository and rename folder to project01
3. Mount Google Drive
4. Load config and initialize WandB
5. Prepare VOC / COCO data
6. Train
7. Evaluate
8. Prepare submit/img
9. Run inference
10. Zip predictions
11. Export ONNX if needed
```

## 7. Evaluation

Evaluate the best checkpoint on the validation split:

```bash
python -m src.eval
```

Or in Python:

```python
from src.config.config import load_config
from src.eval import main as eval_main

cfg = load_config("src/config/default.yaml")
result = eval_main(cfg)
print(result)
```

The metric is mIoU. Pixels with `ignore_index=255` are excluded from both the loss and mIoU computation.

## 8. Inference

Place test images in:

```text
submit/img/
```

Run inference:

```bash
python -m src.infer
```

Predictions are saved to:

```text
submit/pred/
```

The output filename matches the input filename stem:

```text
submit/img/0001.jpg  ->  submit/pred/0001.png
submit/img/0002.png  ->  submit/pred/0002.png
```

Each output PNG stores class indices in `[0, 20]`.

## 9. Reproduce Submitted Result

To reproduce the submitted prediction files:

1. Put the final checkpoint at:

```text
checkpoints/model.pth
```

2. Set `cfg.checkpoint.resume_path` in `src/config/default.yaml` to:

```yaml
checkpoint:
  resume_path: checkpoints/model.pth
```

3. Put test images in:

```text
submit/img/
```

4. Run:

```bash
python -m src.infer
```

5. Check prediction outputs in:

```text
submit/pred/
```

## 10. FLOPs Measurement

The project score considers FLOPs at input size:

```text
1 x 3 x 480 x 640
```

A grader-compatible way is to export the model to ONNX with input size `(1, 3, 480, 640)` and then count ONNX FLOPs.

Example ONNX export:

```python
from pathlib import Path
import torch

from src.config.config import load_config
from src.models.build import build_model

cfg = load_config("src/config/default.yaml")

model = build_model(cfg)
ckpt = torch.load("checkpoints/model.pth", map_location="cpu")

if "model" in ckpt:
    model.load_state_dict(ckpt["model"])
else:
    model.load_state_dict(ckpt)

model.eval()

onnx_path = "checkpoints/model.onnx"
Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

dummy = torch.randn(1, 3, 480, 640)

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=16,
    input_names=["input"],
    output_names=["output"],
)
```

If the standalone grader-compatible FLOPs counter is included as `tools/grader_flops.py`, run:

```bash
python tools/grader_flops.py checkpoints/model.onnx --verbose
```

If the FLOPs script is placed in the project root, run:

```bash
python grader_flops.py checkpoints/model.onnx --verbose
```

## 11. WandB

WandB is used for training monitoring.

The logger records:

```text
train/loss
val/loss
val/miou
learning rate
```

Before training, log in if needed:

```bash
wandb login
```

WandB settings are controlled by `src/config/default.yaml`.

## 12. Submission Checklist

Final zip structure:

```text
2025xxxxxx_project01.zip
├── src/
├── checkpoints/
│   └── model.pth
├── submit/
│   ├── img/
│   └── pred/
├── 2025xxxxx_project01_report.pdf
├── pyproject.toml
└── README.md
```

Before zipping, check:

```bash
ls checkpoints/model.pth
ls submit/pred
```

`submit/img/` should be included as an empty folder unless otherwise instructed.

## 13. Notes

- This project uses PyTorch and TorchVision.
- The backbone uses TorchVision ImageNet-1K classification pretrained EfficientNet-B3 weights.
- It does not use TorchVision pretrained semantic segmentation models.
- It does not use HuggingFace, TIMM, Albumentations, PyTorch Lightning, or Accelerate.
