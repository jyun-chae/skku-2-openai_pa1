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
│   │   ├── aug_samples/
│   │   │   ├── images/
│   │   │   │   ├── 00?.png
│   │   │   ├── masks/
│   │   │   │   ├── 00?.png
│   │   ├── aug_vis_samples/
│   │   │   ├── 00?.png
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
├── submit/
│   ├── img.zip -> not uploaded
├── main.ipynb
├── pyproject.toml
└── README.md
```

## 2. Environment and Dependencies

The project was designed to run on Google Colab with a T4 / L4 / A100 GPU.

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
check resume_path before running!

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

For backup, copy the selected checkpoint to:

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

Evaluation is not neccesary part
Skip eval when it is not needed
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

Place test images *.zip file in:

```text
submit/img/
```

Run unzip code before inference
It's placed inside main.ipynb

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

0. default prediction moves depend on resome_path and best.pth(if resome is none)

```yaml
checkpoint:
  resume_path: ""
```

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

For FLOP's measure we need ONNX

Make ONNX depends on checkpoints/best.pth

```python
from pathlib import Path
import os
import torch
import onnx
from google.colab import files

from src.models.build import build_model

ckpt_path = getattr(cfg.checkpoint, "resume_path", "")
if ckpt_path is None or ckpt_path == "":
    ckpt_path = os.path.join(cfg.checkpoint.save_dir, "best.pth")

ckpt_path = Path(ckpt_path)
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

model = build_model(cfg).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

onnx_path = Path("/content/project01/submit/model.onnx")
onnx_path.parent.mkdir(parents=True, exist_ok=True)

input_size = int(cfg.data.input_size)
dummy = torch.randn(1, 3, input_size, input_size, device=device)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)

print("ONNX saved:", onnx_path)
print("ONNX size MB:", onnx_path.stat().st_size / 1024 / 1024)

files.download(str(onnx_path))
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


## 12. Notes

- This project uses PyTorch and TorchVision.
- The backbone uses TorchVision ImageNet-1K classification pretrained EfficientNet-B3 weights.
- It does not use TorchVision pretrained semantic segmentation models.
- It does not use HuggingFace, TIMM, Albumentations, PyTorch Lightning, or Accelerate.
