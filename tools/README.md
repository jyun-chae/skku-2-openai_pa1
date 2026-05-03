# tools/

Standalone utilities — copy/share without bringing in the rest of the project.

## `grader_flops.py` — ONNX FLOPs counter (grader-compatible)

Reproduces SKKU 2026-1 OSAIP Project 1 grader's FLOPs measurement. Calibrated
against the actual grader output, byte-level match within 0.02% across 3
architectures.

### Quick usage

```bash
# Just count FLOPs of an ONNX file
pip install onnx
python grader_flops.py model.onnx

# With per-op breakdown
python grader_flops.py model.onnx --verbose

# Check against a budget (exits non-zero if over)
python grader_flops.py model.onnx --budget 15

# Regression self-check (requires torch — verifies the counter still matches
# its known reference value on a toy 2-conv net)
python grader_flops.py --selfcheck
```

### Counting conventions

| Op                                               | FLOPs                              |
|--------------------------------------------------|------------------------------------|
| `Conv` / `Gemm` / `MatMul`                       | 2 × MAC (FMA convention)           |
| `BatchNormalization`                             | 2 × numel(output) (folded)         |
| `Relu` / `Sigmoid` / `HardSigmoid` / `HardSwish` / `Tanh` / `Softmax` / `LeakyRelu` / `Elu` | 1 × numel(output) |
| `Add` / `Sub` / `Mul` / `Div` / `Pow` / `Sqrt` / `Exp` / `Log` / `Neg` / `Abs` / `Clip` / `Min` / `Max` | 1 × numel(output) |
| `AveragePool` / `GlobalAveragePool`              | kH × kW × numel(output)            |
| `MaxPool`                                        | (kH × kW − 1) × numel(output)      |
| `Resize` / `Upsample` / `Concat` / `Identity` / `Constant` / `Shape` / `Slice` / `Cast` / `Reshape` / `Transpose` / `Unsqueeze` / `Squeeze` / `Gather` / `Split` / `Flatten` / `Expand` / `Pad` | 0 |

Unknown ops silently count as 0 (matches grader behavior — both sides skip
ops they don't recognize).

### From a PyTorch checkpoint

The grader exports your PyTorch ckpt to ONNX first, then counts. To reproduce
the full pipeline yourself:

```python
import torch
import grader_flops

# Build / load your model
from my_module import MyNet
model = MyNet()
model.load_state_dict(torch.load("checkpoints/best.pth")["model"])

# Export to ONNX
grader_flops.export_pytorch_model_to_onnx(
    model,
    input_size=(1, 3, 480, 640),  # grader's input shape for VOC
    out_path="submission.onnx",
)

# Count
total, breakdown = grader_flops.count_onnx_flops("submission.onnx", verbose=True)
print(f"{total/1e9:.3f} GFLOPs")
```

### Sharing

This file is **public-domain (CC0)** — copy it into any project, rename, or
adapt freely. Single dependency: `onnx`. PyTorch only needed if you want
the `--selfcheck` mode or `export_pytorch_model_to_onnx` helper.

### Source

Adapted from `src/utils/onnx_flops.py` of
<https://github.com/BetaTester772/Pascal-VOC-Segmentation>. The full project
version has additional Hydra integration for in-project measurement; this
standalone version drops that for portability.
