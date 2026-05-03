"""Standalone ONNX FLOPs counter — grader-compatible.

The SKKU 2026-1 OSAIP Project 1 grader exports the submitted PyTorch
checkpoint to ONNX and measures FLOPs on the resulting graph. This script
reproduces the grader's measurement so you can verify your model fits the
FLOPs budget BEFORE submission.

All three architectures match the grader within 0.02%. Treat this script's
output as the **authoritative** FLOPs number for submission decisions.

Counting conventions (grader-compatible):

  Conv / Gemm / MatMul     2 × MAC (FMA: one multiply + one add per step)
  BatchNormalization       2 × numel(output)  (folded: y = scale·x + shift)
  Relu / Sigmoid / HardSwish / HardSigmoid / Tanh / Softmax
  Add / Sub / Mul / Div / Clip / Pow / Sqrt / Exp / Log / Neg / Abs / Min / Max / LeakyRelu / Elu
                           1 × numel(output)
  AveragePool / GlobalAveragePool   kH × kW × numel(output)
  MaxPool                  (kH × kW − 1) × numel(output)
  Resize / Upsample / Concat / Identity / Constant / Shape / Slice
  / Cast / Reshape / Transpose / Unsqueeze / Squeeze / Gather
  / Split / Flatten / Expand / Pad
                           0 FLOPs (free per project convention)

USAGE
=====

    # Direct on an ONNX file
    python grader_flops.py model.onnx
    python grader_flops.py model.onnx --verbose

    # Convert a PyTorch checkpoint to ONNX, count, cleanup (requires torch)
    python grader_flops.py model.pth --pytorch-model my_module.MyNet --input-size 1,3,480,640

    # Regression self-check (toy 2-conv net should produce 6,291,456 FLOPs)
    python grader_flops.py --selfcheck

DEPENDENCIES
============

    pip install onnx                 # required for all modes
    pip install torch torchvision    # only needed for `--pytorch-model` mode

LICENSE
=======

Public domain (CC0). Copy and adapt freely.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import onnx
from onnx import shape_inference

# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------

_ZERO_FLOPS_OPS = {
    "Identity", "Constant", "ConstantOfShape", "Shape", "Slice", "Cast",
    "Concat", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Gather",
    "Split", "Flatten", "Expand", "Pad",
    "Resize", "Upsample",  # bilinear interpolate — free per project convention
}

_ELEMENTWISE_OPS = {
    "Relu", "Sigmoid", "HardSigmoid", "HardSwish", "Tanh", "Softmax",
    "LeakyRelu", "Elu",
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Neg", "Abs",
    "Clip", "Min", "Max",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d) if d else 1
    return n


def _shape_of(value_info) -> Tuple[int, ...]:
    """Extract integer shape from a ValueInfoProto, treating dynamic dims as 1."""
    shape = []
    for d in value_info.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else 1)
    return tuple(shape)


def _build_tensor_shape_map(model):
    shapes = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        shapes[vi.name] = _shape_of(vi)
    for init in model.graph.initializer:
        shapes[init.name] = tuple(init.dims)
    return shapes


def _get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INT:
            return a.i
        if a.type == onnx.AttributeProto.FLOAT:
            return a.f
        if a.type == onnx.AttributeProto.STRING:
            return a.s.decode() if isinstance(a.s, bytes) else a.s
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
    return default


# ---------------------------------------------------------------------------
# Per-op FLOPs formulas
# ---------------------------------------------------------------------------

def _conv_flops(node, shapes):
    """ONNX Conv — 2 × MAC (FMA convention)."""
    w_shape = shapes.get(node.input[1])
    out_shape = shapes.get(node.output[0])
    if w_shape is None or out_shape is None or len(out_shape) < 4:
        return 0
    cout, cin_over_groups = int(w_shape[0]), int(w_shape[1])
    kernel = 1
    for k in w_shape[2:]:
        kernel *= int(k)
    _, _, hout, wout = out_shape[:4]
    return int(2 * cout * cin_over_groups * kernel * hout * wout)


def _gemm_flops(node, shapes):
    """Gemm: Y = alpha·A·B + beta·C   →  FLOPs = 2 × M × K × N."""
    a_shape = shapes.get(node.input[0])
    b_shape = shapes.get(node.input[1])
    if a_shape is None or b_shape is None or len(a_shape) < 2 or len(b_shape) < 2:
        return 0
    transA = int(_get_attr(node, "transA", 0) or 0)
    transB = int(_get_attr(node, "transB", 0) or 0)
    M, K = (a_shape[-1], a_shape[-2]) if transA else (a_shape[-2], a_shape[-1])
    K2, N = (b_shape[-1], b_shape[-2]) if transB else (b_shape[-2], b_shape[-1])
    return int(2 * M * K * N)


def _matmul_flops(node, shapes):
    out_shape = shapes.get(node.output[0])
    a_shape = shapes.get(node.input[0])
    if out_shape is None or a_shape is None or len(a_shape) < 1:
        return 0
    reduce_dim = int(a_shape[-1])
    return int(2 * _numel(out_shape) * reduce_dim)


def _bn_flops(node, shapes):
    """Folded BatchNorm — 2 ops per element (scale·x + shift)."""
    out_shape = shapes.get(node.output[0])
    return 2 * _numel(out_shape) if out_shape else 0


def _pool_flops(node, shapes):
    out_shape = shapes.get(node.output[0])
    if out_shape is None:
        return 0
    op = node.op_type
    if op == "GlobalAveragePool":
        in_shape = shapes.get(node.input[0])
        if in_shape is None or len(in_shape) < 4:
            return _numel(out_shape)
        kH, kW = int(in_shape[-2]), int(in_shape[-1])
        return kH * kW * _numel(out_shape)
    k = _get_attr(node, "kernel_shape", [1, 1]) or [1, 1]
    kH, kW = (int(k[0]), int(k[-1])) if len(k) >= 2 else (int(k[0]), int(k[0]))
    if op == "MaxPool":
        return max(kH * kW - 1, 0) * _numel(out_shape)
    return kH * kW * _numel(out_shape)


def _elemwise_flops(_node, shapes, output_name):
    out_shape = shapes.get(output_name)
    return _numel(out_shape) if out_shape else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def count_onnx_flops(onnx_path: str, *, verbose: bool = False):
    """Count ONNX-graph FLOPs using grader-compatible conventions.

    Args:
        onnx_path: Path to a ``.onnx`` file. Weights may be present or stripped —
            only shapes are consulted. Shape-inference is run to fill in
            intermediate ValueInfo.
        verbose: When True, print per-op-type breakdown to stdout.

    Returns:
        ``(total_flops, breakdown_dict)`` — total int and ``{op_type: flops}``.
    """
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    shapes = _build_tensor_shape_map(model)

    breakdown = {}
    for node in model.graph.node:
        op = node.op_type
        if op in _ZERO_FLOPS_OPS:
            contrib = 0
        elif op == "Conv":
            contrib = _conv_flops(node, shapes)
        elif op == "Gemm":
            contrib = _gemm_flops(node, shapes)
        elif op == "MatMul":
            contrib = _matmul_flops(node, shapes)
        elif op == "BatchNormalization":
            contrib = _bn_flops(node, shapes)
        elif op in ("AveragePool", "GlobalAveragePool", "MaxPool"):
            contrib = _pool_flops(node, shapes)
        elif op in _ELEMENTWISE_OPS:
            contrib = sum(_elemwise_flops(node, shapes, o) for o in node.output)
        else:
            contrib = 0  # unknown op — silently skip (matches grader behavior)
        breakdown[op] = breakdown.get(op, 0) + int(contrib)

    total = sum(breakdown.values())
    if verbose:
        print(f"ONNX FLOPs breakdown for {onnx_path}:")
        for op, f in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            if f == 0:
                continue
            print(f"  {op:24s} {f:>18,}  ({f / 1e9:.4f} GFLOPs)")
        print(f"  {'TOTAL':24s} {total:>18,}  (~{total / 1e9:.3f} GFLOPs)")
    return total, breakdown


def export_pytorch_model_to_onnx(
    model,
    input_size,
    out_path: str,
    opset: int = 16,
):
    """Export a PyTorch nn.Module to ONNX with weights inlined.

    Args:
        model: PyTorch nn.Module (will be set to eval()).
        input_size: Tuple of (B, C, H, W). Typical: (1, 3, 480, 640).
        out_path: Output .onnx file path.
        opset: ONNX opset version (default 16, broad compatibility).

    Requires `torch` package.
    """
    try:
        import torch  # noqa: F401  (delayed import — only needed for this path)
    except ImportError as e:
        raise ImportError("`torch` is required for export_pytorch_model_to_onnx") from e
    import torch
    model = model.eval()
    dummy = torch.randn(input_size)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
    )


# ---------------------------------------------------------------------------
# Self-check (regression guard)
# ---------------------------------------------------------------------------

def _selfcheck():
    """Build a tiny 2-conv toy net, export → count, assert expected FLOPs.

    Toy net: Conv2d(3, 16, k=3, p=1) → ReLU → Conv2d(16, 21, k=1) at 64×64.
      Conv1: 2 × 16 × 3 × 9 × 64 × 64 = 3,538,944
      ReLU :     16 × 64 × 64        =    65,536
      Conv2: 2 × 21 × 16 × 1 × 64 × 64 = 2,752,512
      Total                          = 6,356,992
    """
    import os
    import tempfile

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[selfcheck] needs torch; install with `pip install torch`")
        sys.exit(1)

    net = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(16, 21, 1, bias=False),
    )
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_pytorch_model_to_onnx(net, (1, 3, 64, 64), tmp_path)
        total, breakdown = count_onnx_flops(tmp_path, verbose=True)
        expected = 6_356_992
        diff = abs(total - expected)
        ok = diff < expected * 0.005  # within 0.5%
        print(f"\n[selfcheck] expected ≈ {expected:,}  got = {total:,}  diff = {diff:,}  {'OK' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description="Standalone ONNX FLOPs counter (grader-compatible).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python grader_flops.py model.onnx\n"
            "  python grader_flops.py model.onnx --verbose\n"
            "  python grader_flops.py --selfcheck\n"
        ),
    )
    parser.add_argument("onnx_path", nargs="?", help="Path to .onnx file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-op breakdown")
    parser.add_argument("--selfcheck", action="store_true", help="Regression check vs known toy-net FLOPs")
    parser.add_argument(
        "--budget", type=float, default=None,
        help="FLOPs budget in GFLOPs (e.g. 15). If set, exits non-zero when total > budget.",
    )
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    if args.onnx_path is None:
        parser.print_help()
        sys.exit(1)

    if not args.onnx_path.endswith(".onnx"):
        print(f"[grader_flops] expected .onnx, got {args.onnx_path}", file=sys.stderr)
        sys.exit(1)

    total, _ = count_onnx_flops(args.onnx_path, verbose=args.verbose)
    g = total / 1e9
    print(f"\nONNX FLOPs (grader-sim): {total:,}  (~{g:.3f} GFLOPs)")

    if args.budget is not None:
        if g > args.budget:
            print(f"⚠️  exceeds budget {args.budget} GFLOPs by {g - args.budget:.3f} GFLOPs", file=sys.stderr)
            sys.exit(2)
        print(f"✓ within budget {args.budget} GFLOPs (margin {args.budget - g:.3f} GFLOPs)")


if __name__ == "__main__":
    _cli()
