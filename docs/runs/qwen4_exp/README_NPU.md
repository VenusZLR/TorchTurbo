# Qwen3.8-Flash-Next training on Ascend NPU

Qwen3.8-Flash-Next checkpoints use the `qwen4_exp` architecture. TorchTurbo
vendors its configuration and Transformers implementation, registers
`TurboQwen4ExpForConditionalGeneration`, and exposes `turbo_qwen4_exp` to
ms-swift. Swift uses the compatible `qwen3_5` processor and template.

## Features

- Text, image, and video training with packed sample boundaries.
- FSDP2, expert parallelism, and Ulysses sequence parallelism, including
  slicing text after multimodal embedding fusion.
- Gated delta networks, query selection attention (QSA), hyperconnections,
  and per-layer embeddings (PLE).
- Frozen n-gram tables shared in CPU memory by the processes on each node;
  row lookup is prefetched before the corresponding PLE layer executes.
- Hugging Face and distributed checkpoints that retain CPU-resident table
  shards and their source dtype, including DPO reference model copies.
- Dynamic-length GDN L2Norm kernels for feature dimensions up to 512, with
  partial-tile gradients and compiled-kernel reuse across packed lengths.
  Larger feature dimensions use the upstream FLA implementation.
- Architecture-specific MFU accounting for heterogeneous experts, QSA,
  hyperconnections, and PLE.

## Environment

Use the repository's Ascend image, which provides CANN, `torch_npu`, Triton,
and the FLA/FLA-NPU operators. Install the Swift dependencies from the repo root:

```bash
bash scripts/runs/qwen4_exp/install_swift_deps.sh
pip install -e .
```

The shared Swift dependency script pins the supported Transformers and Swift
versions. No separate Qwen4 Transformers installation is needed.

## Text SFT

Set `MODEL` to your checkpoint and `DATASET` to a Swift dataset or data file:

```bash
export MODEL
export DATASET
export OUTPUT_DIR=./outputs/qwen38-flash-next
bash scripts/runs/qwen4_exp/swift_text_sft.sh
```

The script requires nonempty `MODEL` and `DATASET`. Its default topology is
one node with 16 processes, SP=8 and EP=16, BF16, full CPU offload, packing,
and reentrant activation checkpointing. Adjust the process count and parallel
sizes together for your allocation. For example, runtime overrides are passed
through to Swift and TorchTurbo:

```bash
MAX_STEPS=5 SEQ_LEN=32768 \
  bash scripts/runs/qwen4_exp/swift_text_sft.sh --tt_sp_size 8 --tt_ep_size 16
```

`TT_CONFIG` selects another engine configuration. Set `NNODES`, `NODE_RANK`,
`MASTER_ADDR`, and `MASTER_PORT` for a multi-node job. Use the same shared
checkpoint path and matching parallel settings on every node.

`TORCHTURBO_PLE_SHM_DIR` defaults to `/dev/shm`. It must have enough capacity
for the frozen tables and be visible to all local worker processes. Its backing
files are unlinked after the local workers map them. This storage is separate
from optimizer or FSDP CPU offload. Table parameters remain frozen while the
surrounding PLE projections are trainable.

The model also accepts image/video datasets through the same Swift processor
and template. The example freezes the vision tower and projector; override
`--tt_freeze_vision_tower` and `--tt_freeze_multi_modal_projector` when training
them. Packed text metadata describes the language sequence and is kept out of
the vision tower's attention calls.

## Checkpoints

The default engine config exports Hugging Face weights at the end of training.
Use `RESUME_FROM_CHECKPOINT` to resume a Swift run. Restoring optimizer state
requires the same expert-parallel size as the saved run. Source PLE table
shards and their dtype are preserved by both HF export and distributed saves.

## Validation

Run model, parallelism, CPU table, checkpoint, and L2Norm regression tests:

```bash
PYTHONPATH=src python3 -m pytest -q \
  tests/models/qwen4_exp \
  tests/models/modules/attention/gdn \
  tests/models/modules/test_cpu_resident.py \
  tests/engine/backends/fsdp2 \
  tests/adapters/msswift/test_registry.py \
  tests/ops/providers/npu/test_gdn_l2norm.py
```

The model tests include small reference comparisons, packed boundaries,
multimodal fusion, CPU shared-table I/O, and multiprocess SP gradients. Native
NPU tests require available Ascend devices; other distributed tests use Gloo.
