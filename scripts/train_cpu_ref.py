#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""The training step of torch_bench.py on the CPU, as the reference for the board's 33.8 ktok/s: the same
four pre-norm blocks at d_model 512, heads 8, seq 256, fp32, batches 1, 4 and 16, Adam."""
import sys, torch
sys.path.insert(0, "/home/akandr")
import torch_bench as tb
print(f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads, "
      f"mkl {torch.backends.mkl.is_available()}, mkldnn {torch.backends.mkldnn.is_available()}")
rows = tb.training(torch.device("cpu"), 512, 8, 4, 256, [1, 4, 16], [torch.float32])
for r in rows:
    print(r)
