#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The GEMM's admission threshold on columns, set at the old 128x128 tile. Does it still belong at 256?
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/thr-bench
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 HSA_ENABLE_SDMA=0
for round in 1 2 3; do
  for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
    for c in 256 128 64 32; do
      GGML_RDNA1_PKF16_MINCOLS=$c timeout -k 30 2400 $B/llama-bench -m /opt/models/$m.gguf \
        -mmp 0 -ngl 99 -fa on -p 32,64,128,256 -n 0 -r 3 -o jsonl > $O/${m}_c${c}_$round.jsonl 2>/dev/null
      echo "round $round $m mincols $c $(date +%T)" >> $O/log
    done
  done
done
echo DONE >> $O/log
