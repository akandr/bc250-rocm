#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The expert path on the packed-fp16 GEMM: off (MMQ) against three column-tile widths.
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/id-bench
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 HSA_ENABLE_SDMA=0
for round in 1 2 3; do
  for arm in off 16 32 64; do
    if [ $arm = off ]; then E="GGML_RDNA1_PKF16_ID=0"; else E="GGML_RDNA1_PKF16_ID_BN=$arm"; fi
    env $E timeout -k 30 2400 $B/llama-bench -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf \
      -mmp 0 -ngl 99 -fa on -p 256,512,2048 -n 0 -r 2 -o jsonl > $O/moe_${arm}_$round.jsonl 2>/dev/null
    echo "round $round arm $arm $(date +%T)" >> $O/log
  done
done
echo DONE >> $O/log
