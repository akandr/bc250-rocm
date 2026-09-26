#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Tile-shape sweep for the packed-fp16 prefill GEMM. One binary, one library, arms interleaved.
# 0 = shipped 128x128, 1 = 128x64, 2 = 128x32, 3 = 64x64, 4 = 64x128.
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/tile-bench
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 HSA_ENABLE_SDMA=0
for round in 1 2 3; do
  for m in qwen2.5-1.5b-q4km qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
    for t in 0 1 2 3 4; do
      GGML_RDNA1_PKF16_TILE=$t timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf \
        -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3 -o jsonl > $O/${m}_pp_t${t}_$round.jsonl 2>/dev/null
      echo "round $round $m tile $t $(date +%T)" >> $O/log
    done
  done
done
echo DONE >> $O/log
