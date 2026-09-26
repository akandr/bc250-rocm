#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# HIP graphs on and off under the campaign's environment (HSA_ENABLE_SDMA=0), three models.
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/graphs-sdma
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 HSA_ENABLE_SDMA=0
for round in 1 2 3; do
  for m in qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
    for arm in on off; do
      if [ $arm = off ]; then export GGML_CUDA_DISABLE_GRAPHS=1; else unset GGML_CUDA_DISABLE_GRAPHS; fi
      timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3 -o jsonl \
        > $O/${m}_tg_g${arm}_$round.jsonl 2>/dev/null
      echo "round $round $m graphs $arm $(date +%T)" >> $O/log
    done
  done
done
echo DONE >> $O/log
