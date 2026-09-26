#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The front page's correctness table and its two verification gates, on the thirteen-patch build.
# Both predate the prefill GEMM, which changes the arithmetic of every prefill matmul it takes over.
set -u
H=~/llama-master/build-hip-pkf16/bin
V=~/llama-master/build-vk-f44/bin
O=~/correctness13
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64
ppl () { grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" "$1" | tail -1 | sed 's/Final estimate: PPL = //'; }

# the table: context 2048, eight chunks, flash attention on, default compute type
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b deepseek-r1-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  timeout -k 30 3600 $H/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 \
    -f ~/wiki.test.raw --chunks 8 > $O/hip_$m.log 2>&1
  timeout -k 30 3600 $V/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 \
    -f ~/wiki.test.raw --chunks 8 > $O/vk_$m.log 2>&1
  echo "table $m rocm=$(ppl $O/hip_$m.log) vulkan=$(ppl $O/vk_$m.log)" >> $O/log
done

# the two gates of step 8
timeout -k 30 3600 $H/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on \
  -c 4096 -f ~/wiki.test.raw --chunks 8 > $O/gate_1.5b.log 2>&1
echo "gate 1.5b c4096 chunks8 = $(ppl $O/gate_1.5b.log)" >> $O/log
timeout -k 30 3600 $H/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on \
  -c 2048 -f ~/wiki.test.raw --chunks 2 > $O/gate_8b_fp16.log 2>&1
echo "gate 8b c2048 chunks2 default = $(ppl $O/gate_8b_fp16.log)" >> $O/log
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 timeout -k 30 3600 $H/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf \
  --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 > $O/gate_8b_f32.log 2>&1
echo "gate 8b c2048 chunks2 f32 = $(ppl $O/gate_8b_f32.log)" >> $O/log
echo DONE >> $O/log
