#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# llama.cpp HIP backend rebuilt with -mf16c (no half helpers) + rocBLAS with repaired helpers.
H=~/llama-master/build-hip-f16c/bin; D=~/s0915/f16cbuild/log
export LD_LIBRARY_PATH=$HOME/rocblas-f16patch/lib HSA_ENABLE_SDMA=0
ppl() { label=$1 model=$2 c=$3 ch=$4; shift 4
  env "$@" timeout -k 30 1800 $H/llama-perplexity -m $model --no-mmap -ngl 99 -fa on -c $c -f ~/wiki.test.raw --chunks $ch > $D/$label.log 2>&1
  echo "$label: $(grep -aoE "Final estimate: PPL = [0-9.]+" $D/$label.log)"; }
M=/opt/models
ppl gate_q15 $M/qwen2.5-1.5b-q4km.gguf 4096 8 X=1
ppl q8_default_1 $M/qwen3-8b-q8_0.gguf 2048 2 X=1
ppl q8_default_2 $M/qwen3-8b-q8_0.gguf 2048 2 X=1
ppl q8_f16_tcache1 $M/qwen3-8b-q8_0.gguf 2048 2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GLIBC_TUNABLES=glibc.malloc.tcache_count=1
ppl q8_f32 $M/qwen3-8b-q8_0.gguf 2048 2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
ppl q14_default $M/qwen3-14b.gguf 2048 2 X=1
cd ~/llama-master && cmake --build build-hip-f16c -j6 --target llama-bench > $D/bench_build.log 2>&1
for b in build-hip build-hip-f16c build-hip build-hip-f16c; do
  timeout -k 20 900 ~/llama-master/$b/bin/llama-bench -m $M/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/bench_$b.$RANDOM.log 2>&1
done
for f in $D/bench_*.log; do echo "$(basename $f): $(grep -aoE "pp512 +\| +[0-9.]+" $f | grep -oE "[0-9.]+$") $(grep -aoE "tg64 +\| +[0-9.]+" $f | grep -oE "[0-9.]+$")"; done
echo DONE
