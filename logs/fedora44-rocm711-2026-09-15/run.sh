#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Fedora 44 userspace (ROCm 7.1.1, system rocBLAS), kernel 7.1.8 with production amdgpu settings.
export PATH=/usr/bin:/usr/sbin HSA_ENABLE_SDMA=0
unset LD_LIBRARY_PATH
H=~/llama-master/build-hip-f44/bin; D=~/s0915/f44/log; M=/opt/models
echo "rocblas: $(ldd $H/libggml-hip.so.0 | grep -E "rocblas|amdhip")"
echo "helpers: $(readelf -Ws $H/libggml-hip.so.0 | grep -c "hfsf2\|sfhf2") symbols, $(objdump -d --no-show-raw-insn /usr/lib64/librocblas.so.5 2>/dev/null | grep -c "call.*extendhfsf2") rocblas calls"
ppl() { label=$1 model=$2 c=$3 ch=$4; shift 4
  env "$@" timeout -k 30 1800 $H/llama-perplexity -m $model --no-mmap -ngl 99 -fa on -c $c -f ~/wiki.test.raw --chunks $ch > $D/$label.log 2>&1
  echo "$label: rc=$? $(grep -aoE "Final estimate: PPL = [0-9.]+" $D/$label.log) $(grep -aE "error|abort|STATUS" $D/$label.log | grep -v "^load" | head -1 | cut -c1-120)"; }
ppl gate_q15 $M/qwen2.5-1.5b-q4km.gguf 4096 8 X=1
ppl q8_default_1 $M/qwen3-8b-q8_0.gguf 2048 2 X=1
ppl q8_default_2 $M/qwen3-8b-q8_0.gguf 2048 2 X=1
ppl q8_f16_tcache1 $M/qwen3-8b-q8_0.gguf 2048 2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GLIBC_TUNABLES=glibc.malloc.tcache_count=1
ppl q8_f32 $M/qwen3-8b-q8_0.gguf 2048 2 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
ppl q14_default $M/qwen3-14b.gguf 2048 2 X=1
timeout -k 20 900 $H/llama-bench -m $M/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/bench.log 2>&1
echo "bench: $(grep -aE "pp512|tg64" $D/bench.log | awk -F"|" "{print \$(NF-2), \$(NF-1)}" | tr "\n" " ")"
timeout 5400 $H/test-backend-ops -b ROCm0 > $D/tbo_full.log 2>&1; rc=$?
echo "test-backend-ops: rc=$rc ok=$(grep -ac ": .\[1;32mOK" $D/tbo_full.log) fail=$(grep -a "FAIL" $D/tbo_full.log | grep -vc "Backend ROCm0") $(tail -2 $D/tbo_full.log | head -1)"
echo DONE
