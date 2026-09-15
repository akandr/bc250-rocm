#!/bin/bash
# Both repaired libraries on the path: rocBLAS and llama.cpp libggml-hip, qwen3-8B/14B, 2 chunks.
H=~/llama-master/build-hip/bin; D=~/s0915/ggmlfix/log
export LD_LIBRARY_PATH=$HOME/s0915/ggmlfix/lib:$HOME/rocblas-f16patch/lib HSA_ENABLE_SDMA=0
run() { label=$1 model=$2; shift 2
  env "$@" timeout -k 30 900 $H/llama-perplexity -m $model --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 > $D/$label.log 2>&1
  echo "$label: $(grep -o "Final estimate: PPL = [0-9.]*" $D/$label.log)"; }
M8=/opt/models/qwen3-8b-q8_0.gguf; M14=/opt/models/qwen3-14b.gguf
run both_default_1 $M8 X=1
run both_default_2 $M8 X=1
run both_f16_1 $M8 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16
run both_f16_tcache1 $M8 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GLIBC_TUNABLES=glibc.malloc.tcache_count=1
run both_f32 $M8 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
run q14_both_default $M14 X=1
echo DONE
