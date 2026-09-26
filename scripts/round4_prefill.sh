#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round four: the prefill path. ROCm's pp512 is half Vulkan's and the kernel trace says the MMQ GEMM is
# 78 percent of it. Before touching MMQ's tile configuration (RDNA1 falls through to the RDNA2 table),
# the free question: which of the two prefill paths is faster on this board, MMQ or dequantise + rocBLAS
# (GGML_CUDA_FORCE_CUBLAS), and how does batch size change the answer.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round4; F=$M/build-hip-final/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { env $1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $F/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p $3 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.6-35b-a3b-iq2m; do
    for k in "X=1" "GGML_CUDA_FORCE_MMQ=1" "GGML_CUDA_FORCE_CUBLAS=1"; do
      log "p$pass $m [$k]: $(pp "$k" $m 128,512,2048)"
    done
  done
done
# the 1.5B's prefill kernels, traced, on each path
for k in "X=1" "GGML_CUDA_FORCE_CUBLAS=1"; do
  tag=$(echo $k | tr -d '=1' | tr ' ' '_')
  KERNTRACE_OUT=$O/trace-1.5b-pp-$tag.txt env $k LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $F/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 0 -r 1 > /dev/null 2>&1
  log "trace 1.5B pp512 [$k]: $(head -1 $O/trace-1.5b-pp-$tag.txt | cut -c1-110)"
  sed -n '5,9p' $O/trace-1.5b-pp-$tag.txt | sed -E 's/^ +//; s/ +/ /g' | cut -c1-120 | while read l; do log "   $l"; done
done
# gate on the cuBLAS path, to know whether it is even usable here
log "1.5B gate, FORCE_CUBLAS: $(GGML_CUDA_FORCE_CUBLAS=1 LD_LIBRARY_PATH=$L timeout -k 30 2400 $F/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (8.9498 on the default path)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
