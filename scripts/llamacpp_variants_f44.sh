#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does a different llama.cpp build route beat 7ba604f on Fedora 44, now that rocBLAS 7.1.1 is faster?
#   base    7ba604f + three patches (build-hip-f44)
#   cublas  7ba604f + three patches, -DGGML_CUDA_FORCE_CUBLAS=ON (dequant + hipBLAS for every quantized matmul)
#   master  bfdc321 + 0002 and 0003 (0001 is upstream), whose selector already sends dense prefill on
#           pre-DP4A GPUs to hipBLAS and MoE to MMQ
# Same flags for all: llama-bench -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3, and the two short gates.
# master renamed --no-mmap to --load-mode none; the gate command below passes the right one per build.
set -u
D=${1:-~/variants-f44}; mkdir -p $D
export HSA_ENABLE_SDMA=0
declare -A BIN=( [base]=~/llama-master/build-hip-f44/bin [cublas]=~/llama-master/build-hip-f44-cublas/bin [master]=~/llama-new/build-hip-f44/bin )
log () { echo "[$(date +%T)] $*" | tee -a $D/log; sync; }
val () { grep -aoE "$1 +\| +[0-9.]+" "$2" | grep -oE "[0-9.]+$" | head -1; }
for v in base cublas master; do
  B=${BIN[$v]}; [ -x $B/llama-bench ] || { log "$v: no build at $B"; continue; }
  NM="--no-mmap"; [ $v = master ] && NM="--load-mode none"
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b qwen3.6-35b-a3b-iq2m; do
    timeout -k 30 2400 $B/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/${v}_$m.log 2>&1
    log "  $v $m: pp512=$(val pp512 $D/${v}_$m.log) tg64=$(val tg64 $D/${v}_$m.log)"
  done
  g1=$(timeout -k 30 1800 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf $NM -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+")
  g2=$(timeout -k 30 1800 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf $NM -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+")
  log "  $v gates: 1.5B $g1 | 8B $g2"
done
log DONE
