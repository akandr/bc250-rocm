#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# What is llama.cpp's existing kernel fusion worth on a board with a 1.78 us dispatch floor?
#
# logs/dispatch-floor-2026-09-22 measured the floor. A trace of the MoE's decode shows 1298 dispatches
# a token, of which 748 are elementwise, normalisation, copy or gather kernels whose average duration is
# comparable to the floor: k_bin_bcast 217 a token at 2.47 us, rms_norm 129 at 3.09, and unary_op_kernel
# 69 at 1.51 us, which is shorter than the launch it costs.
#
# That trace was taken with fusion ON, which is the default, so those are what survives it. ggml-cuda
# already fuses rms_norm+mul+rope, rope+set_rows, mul_mat+add+glu and the mul_mat_id forms, plus topk_moe.
# GGML_CUDA_DISABLE_FUSION=1 turns all of it off, which measures what it is currently buying here and so
# whether the floor framing predicts anything.
#
# If fusion is worth much more on this board than the dispatches it removes would suggest elsewhere, that
# is the argument for extending it. If it is worth nothing, the framing is wrong and this direction dies.
# Usage: fusion_ab.sh [pairs]
set -u
N=${1:-6}
O=~/fusion-ab; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N pairs, edge $(temp)"
for p in $(seq 1 "$N"); do
  for m in qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
    for f in on off; do
      if [ $f = off ]; then export GGML_CUDA_DISABLE_FUSION=1; else unset GGML_CUDA_DISABLE_FUSION; fi
      r=$(LD_LIBRARY_PATH=$L timeout -k 20 900 "$HIP/llama-bench" -m /opt/models/$m.gguf -mmp 0 \
            -ngl 99 -fa on -p 0 -n 64 -r 3 2>/dev/null | grep -aE "tg64" |
          awk -F'|' '{v=$(NF-1); gsub(/^ +| +$/,"",v); gsub(/ /,"",v); print v}')
      log "p$p $m fusion=$f tg64=${r:-FAIL} edge=$(temp)"
      sleep 10
    done
  done
done
log done; touch "$O/DONE"
