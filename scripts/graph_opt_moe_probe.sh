#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The MoE identifies 70 fork regions and launches none, because is_valid() rejects regions whose
# branches write to overlapping memory. Buffer reuse is what creates that overlap, so try the
# knobs that change allocation and see whether any region becomes launchable.
# Also check prefill, since the earlier -v check only covered decode.
set -u
HIPB=${HIPB:-$HOME/llama-master/build-hip-pkf16/bin}
M=/opt/models/qwen3.6-35b-a3b-iq2m.gguf
launches(){ LD_LIBRARY_PATH=$HIPB env $1 GGML_CUDA_GRAPH_OPT=1 "$HIPB/llama-bench" -m "$M" $2 -r 1 -ngl 99 -v 2>&1 \
            | grep -cE "Launching [0-9]+ streams"; }
adds(){ LD_LIBRARY_PATH=$HIPB env $1 GGML_CUDA_GRAPH_OPT=1 "$HIPB/llama-bench" -m "$M" $2 -r 1 -ngl 99 -v 2>&1 \
            | grep -c "Adding stream at node"; }
echo "MoE, GGML_CUDA_GRAPH_OPT=1, forks identified vs streams launched"
printf "%-38s %-10s %8s %10s\n" condition phase adds launches
for c in "X=1" "GGML_CUDA_POOL_NOREUSE=1" "GGML_CUDA_NO_POOL=1"; do
  for p in "-p 0 -n 4:decode" "-p 512 -n 0:prefill"; do
    f=${p%%:*}; l=${p##*:}
    printf "%-38s %-10s %8s %10s\n" "$c" "$l" "$(adds "$c" "$f")" "$(launches "$c" "$f")"
  done
done
echo done
