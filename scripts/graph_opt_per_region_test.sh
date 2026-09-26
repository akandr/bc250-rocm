#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the per-region gate unlock the MoE, and does it cost anything on the models that already worked?
# BASE is the shipped thirteen-patch build, NEW is the same tree with the per-region change.
# Both arms run with GGML_CUDA_GRAPH_OPT=1, because the gate only matters when the pass runs at all.
set -u
BASE=${BASE:-$HOME/llama-master/build-hip-pkf16/bin}
NEW=${NEW:-$HOME/llama-master/build-hip-perregion/bin}
N=${N:-6}
export HSA_ENABLE_SDMA=0

launches(){ LD_LIBRARY_PATH=$1 GGML_CUDA_GRAPH_OPT=1 "$1/llama-bench" -m /opt/models/$2.gguf -mmp 0 -p 0 -n 4 -r 1 -ngl 99 -v 2>&1 | grep -cE "Launching [0-9]+ streams"; }
tg(){ LD_LIBRARY_PATH=$1 GGML_CUDA_GRAPH_OPT=1 "$1/llama-bench" -m /opt/models/$2.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3 2>/dev/null | awk -F'|' '/tg64/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"+"); print a[1]}'; }

echo "=== streams launched, GGML_CUDA_GRAPH_OPT=1, base vs per-region ==="
printf "%-26s %10s %12s\n" model base per-region
for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs qwen2.5-1.5b-q4km; do
  printf "%-26s %10s %12s\n" "$m" "$(launches "$BASE" "$m")" "$(launches "$NEW" "$m")"
done

echo
echo "=== decode, both arms with GRAPH_OPT=1, interleaved ==="
for m in qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km; do
  echo "== $m"
  for i in $(seq 1 "$N"); do
    a=$(tg "$BASE" "$m"); b=$(tg "$NEW" "$m")
    printf "   %d  base %8s  per-region %8s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null|awk '/edge/{print $2}')"
  done
done
echo done
