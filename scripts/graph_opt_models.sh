#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# GGML_CUDA_GRAPH_OPT=1 across the model set, on the thirteen-patch build. Decode only:
# prefill measured flat on the 1.5B, and prefill has little independent work to overlap.
# One process per data point, arms interleaved within each model.
set -u
HIPB=${HIPB:-$HOME/llama-master/build-hip-final/bin}
N=${N:-6}
tg(){ LD_LIBRARY_PATH=$HIPB "$HIPB/llama-bench" -m "$1" -p 0 -n 32 -r 3 -ngl 99 2>/dev/null \
      | awk -F'|' '/tg32/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"+"); print a[1]}'; }
echo "start $(date -Iseconds)  free $(free -m|awk '/Mem:/{print $7}') MiB"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b deepseek-r1-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  M=/opt/models/$m.gguf
  [ -f "$M" ] || continue
  streams=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_GRAPH_OPT=1 "$HIPB/llama-bench" -m "$M" -p 0 -n 4 -r 1 -ngl 99 -v 2>&1 | grep -c "Launching")
  echo "== $m   (concurrent launches seen: $streams)"
  for i in $(seq 1 "$N"); do
    a=$(tg "$M"); b=$(GGML_CUDA_GRAPH_OPT=1 tg "$M")
    printf "   %d  off %8s  on %8s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null|awk '/edge/{print $2}')"
  done
done
echo "end $(date -Iseconds)"
