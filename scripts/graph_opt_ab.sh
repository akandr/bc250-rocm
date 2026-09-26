#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# GGML_CUDA_GRAPH_OPT=1 turns on ggml-cuda's multi-stream graph optimisation, which is opt-in
# and off by default. It forks independent branches onto separate streams. ggml-vulkan's own
# overlap is worth 10.8% of its GPU span on this board, so the headroom is real.
# One process per data point, interleaved.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
M=${M:-/opt/models/qwen2.5-1.5b-q4km.gguf}
N=${N:-12}
run(){ LD_LIBRARY_PATH=$HIPB "$HIPB/llama-bench" -m "$M" $2 -r 3 -ngl 99 2>/dev/null \
       | awk -F'|' -v t="$3" '$0 ~ t {gsub(/ /,"",$(NF-1)); split($(NF-1),a,"+"); print a[1]}'; }
echo "n=$N  $(basename "$M")  start $(date -Iseconds)"
echo "run  tg_off   tg_on    pp_off   pp_on    edge"
for i in $(seq 1 "$N"); do
  a=$(run "" "-p 0 -n 64" tg64)
  b=$(GGML_CUDA_GRAPH_OPT=1 run "" "-p 0 -n 64" tg64)
  c=$(run "" "-p 512 -n 0" pp512)
  d=$(GGML_CUDA_GRAPH_OPT=1 run "" "-p 512 -n 0" pp512)
  printf "%3d  %7s %7s  %8s %8s  %s\n" "$i" "$a" "$b" "$c" "$d" "$(sensors 2>/dev/null|awk '/edge/{print $2}')"
done
echo "end $(date -Iseconds)"
