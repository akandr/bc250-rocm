#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the bimodal dispatch floor show up end to end in a real model?
# One llama-bench process per data point, graphs off (the exposed arm) vs on (the control).
# The floor is 12.9% of this model's token, so a 25% higher floor predicts about 3% slower.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
M=${M:-/opt/models/qwen2.5-1.5b-q4km.gguf}
N=${N:-16}
tg(){ "$HIPB/llama-bench" -m "$M" -p 0 -n 64 -r 3 -ngl 99 2>/dev/null | awk -F'|' '/tg64/{gsub(/ /,"",$(NF-1)); split($(NF-1),a,"+"); print a[1]}'; }
echo "n=$N  start $(date -Iseconds)  free $(free -m | awk '/Mem:/{print $7}') MiB"
echo "run  graphs_on  graphs_off  edge"
for i in $(seq 1 "$N"); do
  a=$(LD_LIBRARY_PATH=$HIPB tg)
  b=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 tg)
  printf "%3d  %9s  %10s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
echo "end $(date -Iseconds)"
