#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Is ROCm's per-dispatch cost bimodal when HIP graph capture is off?
# Same program, same board, interleaved, only GGML_CUDA_DISABLE_GRAPHS differs.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
N=${N:-24}
cd /tmp
val(){ grep -E "^ +1024 " | head -1 | awk '{print $2}'; }
echo "n=$N  start $(date -Iseconds)  edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
echo "run  graphs_on  graphs_off  edge"
for i in $(seq 1 "$N"); do
  a=$(LD_LIBRARY_PATH=$HIPB ./df_hip 2>/dev/null | val)
  b=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 ./df_hip 2>/dev/null | val)
  printf "%3d  %9s  %10s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
echo "end $(date -Iseconds)"
