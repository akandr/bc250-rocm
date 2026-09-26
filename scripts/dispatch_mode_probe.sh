#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Two questions about the slow mode, in one run.
#   Is it a clock effect?  A lower clock would slow the bandwidth-bound large tensor too.
#   Is it the queue count?  GPU_MAX_HW_QUEUES=1 is the documented knob on this board.
# Reports the per-node floor at 1024 elements and the GB/s reached at 1048576, per process.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
N=${N:-14}
cd /tmp
echo "run  floor_us  GBs_at_1M  edge"
for i in $(seq 1 "$N"); do
  o=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 ./df_hip 2>/dev/null)
  f=$(echo "$o" | grep -E "^ +1024 " | head -1 | awk '{print $2}')
  b=$(echo "$o" | grep -E "^ +1048576 " | head -1 | awk '{print $3}')
  printf "%3d  %8s  %9s  %s\n" "$i" "$f" "$b" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
