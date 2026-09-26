#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the kernel-argument pool decide which dispatch mode a process gets?
# A fixed per-dispatch cost is what a kernarg placement choice would move.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
N=${N:-16}
cd /tmp
val(){ grep -E "^ +1024 " | head -1 | awk '{print $2}'; }
echo "run  default  devkernarg  edge"
for i in $(seq 1 "$N"); do
  a=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 ./df_hip 2>/dev/null | val)
  b=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 HIP_FORCE_DEV_KERNARG=1 ./df_hip 2>/dev/null | val)
  printf "%3d  %7s  %10s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
