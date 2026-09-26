#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does address-space randomisation decide which dispatch mode a process gets?
# setarch -R disables ASLR for the child. Interleaved against the default.
set -u
HIPB=${HIPB:-$HOME/llama-new/build-hip-f44/bin}
N=${N:-14}
cd /tmp
val(){ grep -E "^ +1024 " | head -1 | awk '{print $2}'; }
echo "run  aslr_on  aslr_off  edge"
for i in $(seq 1 "$N"); do
  a=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 ./df_hip 2>/dev/null | val)
  b=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 setarch "$(uname -m)" -R ./df_hip 2>/dev/null | val)
  printf "%3d  %7s  %8s  %s\n" "$i" "$a" "$b" "$(sensors 2>/dev/null | awk '/edge/{print $2}')"
done
