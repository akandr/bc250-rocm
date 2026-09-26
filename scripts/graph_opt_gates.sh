#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# GGML_CUDA_GRAPH_OPT=1 reorders graph nodes to interleave independent branches across streams.
# Reordering can change accumulation order, so the documented gates are re-run with it on.
# Expected with the thirteen patches: 1.5B c4096 chunks8 = 8.9274, 8B c2048 chunks2 = 9.1125.
set -u
HIPB=${HIPB:-$HOME/llama-master/build-hip-final/bin}
P="$HIPB/llama-perplexity"
gate(){ LD_LIBRARY_PATH=$HIPB $P -m "$1" --no-mmap -ngl 99 -fa on -c "$2" -f ~/wiki.test.raw --chunks "$3" 2>&1 \
        | grep -oE "Final estimate: PPL = [0-9.]+" | awk '{print $5}'; }
echo "start $(date -Iseconds)"
for spec in "qwen2.5-1.5b-q4km 4096 8 8.9274" "qwen3-8b-q8_0 2048 2 9.1125"; do
  set -- $spec; m=$1; c=$2; ch=$3; want=$4
  a=$(gate /opt/models/$m.gguf "$c" "$ch")
  b=$(GGML_CUDA_GRAPH_OPT=1 gate /opt/models/$m.gguf "$c" "$ch")
  printf "%-22s documented %s   off %s   GRAPH_OPT=1 %s\n" "$m" "$want" "$a" "$b"
done
echo "end $(date -Iseconds)"
