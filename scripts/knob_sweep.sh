#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Runtime knobs measured again on the thirteen-patch build. They were last swept on the three-patch
# build, before the kernels and the dispatch mix changed.
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/knob-bench
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 HSA_ENABLE_SDMA=0
run () {   # name, then the environment assignments
  local name=$1; shift
  for m in qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
    env "$@" timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf \
      -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3 -o jsonl > $O/${m}_${name}_$ROUND.jsonl 2>/dev/null
    echo "round $ROUND $m $name $(date +%T)" >> $O/log
  done
}
for ROUND in 1 2 3; do
  run base    DUMMY=0
  run hwq1    GPU_MAX_HW_QUEUES=1
  run kernarg HIP_FORCE_DEV_KERNARG=1
  run noint   HSA_ENABLE_INTERRUPT=0
done
echo DONE >> $O/log
