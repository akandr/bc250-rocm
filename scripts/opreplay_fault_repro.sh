#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Can the 20 September page fault be reproduced? It came from a test-backend-ops replay of the MoE's
# pp512 graph. Three runs with the packed-fp16 GEMM on, then three with it off, counting the kernel's
# fault lines before and after each.
set -u
B=~/llama-master/build-hip-pkf16/bin
O=~/fault-repro
mkdir -p $O
: > $O/log
export LD_LIBRARY_PATH=/opt/bc250-rocm/lib64
PAT="page fault \(src_id|memory access fault"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
echo "start faults=$(faults)" >> $O/log
for arm in on off; do
  [ $arm = off ] && export GGML_RDNA1_PKF16=0 || unset GGML_RDNA1_PKF16
  for i in 1 2 3; do
    before=$(faults)
    timeout -k 30 2400 $B/test-backend-ops perf --test-file ~/opgraph/ops-moe-pp512.txt -b ROCm0 > $O/${arm}_$i.log 2>&1
    rc=$?
    echo "gemm=$arm run=$i rc=$rc faults $before -> $(faults)" >> $O/log
  done
done
echo DONE >> $O/log
