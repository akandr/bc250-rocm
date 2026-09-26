#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The first gdb run of the September repro under the patched ROCr ended in a clean ggml abort instead of
# the HIP segfault seen at 14:56. Repeat it, under gdb, until one run faults or three have been done, so
# the two outcomes' frequency is on record and a faulting run has its backtrace. Waits for perf3.
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/rocr-repro
until [ -f ~/f32all-chain/perf3.done ]; do sleep 30; done
sleep 60
for i in 1 2 3; do
  LD_LIBRARY_PATH=$HOME/rocr-src/lib:$L timeout -k 30 1500 gdb -q -batch -x $O/gdb.cmds --args $M/build-hip-f32mv/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 > $O/gdb-hip-crash-r$i.log 2>&1
  if grep -q "SIGSEGV" $O/gdb-hip-crash-r$i.log; then echo "run $i: SIGSEGV" >> $O/repeat.log; break; else echo "run $i: $(grep -aoE 'received signal [A-Z]+|ROCm error.*|CUDA error.*' $O/gdb-hip-crash-r$i.log | head -2 | tr '\n' ' ')" >> $O/repeat.log; fi
  sleep 60
done
echo REPEAT DONE >> $O/repeat.log
