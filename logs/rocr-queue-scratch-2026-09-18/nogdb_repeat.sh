#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Under gdb the September repro with the patched ROCr ended in a clean ggml abort four times out of four;
# the one run without gdb (14:56) died in HIP. Three runs without gdb, verbose so the HIP error string is
# printed, exit code and the kernel's segfault line recorded. Waits for perf4.
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/rocr-repro
until [ -f ~/f32iq-chain/perf4/done ]; do sleep 60; done
sleep 60
for i in 1 2 3; do
  LD_LIBRARY_PATH=$HOME/rocr-src/lib:$L timeout -k 30 1500 $M/build-hip-f32mv/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 -v > $O/nogdb-r$i.log 2>&1
  rc=$?
  echo "run $i: rc=$rc $(grep -aoE 'ROCm error.*|CUDA error.*' $O/nogdb-r$i.log | head -1) | $(journalctl -k --since '-6min' --no-pager 2>/dev/null | grep -a segfault | tail -1 | grep -oE 'segfault at [0-9a-f]+ .*' | cut -c1-120)" >> $O/nogdb.log
  sleep 60
done
echo NOGDB DONE >> $O/nogdb.log
