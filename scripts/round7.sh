#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round seven: the same packed-fp16 GEMM, but dequantising tiles inside the kernel, so nothing is
# materialised. Correctness, then per-shape against MMQ, then end to end, then the gates.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round7; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
while pgrep -x test-backend-op >/dev/null || pgrep -x llama-bench >/dev/null; do sleep 20; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
for v in 1 0; do
  GGML_RDNA1_PKF16=$v LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/test-backend-ops test -o MUL_MAT -b ROCm0 2>&1 | grep -a "MUL_MAT" > $O/tbo-v$v.log
  log "MUL_MAT correctness, in-kernel dequant=$v: OK $(grep -ac 'OK' $O/tbo-v$v.log) FAIL $(grep -ac 'FAIL' $O/tbo-v$v.log)"
done
grep -a FAIL $O/tbo-v1.log | head -8 | cut -c1-170 | while read l; do log "   $l"; done
for b in $F $B $F $B; do
  LD_LIBRARY_PATH=$L timeout -k 30 1200 $b/test-backend-ops perf --test-file ~/opgraph/ops-1.5b-pp2048.txt -b ROCm0 2>/dev/null \
    | grep -a "us/run" | grep -aE "sources=q[46]_K" \
    | sed -E "s|.*name=([^,]+),.*sources=(q[46]_K\[[0-9]+,[0-9]+)[^]]*\],f32\[([0-9]+),([0-9]+)[^]]*\].*- +([0-9.]+) us.*|$(basename $(dirname $b)) \1 \2] x\4 \5|" \
    | grep x2048 >> $O/shapes.log
  sleep 15
done
log "per-shape written"
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 128,512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.8-27b-iq3xxs; do
  log "p$pass $m MMQ:   $(pp $F $m)"
  log "p$pass $m pkf16: $(pp $B $m)"
done; done
log "1.5B gate: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/../build-hip-pkf16/bin/llama-bench --help >/dev/null 2>&1; LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/build-hip-pkf16/bin/test-backend-ops --help >/dev/null 2>&1; echo skipped-no-perplexity-binary)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
