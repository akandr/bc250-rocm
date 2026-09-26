#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round fourteen: double buffering. The promotion result says the kernel is bound by its tile path rather
# than its arithmetic, so the next thing to try is fetching stage k+1 while stage k computes.
# GGML_RDNA1_PKF16_DBUF switches it inside one binary, so both arms share a library.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round14; B=$M/build-hip-pkf16/bin; F=$M/build-hip-final/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
while pgrep -x test-backend-op >/dev/null || pgrep -x llama-bench >/dev/null; do sleep 20; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { env GGML_RDNA1_PKF16_DBUF=$1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $B/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-14b qwen3.8-27b-iq3xxs; do
  log "p$pass $m dbuf=0: $(pp 0 $m)"
  log "p$pass $m dbuf=1: $(pp 1 $m)"
done; done
for d in 1 0; do
  log "1.5B gate dbuf=$d: $(GGML_RDNA1_PKF16_DBUF=$d LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 8.9498)"
  log "27B gate dbuf=$d: $(GGML_RDNA1_PKF16_DBUF=$d LD_LIBRARY_PATH=$L timeout -k 30 3000 $B/llama-perplexity -m /opt/models/qwen3.8-27b-iq3xxs.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 6.2472)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
