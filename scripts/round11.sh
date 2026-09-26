#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round eleven: q8_0 with the in-kernel decoder. It was excluded on the strength of the materialising
# version, which is not the same kernel; GGML_RDNA1_PKF16_Q8 switches it inside one build.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round11; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
nice -n 5 cmake --build build-hip-pkf16 -j 7 --target llama-bench llama-perplexity > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A3 error: $O/build.log | tee -a $O/log; exit 1; }
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { env $1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $2/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do
  log "p$pass 8B MMQ:        $(pp X=1 $F)"
  log "p$pass 8B pkf16 q8on: $(pp GGML_RDNA1_PKF16_Q8=1 $B)"
  log "p$pass 8B pkf16 q8off:$(pp GGML_RDNA1_PKF16_Q8=0 $B)"
done
log "8B gate q8on:  $(GGML_RDNA1_PKF16_Q8=1 LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 9.1273)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
