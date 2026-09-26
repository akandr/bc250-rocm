#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round thirteen, redone. The first attempt built three variants and copied llama-bench, which is a thin
# executable: the kernel lives in libggml-hip.so, so all three arms ran the same library and the three
# binaries were byte-identical. The interval is a template parameter chosen at run time now, so one
# binary really does hold all three arms.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round13b; B=$M/build-hip-pkf16/bin; F=$M/build-hip-final/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
nice -n 5 cmake --build build-hip-pkf16 -j 7 --target llama-bench llama-perplexity > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A3 error: $O/build.log | tee -a $O/log; exit 1; }
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { env GGML_RDNA1_PKF16_PROMOTE=$1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $B/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
  for pr in 1 2 4; do log "p$pass $m promote=$pr: $(pp $pr $m)"; done
done; done
for pr in 1 2 4; do
  log "1.5B gate promote=$pr: $(GGML_RDNA1_PKF16_PROMOTE=$pr LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 8.9498)"
  log "27B gate promote=$pr: $(GGML_RDNA1_PKF16_PROMOTE=$pr LD_LIBRARY_PATH=$L timeout -k 30 3000 $B/llama-perplexity -m /opt/models/qwen3.8-27b-iq3xxs.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 6.2472)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
