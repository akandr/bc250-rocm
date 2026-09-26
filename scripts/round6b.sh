#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# End to end with the coalesced store, and a guard so the narrow matrices keep MMQ.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round6b; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
while pgrep -x test-backend-op >/dev/null || pgrep -x llama-bench >/dev/null; do sleep 20; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 128,512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.8-27b-iq3xxs; do
  log "p$pass $m MMQ:   $(pp $F $m)"
  log "p$pass $m pkf16: $(pp $B $m)"
done; done
log "tg64 1.5B: MMQ $(LD_LIBRARY_PATH=$L $F/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}') | pkf16 $(LD_LIBRARY_PATH=$L $B/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}')"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
