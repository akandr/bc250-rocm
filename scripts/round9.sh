#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round nine: threshold at 256 tokens, then the full split campaign on the final tree plus this kernel.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round9; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
nice -n 5 cmake --build build-hip-pkf16 -j 7 --target llama-bench llama-perplexity llama-cli > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A3 error: $O/build.log | tee -a $O/log; exit 1; }
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 128,256,512 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for m in qwen2.5-1.5b-q4km qwen3-14b; do
  log "$m MMQ:   $(pp $F $m)"
  log "$m pkf16: $(pp $B $m)"
done
log "1.5B gate: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
sleep 60
HIPBIN=$B ~/campaign_split_f44.sh ~/campaign-pkf16 > $O/campaign.out 2>&1
log "campaign rc=$? lines=$(wc -l < ~/campaign-pkf16/log)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
