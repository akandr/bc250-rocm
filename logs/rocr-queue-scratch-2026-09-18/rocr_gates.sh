#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Perplexity gates under the installed (patched) ROCr and under the stock one. The stock library is
# reached through a directory holding only a symlink to it, so LD_LIBRARY_PATH does not also pull the
# stock rocBLAS and comgr from /usr/lib64. Uses build-hip-f32mv, which has llama-perplexity.
L=/opt/bc250-rocm/lib64; M=~/llama-master; B=$M/build-hip-f32mv/bin; O=~/rocr-repro
until grep -q "VALIDATE DONE" $O/validate.log 2>/dev/null; do sleep 30; done
sleep 30
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
mkdir -p ~/stock-rocr && ln -sfn /usr/lib64/libhsa-runtime64.so.1.18.0 ~/stock-rocr/libhsa-runtime64.so.1
log() { echo "[$(date +%T)] $*" | tee -a $O/gates.log; }
: > $O/gates.log
gate() { LD_LIBRARY_PATH=$1 timeout -k 30 2400 $B/llama-perplexity -m /opt/models/$2.gguf --no-mmap -ngl 99 -fa on -c $3 -f ~/wiki.test.raw --chunks $4 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*'; }
log "which hsa, installed: $(ldd $B/llama-bench | grep -a hsa-runtime | tr -s ' ')"
log "which hsa, stock dir: $(LD_LIBRARY_PATH=$HOME/stock-rocr:$L ldd $B/llama-bench | grep -a hsa-runtime | tr -s ' ')"
log "installed 1.5B gate: $(gate $L qwen2.5-1.5b-q4km 4096 8)  (expect 8.9498)"
log "stock     1.5B gate: $(gate $HOME/stock-rocr:$L qwen2.5-1.5b-q4km 4096 8)"
log "installed 8B gate: $(gate $L qwen3-8b-q8_0 2048 2)  (expect 9.1273)"
log "stock     8B gate: $(gate $HOME/stock-rocr:$L qwen3-8b-q8_0 2048 2)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
echo "GATES DONE" >> $O/gates.log
