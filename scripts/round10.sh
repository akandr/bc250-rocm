#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round ten: the IQ types and q5_K in the same kernel. The 27B is IQ3_S 53 percent, IQ3_XXS 23, IQ4_XS 13
# and Q5_K 7, so it is the model this should move; the MoE gains only its dense q5_K, since its experts
# go through MUL_MAT_ID, which this hook does not touch.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round10; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
while pgrep -x test-backend-op >/dev/null || pgrep -x llama-bench >/dev/null; do sleep 20; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
# gates first: a wrong dequantiser shows up here, and both models exercise the new types at batch 512
log "27B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 3000 $B/llama-perplexity -m /opt/models/qwen3.8-27b-iq3xxs.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "27B gate MMQ:   $(LD_LIBRARY_PATH=$L timeout -k 30 3000 $F/llama-perplexity -m /opt/models/qwen3.8-27b-iq3xxs.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "MoE gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 3000 $B/llama-perplexity -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 3 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 6.7545)"
log "1.5B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 8.9498, q4/q6-only build 8.9274)"
for b in $F $B; do
  LD_LIBRARY_PATH=$L timeout -k 10 900 $b/llama-cli -m /opt/models/qwen3.8-27b-iq3xxs.gguf -ngl 99 -fa on -c 4096 -n 48 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-27b-$(basename $(dirname $b)).txt 2>&1
done
if diff -q <(grep -a -A30 'primary colours' $O/text-27b-build-hip-final.txt | grep -av 'Prompt:') <(grep -a -A30 'primary colours' $O/text-27b-build-hip-pkf16.txt | grep -av 'Prompt:') >/dev/null; then log "27B greedy text IDENTICAL"; else log "27B greedy text DIFFERS"; fi
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km; do
  log "p$pass $m MMQ:   $(pp $F $m)"
  log "p$pass $m pkf16: $(pp $B $m)"
done; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
