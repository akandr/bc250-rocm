#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round eight: rebuild everything (the gate binary was stale), then the checks that actually exercise
# this kernel: the 1.5B perplexity gate and greedy text run prefill at batch 512 with q4_K and q6_K
# weights, which is exactly the path. Then pp on the 1.5B (should win) and the 8B (q8_0 now excluded,
# should be unchanged), and a count of how many test-backend-ops MUL_MAT cases reach the kernel at all.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round8; F=$M/build-hip-final/bin; B=$M/build-hip-pkf16/bin
mkdir -p $O; : > $O/log
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
nice -n 5 cmake --build build-hip-pkf16 -j 7 --target llama-bench llama-perplexity llama-cli test-backend-ops > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A3 "error:" $O/build.log | tee -a $O/log; exit 1; }
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
GGML_RDNA1_PKF16=1 LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/test-backend-ops test -o MUL_MAT -b ROCm0 2>&1 | grep -a "MUL_MAT" > $O/tbo.log
log "MUL_MAT: OK $(grep -ac 'OK' $O/tbo.log) FAIL $(grep -ac 'FAIL' $O/tbo.log); cases with n>=64 and m>=512 (the ones this kernel can take): $(grep -aoE 'm=[0-9]+,n=[0-9]+' $O/tbo.log | awk -F'[=,]' '$2>=512 && $4>=64' | wc -l)"
log "1.5B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (MMQ 8.9498)"
log "14B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen3-14b.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "14B gate MMQ:   $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $F/llama-perplexity -m /opt/models/qwen3-14b.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
for b in $F $B; do
  LD_LIBRARY_PATH=$L timeout -k 10 600 $b/llama-cli -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa on -c 4096 -n 48 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-$(basename $(dirname $b)).txt 2>&1
done
if diff -q <(grep -a -A30 'primary colours' $O/text-build-hip-final.txt | grep -av 'Prompt:') <(grep -a -A30 'primary colours' $O/text-build-hip-pkf16.txt | grep -av 'Prompt:') >/dev/null; then log "greedy text IDENTICAL"; else log "greedy text DIFFERS"; fi
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 128,512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1200 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b qwen3.6-35b-a3b-iq2m; do
  log "p$pass $m MMQ:   $(pp $F $m)"
  log "p$pass $m pkf16: $(pp $B $m)"
done; done
for m in qwen2.5-1.5b-q4km qwen3-14b; do log "tg64 $m: MMQ $(tg $F $m) | pkf16 $(tg $B $m)"; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
