#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round six: the packed-fp16 prefill GEMM in llama.cpp. Build, correctness against the CPU on the
# MUL_MAT cases, perplexity gates, per-op replay on the 1.5B's prefill shapes, and pp128/512/2048 on
# four models against the seven-patch build. GGML_RDNA1_PKF16=0 turns it off inside the same build.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round6; B=$M/build-hip-pkf16/bin; F=$M/build-hip-final/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
cmake -S . -B build-hip-pkf16 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/x86_64-redhat-linux/16/libgomp.so -DOpenMP_pthread_LIBRARY=/usr/lib64/libpthread.a > $O/configure.log 2>&1 || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build build-hip-pkf16 -j 7 --target llama-bench llama-perplexity test-backend-ops llama-cli > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A4 "error:" $O/build.log | tee -a $O/log; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 150
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
for v in 1 0; do
  GGML_RDNA1_PKF16=$v LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/test-backend-ops test -o MUL_MAT -b ROCm0 2>&1 | grep -a "MUL_MAT" > $O/tbo-mm-v$v.log
  log "MUL_MAT correctness, pkf16=$v: OK $(grep -ac 'OK' $O/tbo-mm-v$v.log) FAIL $(grep -ac 'FAIL' $O/tbo-mm-v$v.log)"
done
grep -a FAIL $O/tbo-mm-v1.log | head -12 | cut -c1-190 | while read l; do log "   $l"; done
log "1.5B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (8.9498 with MMQ)"
log "8B gate pkf16: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (9.1273 with MMQ)"
pp() { env $1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $2/llama-bench -m /opt/models/$3.gguf -ngl 99 -fa 1 -p 128,512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  log "p$pass $m MMQ:   $(pp X=1 $F $m)"
  log "p$pass $m pkf16: $(pp X=1 $B $m)"
done; done
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1200 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for m in qwen2.5-1.5b-q4km qwen3.6-35b-a3b-iq2m; do log "tg64 $m: MMQ $(tg $F $m) | pkf16 $(tg $B $m)"; done
KERNTRACE_OUT=$O/trace-1.5b-pp.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $B/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 0 -r 1 > /dev/null 2>&1
log "trace 1.5B pp512: $(head -1 $O/trace-1.5b-pp.txt | cut -c1-110)"
sed -n '5,9p' $O/trace-1.5b-pp.txt | sed -E 's/^ +//; s/ +/ /g' | cut -c1-120 | while read l; do log "   $l"; done
LD_LIBRARY_PATH=$L timeout -k 10 600 $B/llama-cli -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa on -c 4096 -n 48 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-1.5b.txt 2>&1
log "greedy text: $(grep -a -A3 'primary colours are' $O/text-1.5b.txt | grep -av 'Prompt:' | tr '\n' ' ' | cut -c1-180)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
