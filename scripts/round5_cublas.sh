#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round five: the prefill path, properly. GGML_CUDA_FORCE_CUBLAS is a compile-time option, so round four
# measured the same build three times. This builds it: weights dequantised to f16 and multiplied by
# rocBLAS (the native gfx1013 one) instead of MMQ's emulated int8 tiles. On RDNA1 the emulated dp4a is
# what made the int8 matvec lose to a float kernel, and Vulkan's prefill, also float, is twice ROCm's.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round5; B=$M/build-hip-cublas/bin; F=$M/build-hip-final/bin
mkdir -p $O
rm -rf $M/build-hip-cublas
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
cmake -S . -B build-hip-cublas -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF -DGGML_CUDA_FORCE_CUBLAS=ON -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/x86_64-redhat-linux/16/libgomp.so -DOpenMP_pthread_LIBRARY=/usr/lib64/libpthread.a > $O/configure.log 2>&1 || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build build-hip-cublas -j 7 --target llama-bench llama-perplexity test-backend-ops > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A2 "error:" $O/build.log | tee -a $O/log; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 180
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 128,512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp[0-9]+" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1200 $1/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  log "p$pass $m MMQ:    $(pp $F $m)"
  log "p$pass $m cuBLAS: $(pp $B $m)"
done; done
for m in qwen2.5-1.5b-q4km qwen3.6-35b-a3b-iq2m; do log "tg64 $m: MMQ $(tg $F $m) | cuBLAS $(tg $B $m)"; done
log "1.5B gate, cuBLAS build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (8.9498 with MMQ)"
log "8B gate, cuBLAS build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (9.1273 with MMQ)"
KERNTRACE_OUT=$O/trace-1.5b-pp-cublas.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $B/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 0 -r 1 > /dev/null 2>&1
log "trace 1.5B pp512 cuBLAS: $(head -1 $O/trace-1.5b-pp-cublas.txt | cut -c1-110)"
sed -n '5,10p' $O/trace-1.5b-pp-cublas.txt | sed -E 's/^ +//; s/ +/ /g' | cut -c1-130 | while read l; do log "   $l"; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
