#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round two after the seven-patch assembly. CPU first (sweeps, one variant build, the tracer), then GPU.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round2; F=$M/build-hip-final/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.seven      # the seven-patch state (base for the sweeps)
# --- CPU: sweeps (fa_row_sweep uses /tmp/fattn-tile.cuh.patched4 as its base; point it at the seven-patch file)
cp /tmp/fattn-tile.cuh.patched4 /tmp/fattn-tile.cuh.patched4.bak && cp /tmp/fattn-tile.cuh.seven /tmp/fattn-tile.cuh.patched4
~/fa_row_sweep.sh 128 "1:32:4:32:64" "1:64:4:32:64" "1:32:8:32:64" "1:64:8:32:32" "1:32:4:32:32" > $O/sweep128-c1.log 2>&1
~/fa_row_sweep.sh 256 "32:256:2:64:64" "32:512:2:64:128" "32:256:1:32:64" "32:256:3:32:64" "32:256:2:32:32" "32:256:2:64:32" > $O/sweep256b.log 2>&1
cp /tmp/fattn-tile.cuh.patched4.bak /tmp/fattn-tile.cuh.patched4
cp /tmp/fattn-tile.cuh.seven ggml/src/ggml-cuda/fattn-tile.cuh
log "sweeps done"
# --- CPU: the q8-off variant and the tracer
cp ggml/src/ggml-cuda/mmvq-rdna1-f32.cu /tmp/mmvq-rdna1-f32.cu.seven
python3 ~/apply_q8off.py ggml/src/ggml-cuda | tee -a $O/log
cmake -S . -B build-hip-q8off -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1
nice -n 5 cmake --build build-hip-q8off -j 7 --target llama-bench test-backend-ops > $O/build-q8off.log 2>&1 && log "build q8off ok" || log "build q8off FAILED"
cp /tmp/mmvq-rdna1-f32.cu.seven ggml/src/ggml-cuda/mmvq-rdna1-f32.cu
(cd ~ && nice -n 5 hipcc -O2 -fPIC -shared kerntrace.cpp -o libkerntrace.so -lroctracer64 -ldl 2>&1 | grep -i error | head -3)
log "tracer rebuilt"
sleep 180
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
# --- GPU: env knobs on the final build
bench() { env $1 LD_LIBRARY_PATH=$L timeout -k 20 900 $F/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -a "pp512\|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
  for k in "X=1" "HSA_ENABLE_INTERRUPT=0" "GGML_CUDA_DISABLE_GRAPHS=1" "HSA_ENABLE_INTERRUPT=0 GGML_CUDA_DISABLE_GRAPHS=1" "HIP_FORCE_DEV_KERNARG=1" "GPU_MAX_HW_QUEUES=1"; do
    log "p$pass $m [$k]: $(bench "$k" $m)"
  done
done; done
# --- GPU: q8 off the float kernel, the 8B, against the final build
tg() { LD_LIBRARY_PATH=$L timeout -k 20 900 $M/$1/bin/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2 3; do log "p$pass 8B tg64: final $(tg build-hip-final) | q8off $(tg build-hip-q8off)"; done
# --- GPU: tracer with the gap histogram
for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
  KERNTRACE_OUT=$O/trace-$m-tg.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $F/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 1 > /dev/null 2>&1
  log "gaps $m: $(grep -a 'gaps between' $O/trace-$m-tg.txt | cut -c1-260)"
  KERNTRACE_OUT=$O/trace-$m-tg-nograph.txt GGML_CUDA_DISABLE_GRAPHS=1 LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $F/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 1 > /dev/null 2>&1
  log "gaps $m, graphs off: $(grep -a 'gaps between' $O/trace-$m-tg-nograph.txt | cut -c1-260)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
