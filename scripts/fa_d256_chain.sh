#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment J: the D=256 flash-attention tile on RDNA1 (MoE and 27B prefill). Three row builds, each
# measured with the 16-column cap on and off: B1 row16 = 512:2:32:64 (zero-spill instances), B2 = the
# current rows, B3 row32 = 256:2:32:64 (near zero spill for the MoE's 4x8 instance). FA lines replayed,
# correctness, then pp512/pp2048 end to end for baseline, B1 cap on, B3 cap off. Ends with kernel traces.
# Waits for the CLR rebuild (CPU) to finish.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/fa-d256; B=$M/build-hip-gdn/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "CLR2 DONE" ~/clr-src/log2 2>/dev/null; do sleep 60; done
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.final
build() { # $1 tag, $2 row16, $3 row32
  cp /tmp/fattn-tile.cuh.final ggml/src/ggml-cuda/fattn-tile.cuh
  python3 ~/apply_rdna1_fa_d256.py ggml/src/ggml-cuda "$2" "$3" | tee -a $O/log
  nice -n 5 cmake --build build-hip-gdn -j 7 --target test-backend-ops llama-bench > $O/build-$1.log 2>&1 && log "build $1 ok" || { log "build $1 FAILED"; grep -m3 -A2 "error:" $O/build-$1.log | tee -a $O/log; return 1; }
  sleep 120
  for cap in 1 0; do
    for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
      GGML_FA_D256_CAP16=$cap LD_LIBRARY_PATH=$L timeout -k 30 900 $B/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$1-cap$cap-$m.log 2>&1
      log "$1 cap16=$cap $m FA us: $(grep -ao '[0-9.]* us/run' $O/fa-$1-cap$cap-$m.log | tr '\n' ' ')"
    done
  done
  GGML_FA_D256_CAP16=1 LD_LIBRARY_PATH=$L timeout -k 30 1800 $B/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-$1.log
  log "$1 cap16=1 FLASH_ATTN_EXT correctness: OK $(grep -ac 'OK' $O/tbo-$1.log) FAIL $(grep -ac 'FAIL' $O/tbo-$1.log)"
}
build B2 "512:3:32:128" "512:3:32:128"      # current rows; cap off = baseline
cp -a $B/llama-bench $O/llama-bench-B2
build B1 "512:2:32:64"  "512:3:32:128"
cp -a $B/llama-bench $O/llama-bench-B1
build B3 "512:3:32:128" "256:2:32:64"
cp -a $B/llama-bench $O/llama-bench-B3
pp() { GGML_FA_D256_CAP16=$1 LD_LIBRARY_PATH=$L timeout -k 20 1800 $2 -m /opt/models/$3.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp512|pp2048" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do
  for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
    log "p$pass $m baseline (B2, cap off): $(pp 0 $O/llama-bench-B2 $m)"
    log "p$pass $m B1 cap on:  $(pp 1 $O/llama-bench-B1 $m)"
    log "p$pass $m B3 cap off: $(pp 0 $O/llama-bench-B3 $m)"
  done
done
# leave the tree on the current rows (cap code present, env default on -> the final decision is applied later)
cp /tmp/fattn-tile.cuh.final ggml/src/ggml-cuda/fattn-tile.cuh
# kernel traces with the interposing tracer, front-page build
for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
  KERNTRACE_OUT=$O/trace-$m-tg.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $M/build-hip-f32iq/bin/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 1 > $O/trace-$m-tg.bench 2>&1
  KERNTRACE_OUT=$O/trace-$m-pp.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $M/build-hip-f32iq/bin/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 512 -n 0 -r 1 > $O/trace-$m-pp.bench 2>&1
  log "trace $m: $(head -1 $O/trace-$m-tg.txt 2>/dev/null | cut -c1-110) | pp: $(head -1 $O/trace-$m-pp.txt 2>/dev/null | cut -c1-110)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
