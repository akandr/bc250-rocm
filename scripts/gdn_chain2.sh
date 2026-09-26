#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment H, final form: GDN lanes chosen by token count (8 at prefill, one column per wave at one
# token); rebuild build-hip-gdn, correctness, per-op replay (default env) of the linear-attention lines,
# pp512/tg64 A/B vs build-hip-f32iq, MoE gate reference on build-hip-f32mv, and a kerntrace debug run.
# Waits for the FA replay.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/gdn-chain2; BASE=build-hip-f32iq; NEW=build-hip-gdn
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "FA DONE" ~/fa-remainder/log 2>/dev/null; do sleep 60; done
sleep 30
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cp /tmp/gated_delta_net.cu.orig ggml/src/ggml-cuda/gated_delta_net.cu && python3 ~/apply_rdna1_gdn_lanes.py ggml/src/ggml-cuda | tee -a $O/log
nice -n 5 cmake --build $NEW -j 7 --target test-backend-ops llama-bench llama-cli llama-perplexity > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A3 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 90
LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$NEW/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0 2>&1 | grep -a "GATED_DELTA_NET" > $O/tbo-gdn.log
log "GDN correctness: OK $(grep -ac 'OK' $O/tbo-gdn.log) FAIL $(grep -ac 'FAIL' $O/tbo-gdn.log)"
for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
  for p in 1 2; do LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$NEW/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-gdn.txt -b ROCm0 > $O/ops-$m-$NEW-p$p.log 2>&1; done
  log "$m replayed"
done
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -aE "pp512|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do
  for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
    log "p$pass $m $BASE: $(pp $BASE $m)"
    log "p$pass $m $NEW: $(pp $NEW $m)"
  done
done
log "MoE gate build-hip-f32mv (reference): $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/build-hip-f32mv/bin/llama-perplexity -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 3 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "MoE gate $NEW: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$NEW/bin/llama-perplexity -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 3 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
# kerntrace debug: small run, stderr kept
KERNTRACE_OUT=$O/kt-debug.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 300 $M/$NEW/bin/test-backend-ops perf -o ADD -b ROCm0 > $O/kt-debug.stderr 2>&1
log "kerntrace debug: file $(ls -la $O/kt-debug.txt 2>&1 | awk '{print $5}') bytes; stderr tail: $(tail -2 $O/kt-debug.stderr | tr '\n' ' ' | cut -c1-200)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
