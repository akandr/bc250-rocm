#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment I, final form: the narrow rule (RDNA1, one token, gqa_ratio_eff == 1 -> vector kernel).
# Rebuild build-hip-gdn from the original fattn.cu, correctness, FA lines for the 14B, 8B and 1.5B, and
# tg32 at depth 4096 and 0 for deepseek-14B and the 8B, two passes, against the numbers of the
# experiment build. Waits for the D=256 chain.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/fa-vec-final; B=$M/build-hip-gdn/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "CHAIN DONE" ~/fa-d256/log 2>/dev/null; do sleep 60; done
sleep 60
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cp /tmp/fattn.cu.orig ggml/src/ggml-cuda/fattn.cu && python3 ~/apply_rdna1_fa_vec_decode.py ggml/src/ggml-cuda | tee -a $O/log
nice -n 5 cmake --build build-hip-gdn -j 7 --target test-backend-ops llama-bench > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A2 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 120
LD_LIBRARY_PATH=$L timeout -k 30 1800 $B/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-fa.log
log "FLASH_ATTN_EXT correctness: OK $(grep -ac 'OK' $O/tbo-fa.log) FAIL $(grep -ac 'FAIL' $O/tbo-fa.log)"
for m in deepseek-r1-14b qwen3-14b qwen3-8b-q8_0 qwen2.5-1.5b-q4km; do
  LD_LIBRARY_PATH=$L timeout -k 30 900 $B/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$m.log 2>&1
  log "$m FA us: $(grep -ao '[0-9.]* us/run' $O/fa-$m.log | tr '\n' ' ')"
done
tgd() { LD_LIBRARY_PATH=$L timeout -k 20 1500 $B/llama-bench -m /opt/models/$1.gguf -ngl 99 -fa 1 -p 0 -n 32 -d $2 -r 3 2>/dev/null | grep -a "tg32" | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do
  for m in deepseek-r1-14b qwen3-8b-q8_0; do
    log "p$pass $m tg32 d=4096: $(tgd $m 4096)   d=0: $(tgd $m 0)"
  done
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
