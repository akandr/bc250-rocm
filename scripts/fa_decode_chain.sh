#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment I: one-token flash attention on RDNA1 through the vector kernel instead of the tile kernel.
# Builds build-hip-gdn (which already carries the GDN and concat changes) with the chooser change; the FA
# lines of all six graphs replayed with the vector choice on and off; decode at depth 4096 and 0 on the
# 1.5B, 8B, deepseek-14B, MoE and 27B with both settings, interleaved; correctness of FLASH_ATTN_EXT.
# Waits for the CLR build (CPU) so the GPU measurements run on a cool package.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/fa-decode; B=$M/build-hip-gdn/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "ALL DONE" ~/clr-src/log 2>/dev/null; do sleep 60; done
sleep 120
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cp ggml/src/ggml-cuda/fattn.cu /tmp/fattn.cu.orig
python3 ~/apply_rdna1_fa_vec_decode.py ggml/src/ggml-cuda | tee -a $O/log
nice -n 5 cmake --build build-hip-gdn -j 7 --target test-backend-ops llama-bench > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A3 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 150
for v in 1 0; do
  GGML_FA_VEC_DECODE=$v LD_LIBRARY_PATH=$L timeout -k 30 1800 $B/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-fa-v$v.log
  log "FLASH_ATTN_EXT correctness, vec_decode=$v: OK $(grep -ac 'OK' $O/tbo-fa-v$v.log) FAIL $(grep -ac 'FAIL' $O/tbo-fa-v$v.log) NOT_SUPPORTED $(grep -ac 'not supported' $O/tbo-fa-v$v.log)"
done
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  for v in 1 0; do
    GGML_FA_VEC_DECODE=$v LD_LIBRARY_PATH=$L timeout -k 30 900 $B/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$m-v$v.log 2>&1
  done
  log "$m FA us: vec $(grep -ao '[0-9.]* us/run' $O/fa-$m-v1.log | tr '\n' ' ')| tile $(grep -ao '[0-9.]* us/run' $O/fa-$m-v0.log | tr '\n' ' ')"
done
tgd() { GGML_FA_VEC_DECODE=$1 LD_LIBRARY_PATH=$L timeout -k 20 1500 $B/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 32 -d $3 -r 3 2>/dev/null | grep -a "tg32" | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
    for d in 4096 0; do
      log "p$pass $m d=$d tg32: tile $(tgd 0 $m $d) | vec $(tgd 1 $m $d)"
    done
  done
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
