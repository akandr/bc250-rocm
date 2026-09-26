#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Decode A/B: the final four-patch build (build-hip-fatile, MMVQ on the GENERIC parameter table that
# RDNA1 falls into) against the same tree with RDNA1 routed to the RDNA2 MMVQ table (build-hip-mmvq).
#
# Per model and pass: tg128 with -fa on, its own invocation, builds interleaved. Then the n=1 matmul
# shapes replayed through test-backend-ops on both builds, the 1.5B gate on the new build (decode
# does not enter perplexity, so it is a smoke test only), and decode text on the new build.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=${OUT:-~/mmvq-ab}
BASE=${BASE:-build-hip-fatile}; NEW=${NEW:-build-hip-mmvq}
mkdir -p $O; cd $O || exit 1
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
log "clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')  governor $(systemctl is-active oberon-governor)  ollama $(systemctl is-active ollama)"
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1200 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 3 2>/dev/null | grep -a tg128 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do
  for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b qwen3.6-35b-a3b-iq2m; do
    for b in $BASE $NEW; do
      log "p$pass $m $b tg128: $(tg $b $m)"
    done
  done
done
for b in $BASE $NEW; do
  $M/$b/bin/test-backend-ops perf --test-file ~/opgraph/ops-1.5b-pp2048.txt -b ROCm0 > $O/ops-$b.log 2>&1
  log "$b n=1 matvec ops: $(grep -a -A1 'MUL_MAT(.*ne=\[[0-9]*,1,1,1\]' $O/ops-$b.log | grep -ao '[0-9.]* us/run' | tr '\n' ' ')"
done
log "1.5B gate $NEW: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$NEW/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (final build 8.9498)"
log "decode text $NEW: $(LD_LIBRARY_PATH=$L timeout -k 10 300 $M/$NEW/bin/llama-cli -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa on -c 4096 -n 24 --seed 1 -no-cnv -st -p 'The capital of France is' < /dev/null 2>&1 | grep -a -A2 'capital of France' | grep -av 'capital of France' | tr '\n' ' ' | cut -c1-120)"
log "clock after $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log DONE
