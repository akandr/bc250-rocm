#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The two models the four-patch campaign did not move: deepseek-r1-14B and qwen3-14B, both 40 query
# heads over 8 KV heads, which llama.cpp routes to the ncols2 = 1 flash-attention variants. Compares
# the production three-patch build against the build whose D=128 row 64 was retuned for that variant.
#
# Per model and pass: pp512 and pp2048 with -fa on, each its own invocation, builds interleaved;
# then the perplexity gate on each build. Nothing else runs on the board meanwhile.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=${OUT:-~/ab14b}
BASE=${BASE:-build-hip-f44}; NEW=${NEW:-build-hip-fatile}
mkdir -p $O; cd $O || exit 1
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
log "clock $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')  governor $(systemctl is-active oberon-governor)  ollama $(systemctl is-active ollama)"
bench() { LD_LIBRARY_PATH=$L timeout -k 20 1200 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p $3 -n 0 -r 3 2>/dev/null | grep -a "pp$3" | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for m in deepseek-r1-14b qwen3-14b; do
  for pass in 1 2; do
    for b in $BASE $NEW; do
      log "$m p$pass $b pp512:  $(bench $b $m 512)"
      log "$m p$pass $b pp2048: $(bench $b $m 2048)"
    done
  done
  for b in $BASE $NEW; do
    log "$m gate $b: $(LD_LIBRARY_PATH=$L timeout -k 30 3600 $M/$b/bin/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
  done
done
log "clock after $(cat /sys/class/drm/card*/device/pp_dpm_sclk | tr -d '\n')"
log DONE
