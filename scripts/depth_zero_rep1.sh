#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the slow first repetition need the depth machinery?
#
# With -d N, llama-bench prefills on the first repetition and restores a saved state on the rest, so
# repetition 1 enters its timed window out of a GPU prefill and the others out of a host state upload.
# That is one candidate for why repetition 1 reads low (logs/decode-variance-clock-2026-09-22).
#
# At -d 0 neither path runs: llama_memory_clear, then straight to timing, identically every repetition.
# So this separates that candidate from the rest without needing to identify any of them. If repetition
# 1 still reads low at depth 0, prefill-versus-restore is not what causes it. If it does not, the depth
# machinery is implicated and the next question is which half.
#
# Interleaved, same model, same everything else. Depth 0 invocations are quick since there is no prefill.
# Usage: depth_zero_rep1.sh [pairs]
set -u
N=${1:-6}
O=~/d0-rep1; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N pairs, edge $(temp), available $(free -m | awk '/^Mem:/{print $7}') MiB"
for p in $(seq 1 "$N"); do
  for d in 0 16128; do
    j="$O/d${d}_p${p}.jsonl"
    LD_LIBRARY_PATH=$L timeout -k 30 2400 "$HIP/llama-bench" -m "$M" -lm mmap -ngl 99 -fa 1 \
        -p 0 -n 8 -d $d -r 8 -o jsonl > "$j" 2>/dev/null
    s=$(python3 -c "
import json
try:
    d=json.loads(open('$j').read().strip().split(chr(10))[-1])
    v=d['samples_ts']
    warm=v[2:]; m=sum(warm)/len(warm)
    print('rep1=%.2f rep2=%.2f warm_mean=%.2f rep1/warm=%.3f reps=%s'
          % (v[0], v[1], m, v[0]/m, ','.join('%.2f'%x for x in v)))
except Exception as e: print('FAIL', e)
")
    log "p$p d=$d $s edge=$(temp)"
  done
done
log done; touch "$O/DONE"
