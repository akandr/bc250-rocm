#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Is the decode-at-depth variance a start-of-measurement transient?
#
# The measurement that carries it generates EIGHT tokens after a 16128-token prefill. At about 28 t/s
# that is under a second of measured work at the end of a two-and-a-half minute invocation, so anything
# that takes a moment to settle once decode begins is most of what is being timed. Five candidates have
# been eliminated for this variance (CPU pinning, page cache, hardware queue, ASLR, duty cycle) and they
# were all things that differ per process; a transient at the start of the timed window is not.
#
# The test needs no new instrument. Lengthen the window: if the variance is a transient, generating 64
# tokens instead of 8 dilutes it and the spread collapses. If it is a property of the process, the two
# lengths spread alike. The prefill dominates the wall clock either way, so 64 costs little more than 8.
#
# Interleaved, because anything measured in blocks on this board picks up drift.
# Usage: window_length.sh [pairs]
set -u
N=${1:-10}
O=~/window-length; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N pairs, edge $(temp), throttles so far $(thr), available $(free -m | awk '/^Mem:/{print $7}') MiB"

for p in $(seq 1 "$N"); do
  for n in 8 64; do
    # jsonl keeps every repetition, so a slow invocation can be told from a slow first repetition
    j="$O/n${n}_p${p}.jsonl"
    LD_LIBRARY_PATH=$L timeout -k 20 900 "$HIP/llama-bench" -m "$M" -ngl 99 -fa on \
        -p 0 -n $n -d 16128 -r 3 -o jsonl > "$j" 2>/dev/null
    r=$(python3 -c "
import json,sys
try:
    d=json.loads(open('$j').read().strip().split(chr(10))[-1])
    print('%.2f %s' % (d['avg_ts'], ','.join('%.2f'%x for x in d['samples_ts'])))
except Exception: print('FAIL -')
")
    log "p$p n=$n tg=$r edge=$(temp) throttles=$(thr)"
  done
done
log done; touch "$O/DONE"
