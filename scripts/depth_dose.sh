#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the first repetition's variability track depth, or was that two points?
#
# logs/decode-variance-clock-2026-09-22 compared depth 0 against depth 16128 and found the first
# repetition low at both, but low reproducibly at 0 (ratios spanning 0.023, rep1 varying 1.59 percent
# between invocations) and low erratically at 16128 (0.179 and 7.71). That is two points, six
# invocations each, and four effects on this board have looked convincing at that size and not survived.
#
# A sweep is a harder thing to get by accident than a contrast. If depth is what makes the first
# repetition erratic, the spread of the ratio should grow with depth across five of them, not jump
# between two. If it does not, the two-point contrast was a coincidence and the note comes out.
#
# Round-robin across depths, not blocked, because anything blocked on this board picks up drift,
# and the ORDER OF THE DEPTHS ALTERNATES between passes. Always running them ascending would confound
# depth with position in the pass and so with temperature: the earlier runs sat at 60 to 63 C at depth 0
# and 68 to 69 C at depth 16128 because the deep one came last. That is well below the 93 C where
# the governor acts (logs/fault-repro-2026-09-22), so it probably would not matter, but alternating
# costs nothing and removes the argument.
# Usage: depth_dose.sh [passes]
set -u
N=${1:-4}
O=~/depth-dose; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N passes over five depths, edge $(temp), available $(free -m | awk '/^Mem:/{print $7}') MiB"
for p in $(seq 1 "$N"); do
  if [ $(( p % 2 )) -eq 1 ]; then ORDER="0 2048 4096 8192 16128"; else ORDER="16128 8192 4096 2048 0"; fi
  log "pass $p order: $ORDER"
  for d in $ORDER; do
    j="$O/d${d}_p${p}.jsonl"
    LD_LIBRARY_PATH=$L timeout -k 30 2400 "$HIP/llama-bench" -m "$M" -lm mmap -ngl 99 -fa 1 \
        -p 0 -n 8 -d $d -r 8 -o jsonl > "$j" 2>/dev/null
    s=$(python3 -c "
import json
try:
    o=json.loads(open('$j').read().strip().split(chr(10))[-1])
    v=o['samples_ts']; w=sum(v[2:])/len(v[2:])
    print('rep1=%.2f warm=%.2f ratio=%.3f reps=%s' % (v[0], w, v[0]/w, ','.join('%.2f'%x for x in v)))
except Exception as e: print('FAIL', e)
")
    log "p$p d=$d $s edge=$(temp)"
  done
  log "pass $p done, edge $(temp)"
done
log done; touch "$O/DONE"
