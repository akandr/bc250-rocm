#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Reproduce both of August's measurements from the same data.
#
# logs/decode-variance-state-2026-08-21 ran two different commands and compared them. Its three arms,
# the ones that carry the 7.7 to 8.8 percent variance, used -r 1: one repetition per invocation. Its
# within-invocation check, the one that spread only 0.43, used -r 8. The conclusion drawn was that the
# variance lives between processes, not inside them.
#
# If the first repetition of a timed window is slower and more variable than the rest, as
# logs/decode-variance-clock-2026-09-22 suggests, then those two commands were not measuring
# between-process against within-process. They were measuring a cold repetition against an average of
# one cold and seven warm ones, and the difference between them is the warm-up, not the process.
#
# One command settles it. -r 8, jsonl, so every repetition survives:
#   - the spread of repetition 1 across invocations is August's -r 1 arms
#   - the spread within each invocation across all 8 is August's -r 8 check
#   - the spread of repetitions 2 to 8 is what the check would have said without the cold one
# If the third is much tighter than the first, the clue that started this was an artefact of the flags.
# If the within-invocation spread comes back at about 0.43 as August measured, it is not, and the
# warm-up seen yesterday is something that arrived since.
# Usage: first_rep.sh [invocations]
set -u
N=${1:-5}
O=~/first-rep; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
thr () { sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat; }

if [ "$(systemctl is-active ollama 2>/dev/null)" = active ]; then sudo systemctl stop ollama; sleep 10; fi
log "start, $N invocations of -r 8, edge $(temp), throttles so far $(thr)"
# -lm mmap and -fa 1 exactly as scripts/decode_variance_process_state.sh had them
for i in $(seq 1 "$N"); do
  j="$O/r8_$i.jsonl"
  LD_LIBRARY_PATH=$L timeout -k 30 2400 "$HIP/llama-bench" -m "$M" -lm mmap -ngl 99 -fa 1 \
      -p 0 -n 8 -d 16128 -r 8 -o jsonl > "$j" 2>/dev/null
  s=$(python3 -c "
import json
try:
    d=json.loads(open('$j').read().strip().split(chr(10))[-1])
    v=d['samples_ts']
    m=sum(v)/len(v); sd=(sum((x-m)**2 for x in v)/len(v))**.5
    print('avg=%.2f sd=%.2f reps=%s' % (m, sd, ','.join('%.2f'%x for x in v)))
except Exception as e: print('FAIL', e)
")
  log "i$i $s edge=$(temp) throttles=$(thr)"
done
log done; touch "$O/DONE"
