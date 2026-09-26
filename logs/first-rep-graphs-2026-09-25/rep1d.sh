#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Is the fixed first-repetition cost HIP graph instantiation?
export LC_ALL=C
B=/home/akandr/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen2.5-1.5b-q4km.gguf
echo "graphs n_gen inv rep1 mean_rest ratio"
for g in on off; do
  for inv in 1 2 3 4 5 6 7 8 9 10; do
    ENV=""; [ "$g" = off ] && ENV="GGML_CUDA_DISABLE_GRAPHS=1"
    env $ENV HSA_ENABLE_SDMA=0 $B/llama-bench -m $M -mmp 0 -ngl 99 -fa on -p 0 -n 8 -r 6 -o jsonl 2>/dev/null \
      | python3 -c "
import sys,json
g='$g'; inv='$inv'
for line in sys.stdin:
    try: d=json.loads(line)
    except: continue
    s=d.get('samples_ts') or []
    if len(s)>1:
        rest=sum(s[1:])/len(s[1:])
        print(f'{g:>6} {8:>5} {inv:>3} {s[0]:8.2f} {rest:9.2f} {s[0]/rest:6.3f}')
"
    sleep 8
  done
done
echo DONE_REP1D
