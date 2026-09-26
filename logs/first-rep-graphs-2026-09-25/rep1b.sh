#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Is the slow first repetition a fixed start-up cost? If so its share shrinks as the repetition grows.
export LC_ALL=C
B=/home/akandr/llama-master/build-hip-pkf16/bin
M=/opt/models/qwen2.5-1.5b-q4km.gguf
echo "n_gen prewarm inv rep1 mean_rest ratio clock_at_start"
for n in 8 32 128; do
  for pw in 0 1; do
    for inv in 1 2 3 4; do
      if [ "$pw" = 1 ]; then timeout 25 $B/llama-bench -m $M -mmp 0 -ngl 99 -fa on -p 0 -n 512 -r 1 >/dev/null 2>&1; fi
      clk=$(cat /sys/class/drm/card*/device/pp_dpm_sclk | grep '\*' | grep -oE '[0-9]+Mhz')
      HSA_ENABLE_SDMA=0 $B/llama-bench -m $M -mmp 0 -ngl 99 -fa on -p 0 -n $n -r 6 -o jsonl 2>/dev/null \
        | python3 -c "
import sys,json
n='$n'; pw='$pw'; inv='$inv'; clk='$clk'
for line in sys.stdin:
    try: d=json.loads(line)
    except: continue
    s=d.get('samples_ts') or []
    if len(s)>1:
        rest=sum(s[1:])/len(s[1:])
        print(f'{n:>5} {pw:>7} {inv:>3} {s[0]:8.2f} {rest:9.2f} {s[0]/rest:6.3f} {clk}')
"
      sleep 8
    done
  done
done
echo DONE_REP1B
